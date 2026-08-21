# tests/test_rag.py
"""Unit + integration tests for the RAG retriever.

- _chunk_text: pure string logic, tested directly.
- retrieve(): tested against a real in-memory ChromaDB (EphemeralClient)
  with deterministic hash-based embeddings — no DashScope calls, no network.
"""

import chromadb
import pytest

from app.rag.retriever import StandardsRetriever


@pytest.fixture
def retriever():
    r = StandardsRetriever.__new__(StandardsRetriever)
    r.CHUNK_SIZE = 100
    r.CHUNK_OVERLAP = 20
    return r


# ---------------------------- chunking ---------------------------- #
class TestChunking:
    def test_split_by_headers(self, retriever):
        text = ("# Std A\n\n" + "x" * 80 + "\n\n## Section B\n\n" + "y" * 80)
        chunks = retriever._chunk_text(text)
        assert len(chunks) == 2
        assert chunks[0].startswith("# Std A")
        assert chunks[1].startswith("## Section B")

    def test_long_section_split_with_overlap(self, retriever):
        text = "## Long\n" + "a" * 250  # 250 chars -> chunks of ~100 with 20 overlap
        chunks = retriever._chunk_text(text)
        assert len(chunks) >= 3
        # each chunk (except the last) is CHUNK_SIZE long
        assert all(len(c) <= 100 for c in chunks)
        # overlap: chunk i+1 starts 80 chars after chunk i
        assert chunks[1][:20] == chunks[0][80:100]

    def test_tiny_fragments_filtered(self, retriever):
        text = "## A\n" + "b" * 70 + "\n\n## C\n" + "d" * 10  # 10 chars -> dropped
        chunks = retriever._chunk_text(text)
        assert len(chunks) == 1

    def test_empty_text(self, retriever):
        assert retriever._chunk_text("   ") == []


# ---------------------------- retrieval (real Chroma, fake embeddings) ---------------------------- #
def _stable_hash(word: str) -> int:
    return sum(ord(c) * (i + 1) for i, c in enumerate(word)) % 16


def _fake_embed(texts):
    """Deterministic bag-of-words hash embedding (16-dim)."""
    import re
    out = []
    for t in texts:
        v = [0.0] * 16
        for w in re.findall(r"[a-z]+", t.lower()):
            v[_stable_hash(w)] += 1.0
        out.append(v)
    return out


@pytest.fixture
def indexed_retriever():
    r = StandardsRetriever.__new__(StandardsRetriever)
    r.chroma = chromadb.EphemeralClient()
    r.collection = r.chroma.get_or_create_collection(
        name="test_standards",
        metadata={"description": "test", "hnsw:space": "cosine"},
    )
    docs = [
        ("construction_site_safety.md",
         "construction site safety requirements mandate helmets and "
         "high visibility vests for all workers near heavy machinery"),
        ("road_surface_inspection.md",
         "road surface inspection grading criteria for pavement rutting "
         "and pothole severity thresholds on motorways"),
    ]
    r.collection.add(
        ids=[src for src, _ in docs],
        documents=[text for _, text in docs],
        embeddings=_fake_embed([text for _, text in docs]),
        metadatas=[{"source": src, "standard": src, "chunk_index": 0}
                   for src, _ in docs],
    )
    r._embed_batch = _fake_embed  # bypass DashScope
    return r


class TestRetrieve:
    def test_top_result_matches_query_topic(self, indexed_retriever):
        hits = indexed_retriever.retrieve(
            "construction site safety helmets workers", k=2)
        assert len(hits) == 2
        assert hits[0]["source"] == "construction_site_safety.md"
        # results are sorted by ascending cosine distance
        assert hits[0]["distance"] <= hits[1]["distance"]

    def test_k_controls_result_count(self, indexed_retriever):
        assert len(indexed_retriever.retrieve("road pavement", k=1)) == 1

    def test_result_schema(self, indexed_retriever):
        hit = indexed_retriever.retrieve("road surface", k=1)[0]
        assert set(hit) == {"text", "standard", "source", "distance"}
        assert isinstance(hit["distance"], float)


# ---------------------------- indexing (real Chroma, fake embeddings) ---------------------------- #
class TestIndexing:
    def _make_retriever(self, tmp_path):
        r = StandardsRetriever.__new__(StandardsRetriever)
        r.chroma = chromadb.PersistentClient(path=str(tmp_path / "db"))
        r.collection = r.chroma.get_or_create_collection(
            name="inspection_standards",
            metadata={"description": "test", "hnsw:space": "cosine"})
        r._embed_batch = _fake_embed
        return r

    def _make_standards_dir(self, tmp_path):
        std_dir = tmp_path / "standards"
        std_dir.mkdir()
        (std_dir / "a.md").write_text(
            "# Standard A\n\n" + "construction safety helmet rules. " * 4)
        (std_dir / "b.md").write_text(
            "# Standard B\n\n" + "road pavement pothole severity grading. " * 4)
        (std_dir / "notes.txt").write_text("ignored: not markdown")
        return std_dir

    def test_index_standards_chunks_and_stores(self, tmp_path):
        r = self._make_retriever(tmp_path)
        r._index_standards(str(self._make_standards_dir(tmp_path)))
        stored = r.collection.get(include=["metadatas"])
        assert r.collection.count() == 2  # one chunk per (short) standard
        sources = {m["source"] for m in stored["metadatas"]}
        assert sources == {"a.md", "b.md"}  # .txt file skipped

    def test_reindex_is_noop(self, tmp_path):
        r = self._make_retriever(tmp_path)
        std_dir = str(self._make_standards_dir(tmp_path))
        r._index_standards(std_dir)
        count = r.collection.count()
        r._index_standards(std_dir)  # second call -> early return
        assert r.collection.count() == count

    def test_empty_dir_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        r = self._make_retriever(tmp_path)
        with pytest.raises(ValueError, match="No standards found"):
            r._index_standards(str(empty))
