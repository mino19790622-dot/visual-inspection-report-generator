# app/rag/retriever.py
"""RAG standards retriever — embed inspection standards, retrieve via similarity search.

Uses Alibaba Cloud DashScope text-embedding-v2 (OpenAI-compatible API)
with ChromaDB as local persistent vector store.
"""

import os
import re
from openai import OpenAI
from dotenv import load_dotenv
import chromadb

load_dotenv()


class StandardsRetriever:
    """Index inspection standards into ChromaDB, retrieve relevant sections."""

    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    EMBED_MODEL = "text-embedding-v2"
    CHUNK_SIZE = 600        # characters per chunk
    CHUNK_OVERLAP = 100     # overlap between chunks

    def __init__(self, standards_dir: str = "data/standards",
                 db_path: str = ".chroma_db"):
        self.openai = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=self.BASE_URL,
        )
        self.chroma = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma.get_or_create_collection(
            name="inspection_standards",
            metadata={"description": "Visual inspection standards knowledge base"},
        )
        self._index_standards(standards_dir)

    # ---------- Embedding ----------
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts via DashScope (batch of 25 per call)."""
        all_embeddings = []
        for i in range(0, len(texts), 25):
            batch = texts[i:i + 25]
            resp = self.openai.embeddings.create(
                model=self.EMBED_MODEL,
                input=batch,
            )
            all_embeddings.extend([d.embedding for d in resp.data])
        return all_embeddings

    # ---------- Chunking ----------
    def _chunk_text(self, text: str) -> list[str]:
        """Split markdown text into overlapping chunks, keeping headers."""
        # Split by markdown headers (## or ###)
        sections = re.split(r"\n(?=##)", text.strip())
        chunks = []
        for section in sections:
            if len(section) <= self.CHUNK_SIZE:
                chunks.append(section.strip())
                continue
            # Split long sections with overlap
            start = 0
            while start < len(section):
                end = start + self.CHUNK_SIZE
                chunk = section[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                start = end - self.CHUNK_OVERLAP
        return [c for c in chunks if len(c) > 50]  # filter tiny fragments

    # ---------- Indexing ----------
    def _index_standards(self, standards_dir: str):
        """Load markdown standards, chunk, embed, and store in ChromaDB."""
        if self.collection.count() > 0:
            return  # already indexed

        docs, ids, metadatas = [], [], []
        for filename in sorted(os.listdir(standards_dir)):
            if not filename.endswith(".md"):
                continue
            path = os.path.join(standards_dir, filename)
            with open(path) as f:
                content = f.read()

            std_name = content.split("\n")[0].lstrip("# ").strip()
            for i, chunk in enumerate(self._chunk_text(content)):
                docs.append(chunk)
                ids.append(f"{filename}::{i}")
                metadatas.append({
                    "source": filename,
                    "standard": std_name,
                    "chunk_index": i,
                })

        if not docs:
            raise ValueError(f"No standards found in {standards_dir}")

        print(f"  Indexing {len(docs)} chunks from "
              f"{len(set(m['source'] for m in metadatas))} standards...")
        embeddings = self._embed_batch(docs)
        self.collection.add(
            ids=ids,
            documents=docs,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    # ---------- Retrieval ----------
    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        """Retrieve top-k relevant standard chunks for a query."""
        query_emb = self._embed_batch([query])[0]
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=k,
        )

        retrieved = []
        for i in range(len(results["ids"][0])):
            retrieved.append({
                "text": results["documents"][0][i],
                "standard": results["metadatas"][0][i]["standard"],
                "source": results["metadatas"][0][i]["source"],
                "distance": round(results["distances"][0][i], 4),
            })
        return retrieved
