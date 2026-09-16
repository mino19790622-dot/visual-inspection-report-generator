"""Deterministic tests for the retrieval-side metrics.

No network and no LLM: these cover the ranking math, the truth loader, and the
fingerprint guard. The end-to-end retrieval run lives in
``eval/retrieval_eval.py`` and needs the embedding model, so it is a manual /
CI-scheduled job rather than a unit test.
"""
from __future__ import annotations

import pytest

from eval import retrieval_eval as re_

RANKED = ["a.md::0", "b.md::1", "c.md::2", "d.md::3", "e.md::4"]


class TestRecallAtK:
    def test_partial_then_full(self):
        truth = ["c.md::2", "e.md::4"]
        assert re_.recall_at_k(RANKED, truth, 1) == 0.0
        assert re_.recall_at_k(RANKED, truth, 3) == 0.5
        assert re_.recall_at_k(RANKED, truth, 5) == 1.0

    def test_k_beyond_ranking_length_saturates(self):
        assert re_.recall_at_k(RANKED, ["a.md::0"], 99) == 1.0

    def test_repeated_truth_id_is_not_double_counted(self):
        # A duplicate in the truth list must not inflate the denominator.
        assert re_.recall_at_k(RANKED, ["a.md::0", "a.md::0"], 1) == 1.0

    def test_empty_truth_raises_rather_than_scoring_zero(self):
        # Scoring an empty truth set as recall 0 would invent a failure.
        with pytest.raises(ValueError):
            re_.recall_at_k(RANKED, [], 5)

    def test_no_overlap_is_zero(self):
        assert re_.recall_at_k(RANKED, ["z.md::9"], 5) == 0.0


class TestRankOfFirstRelevant:
    def test_finds_one_based_rank(self):
        assert re_.rank_of_first_relevant(RANKED, ["c.md::2"]) == 3

    def test_lowest_rank_wins(self):
        assert re_.rank_of_first_relevant(RANKED, ["e.md::4", "a.md::0"]) == 1

    def test_absent_is_none(self):
        assert re_.rank_of_first_relevant(RANKED, ["z.md::9"]) is None

    def test_empty_truth_is_none(self):
        # Empty truth must not be read as "found at rank 1".
        assert re_.rank_of_first_relevant(RANKED, []) is None


class TestTruthOf:
    def test_reads_chunk_ids(self):
        assert re_.truth_of({"retrieval_truth": {"chunk_ids": ["x.md::1"]}}) == ["x.md::1"]

    def test_missing_shapes_are_empty(self):
        assert re_.truth_of({}) == []
        assert re_.truth_of({"retrieval_truth": {}}) == []
        assert re_.truth_of({"retrieval_truth": {"chunk_ids": None}}) == []


class TestFingerprintGuard:
    def test_reports_mismatched_digest(self):
        meta = {"chunking_fingerprint": {"standards_sha256": {
            "road_surface_inspection.md": "0" * 64}}}
        problems = re_.verify_fingerprint(meta)
        assert problems and "sha256" in problems[0]

    def test_reports_missing_file(self):
        meta = {"chunking_fingerprint": {"standards_sha256": {"nope.md": "0" * 64}}}
        problems = re_.verify_fingerprint(meta)
        assert problems and "missing" in problems[0]

    def test_committed_truth_matches_working_tree(self):
        """The recorded fingerprint must match the standards on disk right now.

        If this fails, the truth was annotated against different chunking and
        every recall@k number is meaningless until it is re-annotated.
        """
        assert re_.verify_fingerprint(re_.load_truth_meta()) == []


class TestCommittedTruthIntegrity:
    def test_golden_set_size_and_truth_shape(self):
        items = re_.load_golden_set()
        assert len(items) == 10
        for it in items:
            truth = it["retrieval_truth"]
            assert isinstance(truth["chunk_ids"], list)
            assert isinstance(truth["no_applicable_clause"], bool)
            # An empty set must be declared, never left implicit.
            assert truth["no_applicable_clause"] is (len(truth["chunk_ids"]) == 0)

    def test_truth_ids_are_wellformed_and_anchored(self):
        items = re_.load_golden_set()
        meta = re_.load_truth_meta()
        anchors = meta["anchors"]
        for it in items:
            for cid in re_.truth_of(it):
                filename, _, index = cid.partition("::")
                assert filename.endswith(".md"), cid
                assert index.isdigit(), cid
                assert cid in anchors, f"no anchor recorded for {cid}"

    def test_exactly_one_empty_truth_item_is_declared(self):
        items = re_.load_golden_set()
        empty = [it["id"] for it in items if not re_.truth_of(it)]
        assert empty == ["marina_aerial"]
