"""Tests for deterministic citation verification (no LLM involved).

These checks are the L1 layer: they must be able to run as a hard CI gate,
which is only true because they involve no model.
"""
import pytest

from app.citations import (
    DEFAULT_MATCH_THRESHOLD,
    CitationStatus,
    check_citation,
    check_findings,
    fuzzy_contains,
)

CHUNK = {
    "chunk_id": "road_surface_inspection.md::2",
    "standard": "Road Surface Inspection",
    "source": "road_surface_inspection.md",
    "text": ("Cracks wider than 3 mm shall be recorded as structural defects "
             "and scheduled for repair within 30 days."),
}


class TestFuzzyContains:
    def test_exact_substring_scores_one(self):
        assert fuzzy_contains(CHUNK["text"], "wider than 3 mm") == 1.0

    def test_case_and_whitespace_insensitive(self):
        assert fuzzy_contains(CHUNK["text"], "  CRACKS   WIDER than 3 MM ") == 1.0

    def test_paraphrase_scores_below_one(self):
        score = fuzzy_contains(CHUNK["text"], "fissures exceeding three millimetres")
        assert 0.0 < score < 1.0

    def test_unrelated_scores_near_zero(self):
        assert fuzzy_contains(CHUNK["text"], "marina berthing fees") < 0.5

    def test_empty_needle_is_zero(self):
        assert fuzzy_contains(CHUNK["text"], "") == 0.0

    def test_needle_longer_than_haystack_does_not_crash(self):
        assert 0.0 <= fuzzy_contains("short", "a " * 50) <= 1.0


class TestCheckCitation:
    def test_ok_when_chunk_retrieved_and_quote_present(self):
        status, reason = check_citation(
            {"chunk_id": CHUNK["chunk_id"],
             "quote": "Cracks wider than 3 mm shall be recorded"},
            [CHUNK])
        assert status is CitationStatus.OK
        assert "match" in reason

    def test_hallucinated_when_citation_missing(self):
        status, _ = check_citation(None, [CHUNK])
        assert status is CitationStatus.HALLUCINATED

    def test_hallucinated_when_chunk_not_retrieved(self):
        """The model invented a reference — the failure this exists to catch."""
        status, reason = check_citation(
            {"chunk_id": "pedestrian_safety.md::9", "quote": "anything"},
            [CHUNK])
        assert status is CitationStatus.HALLUCINATED
        assert "not retrieved" in reason

    def test_unresolved_when_quote_not_in_chunk(self):
        status, reason = check_citation(
            {"chunk_id": CHUNK["chunk_id"],
             "quote": "all personnel must wear high-visibility clothing"},
            [CHUNK])
        assert status is CitationStatus.UNRESOLVED
        assert "threshold" in reason

    def test_hallucinated_when_no_chunk_id(self):
        status, _ = check_citation({"quote": "something"}, [CHUNK])
        assert status is CitationStatus.HALLUCINATED

    def test_threshold_is_configurable(self):
        """A near-miss quote flips status when the gate is raised."""
        # '3mm' (no space) means no exact substring: score is high but < 1.
        citation = {"chunk_id": CHUNK["chunk_id"],
                    "quote": "Cracks wider than 3mm shall be recorded"}
        strict, _ = check_citation(citation, [CHUNK], threshold=0.99)
        loose, _ = check_citation(citation, [CHUNK], threshold=0.2)
        assert strict is CitationStatus.UNRESOLVED
        assert loose is CitationStatus.OK


class TestCheckFindings:
    def test_counts_and_pass_rate(self):
        findings = [
            {"id": "f1", "citation": {"chunk_id": CHUNK["chunk_id"],
                                      "quote": "Cracks wider than 3 mm"}},
            {"id": "f2", "citation": {"chunk_id": "not_retrieved.md::1",
                                      "quote": "x"}},
            {"id": "f3", "citation": None},
        ]
        rep = check_findings(findings, [CHUNK])
        assert rep["total_findings"] == 3
        assert rep["counts"]["ok"] == 1
        assert rep["counts"]["hallucinated"] == 2
        assert rep["attribution_pass_rate"] == pytest.approx(1 / 3)

    def test_empty_findings_gives_none_not_zero(self):
        """A pass rate of 0.0 would be a claim; None means 'nothing measured'."""
        assert check_findings([], [CHUNK])["attribution_pass_rate"] is None

    def test_no_retrieved_chunks_makes_everything_hallucinated(self):
        rep = check_findings(
            [{"id": "f1", "citation": {"chunk_id": "a::1", "quote": "q"}}], [])
        assert rep["counts"]["hallucinated"] == 1

    def test_default_threshold_is_documented_not_measured(self):
        # Guard against silently changing the gate value: if this drifts, every
        # published number has to be re-measured.
        assert DEFAULT_MATCH_THRESHOLD == 0.55


class TestFuzzyContainsBoundaries:
    """Edge cases that decide whether a citation passes, pinned by measurement.

    All values below were read off the implementation, not inferred from it.
    """

    def test_substring_match_ignores_word_boundaries(self):
        """The ``n in h`` fast path is character-level, so a quote can score
        1.00 without appearing as a word. A three-letter quote inside a longer
        word is enough to look perfectly grounded."""
        assert fuzzy_contains("concatenate the values", "cat") == 1.0
        assert fuzzy_contains("The Concatenation step", "cat") == 1.0

    def test_whitespace_only_needle_is_zero_not_one(self):
        """Normalisation strips a blank quote down to '', which is the same
        guard as an absent quote -- not a free 1.0."""
        assert fuzzy_contains("anything at all", "   \n\t ") == 0.0

    def test_needle_longer_than_haystack_scales_by_word_window(self):
        """When the haystack is shorter than the window, one window is the
        whole haystack and the score is a plain ratio -- it neither clamps to
        0 nor saturates at 1."""
        score = fuzzy_contains("alpha beta", "alpha beta gamma delta")
        assert 0.0 < score < 1.0

    def test_needle_longer_than_haystack_can_still_pass_the_gate(self):
        """Consequence of the above: a quote that is half absent from the chunk
        still clears the default threshold, so pass rate is not a purity
        guarantee."""
        status, reason = check_citation(
            {"chunk_id": "c1", "quote": "alpha beta gamma delta"},
            [{"chunk_id": "c1", "text": "alpha beta"}])
        assert status is CitationStatus.OK
        assert "match 0.6" in reason


class TestCheckCitationBoundaries:
    def test_score_equal_to_threshold_passes(self):
        """The gate is ``score < threshold``, strictly: equality is OK, so an
        exact quote survives threshold=1.0."""
        status, _ = check_citation(
            {"chunk_id": "c1", "quote": "hello world"},
            [{"chunk_id": "c1", "text": "hello world"}], threshold=1.0)
        assert status is CitationStatus.OK

    def test_empty_dict_citation_uses_the_missing_citation_message(self):
        """An empty dict is falsy, so it reports 'no citation' rather than
        'no chunk_id' -- the reason string cannot tell the two apart."""
        status, reason = check_citation({}, [{"chunk_id": "c1", "text": "x"}])
        assert status is CitationStatus.HALLUCINATED
        assert reason == "finding has no citation"

    def test_duplicate_chunk_id_resolves_to_the_last_occurrence(self):
        """Chunks are zipped into a dict, so a repeated chunk_id silently keeps
        the last one: the quote from the first copy no longer matches its own
        text (1.00 -> 0.57) though it still clears the default gate."""
        chunks = [{"chunk_id": "c1", "text": "alpha beta"},
                  {"chunk_id": "c1", "text": "gamma delta"}]
        first, _ = check_citation({"chunk_id": "c1", "quote": "alpha beta"}, chunks)
        second, _ = check_citation({"chunk_id": "c1", "quote": "gamma delta"}, chunks)
        assert second is CitationStatus.OK
        assert first is CitationStatus.OK
        stricter, reason = check_citation(
            {"chunk_id": "c1", "quote": "alpha beta"}, chunks, threshold=0.99)
        assert stricter is CitationStatus.UNRESOLVED
        assert "0.57" in reason

    def test_retrieved_chunk_without_text_is_unresolved_not_hallucinated(self):
        """The chunk *was* retrieved, so it is a bad quote, not an invented
        reference -- the distinction the status model is built on."""
        status, reason = check_citation(
            {"chunk_id": "c9", "quote": "anything"}, [{"chunk_id": "c9"}])
        assert status is CitationStatus.UNRESOLVED
        assert "0.00" in reason

    def test_none_chunks_is_tolerated(self):
        """``retrieved_chunks=None`` means 'nothing retrieved', not a crash."""
        status, _ = check_citation({"chunk_id": "c1", "quote": "q"}, None)
        assert status is CitationStatus.HALLUCINATED

    def test_check_findings_accepts_none_chunks(self):
        rep = check_findings([], None)
        assert rep["total_findings"] == 0
        assert rep["attribution_pass_rate"] is None
