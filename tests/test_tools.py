"""Tests for the tool registry: clamping, validation, budget, failure handling.

The negative tests matter more than the positive ones — the risk is not "the
tool doesn't work", it's "the model misuses it and we let it".
"""


class _FakeRetriever:
    def __init__(self, n_docs=5):
        self.n_docs = n_docs
        self.queries = []

    def retrieve(self, query, k=3):
        self.queries.append((query, k))
        return [
            {"chunk_id": f"{query}::{i}", "standard": f"Std {i}",
             "source": f"std_{i}.md", "text": f"clause {i} about {query}",
             "distance": 0.1 * i}
            for i in range(min(k, self.n_docs))
        ]


class _BoomRetriever:
    def retrieve(self, query, k=3):
        raise RuntimeError("chroma exploded")


def _registry(retriever=None, **kw):
    from app.tools import ToolRegistry
    return ToolRegistry(retriever or _FakeRetriever(), **kw)


class TestClamping:
    def test_k_within_range_passes_through(self):
        reg = _registry()
        out = reg.execute("retrieve_standards", {"query": "cracks", "k": 3})
        assert out["k"] == 3
        assert len(out["clauses"]) == 3

    def test_k_above_max_is_clamped_not_rejected(self):
        """The boundary is a code-side guardrail; the model only gets a say."""
        reg = _registry()
        out = reg.execute("retrieve_standards", {"query": "cracks", "k": 99})
        assert out["k"] == reg.max_k

    def test_k_zero_or_negative_is_clamped_to_one(self):
        reg = _registry()
        assert reg.execute("retrieve_standards",
                           {"query": "cracks", "k": 0})["k"] == 1
        assert reg.execute("retrieve_standards",
                           {"query": "cracks", "k": -5})["k"] == 1

    def test_k_defaults_to_three(self):
        reg = _registry()
        assert reg.execute("retrieve_standards", {"query": "cracks"})["k"] == 3

    def test_boolean_is_not_accepted_as_integer(self):
        reg = _registry()
        out = reg.execute("retrieve_standards", {"query": "cracks", "k": True})
        assert out["error"]
        assert any("integer" in i for i in out["issues"])


class TestValidation:
    def test_empty_query_is_rejected_with_a_hint(self):
        reg = _registry()
        out = reg.execute("retrieve_standards", {"query": "   ", "k": 2})
        assert out["error"] == "invalid arguments"
        assert out["hint"]

    def test_missing_query_is_rejected(self):
        reg = _registry()
        assert reg.execute("retrieve_standards", {"k": 2})["error"]

    def test_non_object_arguments_are_rejected(self):
        reg = _registry()
        out = reg.execute("retrieve_standards", "cracks")
        assert out["error"]
        assert "object" in out["issues"][0]

    def test_unknown_tool_returns_structured_error_not_exception(self):
        reg = _registry()
        out = reg.execute("drop_database", {})
        assert "unknown tool" in out["error"]
        assert "retrieve_standards" in out["available"]

    def test_failed_tool_does_not_raise(self):
        """A tool failure must degrade the run, not kill the request."""
        reg = _registry(_BoomRetriever())
        out = reg.execute("retrieve_standards", {"query": "cracks"})
        assert "retrieval failed" in out["error"]
        assert reg.calls_used == 0


class TestBudget:
    def test_budget_counts_only_retrieval_calls(self):
        reg = _registry(max_calls=2)
        reg.execute("retrieve_standards", {"query": "a"})
        reg.execute("submit_findings", {"findings": []})
        assert reg.calls_used == 1
        assert not reg.budget_exhausted

    def test_budget_exhausted_returns_error_with_hint(self):
        reg = _registry(max_calls=1)
        reg.execute("retrieve_standards", {"query": "a"})
        out = reg.execute("retrieve_standards", {"query": "b"})
        assert "budget exhausted" in out["error"]
        assert "submit" in out["hint"]

    def test_submit_still_works_after_budget_exhausted(self):
        """Otherwise a confused model can never finish the run."""
        reg = _registry(max_calls=0)
        out = reg.execute("submit_findings", {"findings": []})
        assert out["submitted"] is True


class TestAccumulation:
    def test_retrieved_is_the_union_without_duplicates(self):
        reg = _registry()
        reg.execute("retrieve_standards", {"query": "cracks", "k": 3})
        reg.execute("retrieve_standards", {"query": "cracks", "k": 3})
        ids = [c["chunk_id"] for c in reg.retrieved]
        assert len(ids) == len(set(ids)) == 3

    def test_call_log_records_args_for_replay(self):
        reg = _registry()
        reg.execute("retrieve_standards", {"query": "cracks", "k": 2})
        assert reg.call_log()[0]["args"] == {"query": "cracks", "k": 2}
        assert reg.call_log()[0]["n_results"] == 2


class TestArgumentParsing:
    def test_valid_json(self):
        from app.tools import parse_tool_arguments
        args, err = parse_tool_arguments('{"query": "x", "k": 2}')
        assert args == {"query": "x", "k": 2} and err is None

    def test_malformed_json_returns_error(self):
        from app.tools import parse_tool_arguments
        args, err = parse_tool_arguments('{"query": "x"')
        assert args is None and "not valid JSON" in err

    def test_empty_arguments(self):
        from app.tools import parse_tool_arguments
        args, err = parse_tool_arguments(None)
        assert args is None and err == "empty arguments"

    def test_json_array_is_rejected(self):
        from app.tools import parse_tool_arguments
        args, err = parse_tool_arguments('[1,2,3]')
        assert args is None and "object" in err
