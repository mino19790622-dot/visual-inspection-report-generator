"""Tests for the grounding agent's loop control and output verification.

The model is scripted (never called for real). What is under test is *our*
control flow: does the loop terminate, does it degrade instead of crashing, and
does a citation that the model invented get caught?
"""
import pytest

from app.citations import CitationStatus
from app.grounding import GroundingAgent, summarise_detections
from tests.fakes import ScriptedClient, response, tool_call

RETRIEVER_CHUNKS = [
    {"chunk_id": "road_surface_inspection.md::2",
     "standard": "Road Surface Inspection", "source": "road_surface_inspection.md",
     "text": "Cracks wider than 3 mm shall be recorded as structural defects.",
     "distance": 0.2},
]


class _Retriever:
    def __init__(self, chunks=None):
        self.chunks = chunks if chunks is not None else RETRIEVER_CHUNKS
        self.calls = []

    def retrieve(self, query, k=3):
        self.calls.append((query, k))
        return self.chunks[:k]


def _agent(client, retriever=None):
    return GroundingAgent(retriever or _Retriever(), client=client)


def _submit(findings=None, unresolved=None):
    return tool_call("c2", "submit_findings", {
        "findings": findings if findings is not None else [
            {"id": "f1", "severity": "high",
             "description": "Structural cracking exceeds the recorded limit.",
             "citation": {"chunk_id": "road_surface_inspection.md::2",
                          "quote": "Cracks wider than 3 mm shall be recorded"}},
        ],
        "unresolved": unresolved or [],
    })


class TestHappyPath:
    def test_retrieve_then_submit_produces_verified_finding(self):
        client = ScriptedClient([
            response(tool_calls=[tool_call("c1", "retrieve_standards",
                                           {"query": "crack width limit", "k": 2})]),
            response(tool_calls=[_submit()]),
        ])
        report = _agent(client).run("A cracked road surface.")

        assert report.parse_failed is False
        assert len(report.findings) == 1
        assert report.findings[0].citation_status == CitationStatus.OK.value
        assert report.citation_report["attribution_pass_rate"] == 1.0
        assert report.max_k_requested == 2
        # the retrieval actually happened, with the model's query
        assert report.tool_calls[0]["args"]["query"] == "crack width limit"

    def test_usage_is_accumulated_across_rounds(self):
        client = ScriptedClient([
            response(tool_calls=[tool_call("c1", "retrieve_standards",
                                           {"query": "q"})],
                     prompt_tokens=100, completion_tokens=10),
            response(tool_calls=[_submit()],
                     prompt_tokens=50, completion_tokens=20),
        ])
        report = _agent(client).run("narrative")
        assert report.usage["prompt_tokens"] == 150
        assert report.usage["completion_tokens"] == 30
        assert report.usage["latency_ms"] >= 0

    def test_tools_are_offered_in_agentic_mode(self):
        client = ScriptedClient([response(tool_calls=[_submit()])])
        _agent(client).run("narrative")
        names = {t["function"]["name"] for t in client.requests[0]["tools"]}
        assert names == {"retrieve_standards", "submit_findings"}


class TestHallucinationCaught:
    def test_citation_to_unretrieved_chunk_is_flagged(self):
        """The single most important case: the model invents a reference."""
        client = ScriptedClient([
            response(tool_calls=[_submit([
                {"id": "f1", "severity": "high", "description": "invented",
                 "citation": {"chunk_id": "never_retrieved.md::9",
                              "quote": "anything at all"}},
            ])]),
        ])
        report = _agent(client).run("narrative")
        assert report.findings[0].citation_status == "hallucinated"
        assert report.citation_report["counts"]["hallucinated"] == 1
        assert report.citation_report["attribution_pass_rate"] == 0.0

    def test_finding_without_citation_is_flagged(self):
        client = ScriptedClient([
            response(tool_calls=[_submit([
                {"id": "f1", "severity": "low", "description": "no basis",
                 "citation": None},
            ])]),
        ])
        report = _agent(client).run("narrative")
        assert report.findings[0].citation_status == "hallucinated"

    def test_quote_not_present_in_chunk_is_unresolved(self):
        client = ScriptedClient([
            response(tool_calls=[tool_call("c1", "retrieve_standards",
                                           {"query": "cracks"})]),
            response(tool_calls=[_submit([
                {"id": "f1", "severity": "low", "description": "misquoted",
                 "citation": {"chunk_id": "road_surface_inspection.md::2",
                              "quote": "all staff must wear hi-vis clothing"}},
            ])]),
        ])
        report = _agent(client).run("narrative")
        assert report.findings[0].citation_status == "unresolved"

    def test_unresolved_observations_are_kept_not_dropped(self):
        client = ScriptedClient([
            response(tool_calls=[_submit(
                findings=[], unresolved=["Standing water of unknown depth"])]),
        ])
        report = _agent(client).run("narrative")
        assert report.unresolved == ["Standing water of unknown depth"]
        assert report.findings == []


class TestDegradation:
    def test_prose_instead_of_tool_call_is_a_failure_not_a_guess(self):
        """Parsing prose into findings is how unverifiable citations are born."""
        client = ScriptedClient([response(content="Here are my findings: ...")])
        report = _agent(client).run("narrative")
        assert report.parse_failed is True
        assert report.failure_reason == "model returned no tool call"
        assert report.findings == []

    def test_model_that_never_submits_fails_cleanly(self):
        client = ScriptedClient([
            response(tool_calls=[tool_call(f"c{i}", "retrieve_standards",
                                           {"query": f"q{i}"})])
            for i in range(10)
        ])
        report = _agent(client).run("narrative")
        assert report.parse_failed is True
        assert "never called submit_findings" in report.failure_reason
        # budget still bounded the damage
        assert report.max_k_requested <= 5

    def test_invalid_severity_fails_schema_validation(self):
        client = ScriptedClient([
            response(tool_calls=[_submit([
                {"id": "f1", "severity": "catastrophic", "description": "x",
                 "citation": {"chunk_id": "a::1", "quote": "q"}},
            ])]),
        ])
        report = _agent(client).run("narrative")
        assert report.parse_failed is True
        assert "schema validation" in report.failure_reason

    def test_malformed_json_arguments_are_fed_back_and_loop_continues(self):
        client = ScriptedClient([
            response(tool_calls=[tool_call("c1", "retrieve_standards",
                                           "{not json")]),
            response(tool_calls=[_submit()]),
        ])
        report = _agent(client).run("narrative")
        assert report.parse_failed is False
        # the error was returned to the model as a tool message
        tool_msgs = [m for m in client.requests[1]["messages"]
                     if m.get("role") == "tool"]
        assert any("not valid JSON" in m["content"] for m in tool_msgs)

    def test_unknown_tool_returns_error_and_loop_continues(self):
        client = ScriptedClient([
            response(tool_calls=[tool_call("c1", "delete_everything", {})]),
            response(tool_calls=[_submit()]),
        ])
        report = _agent(client).run("narrative")
        assert report.parse_failed is False
        tool_msgs = [m for m in client.requests[1]["messages"]
                     if m.get("role") == "tool"]
        assert any("unknown tool" in m["content"] for m in tool_msgs)

    def test_llm_failure_is_captured_not_raised(self):
        class Boom:
            chat = property(lambda self: self)
            completions = property(lambda self: self)

            def create(self, **kwargs):
                raise RuntimeError("upstream 500")

        report = _agent(Boom()).run("narrative")
        assert report.parse_failed is True
        assert "llm call failed" in report.failure_reason

    def test_retriever_failure_is_captured_not_raised(self):
        class BoomRetriever:
            def retrieve(self, query, k=3):
                raise RuntimeError("chroma down")

        client = ScriptedClient([
            response(tool_calls=[tool_call("c1", "retrieve_standards",
                                           {"query": "q"})]),
            response(tool_calls=[_submit()]),
        ])
        report = GroundingAgent(BoomRetriever(), client=client).run("narrative")
        assert report.parse_failed is False  # degraded, but produced findings


class TestFixedMode:
    def test_no_retrieval_tool_is_offered(self):
        client = ScriptedClient([response(tool_calls=[_submit()])])
        _agent(client).run("narrative", mode="fixed")
        names = [t["function"]["name"] for t in client.requests[0]["tools"]]
        assert names == ["submit_findings"]

    def test_chunks_are_pre_retrieved_at_max_depth(self):
        retriever = _Retriever()
        client = ScriptedClient([response(tool_calls=[_submit()])])
        report = GroundingAgent(retriever, client=client).run(
            "narrative", mode="fixed")
        assert retriever.calls[0][1] == 5  # MAX_K
        assert report.citation_report["counts"]["ok"] == 1

    def test_fixed_mode_records_the_pre_retrieval_in_the_call_log(self):
        client = ScriptedClient([response(tool_calls=[_submit()])])
        report = _agent(client).run("narrative", mode="fixed")
        assert report.tool_calls[0]["args"]["k"] == 5


class TestDetectionSummary:
    def test_empty(self):
        assert summarise_detections({}) == "no objects detected"

    def test_counts_and_confidence_range(self):
        det = {"counts": {"person": 2, "car": 1},
               "detections": [{"confidence": 0.9}, {"confidence": 0.45}]}
        out = summarise_detections(det)
        assert "car×1" in out and "person×2" in out
        assert "0.45" in out and "0.90" in out

    def test_counts_without_detections(self):
        assert "person×2" in summarise_detections({"counts": {"person": 2}})


@pytest.mark.parametrize("mode", ["agentic", "fixed"])
def test_both_modes_produce_a_report(mode):
    client = ScriptedClient([
        response(tool_calls=[tool_call("c1", "retrieve_standards",
                                       {"query": "q"})]),
        response(tool_calls=[_submit()]),
    ])
    report = _agent(client).run("narrative", mode=mode)
    assert report.mode == mode
    assert len(report.findings) == 1
