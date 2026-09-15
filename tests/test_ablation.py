# tests/test_ablation.py
"""Deterministic tests for eval/ablation_topk.py.

These caught two real bugs during phase C:
  * `_deterministic_checks` must receive `item["expect"]`, not the whole item
    (the golden set nests the expectations under an `expect` key).
  * `GroundedReport` has no `calls_used` attribute (that lives on ToolRegistry),
    so retrieval calls must be derived from `tool_calls`.

No network / LLM: the agent is faked and the grounded state is constructed
directly from the real schema.
"""
from __future__ import annotations

from app.grounding import Citation, Finding, GroundedReport
from eval import ablation_topk as ab


class FakeAgent:
    def __init__(self, state: dict):
        self._state = state

    def run(self, path, save=True, grounding_mode="agentic"):
        return dict(self._state)


def _item() -> dict:
    return {
        "id": "x",
        "image": "data/test_images/bus.jpg",
        "expect": {
            "min_detections": 0, "max_detections": 9,
            "must_mention_any": ["bus"], "should_flag_safety_concerns": False,
        },
    }


def _grounded() -> GroundedReport:
    return GroundedReport(
        findings=[Finding(id="f1", severity="high", description="guard rail",
                          citation=Citation(chunk_id="std::0", quote="rail"))],
        unresolved=["something"],
        tool_calls=[
            {"tool": "retrieve_standards", "args": {"query": "rail", "k": 2},
             "n_results": 2},
        ],
        citation_report={
            "total_findings": 1,
            "counts": {"ok": 1, "unresolved": 0, "hallucinated": 0},
            "attribution_pass_rate": 1.0, "detail": [],
        },
        usage={"prompt_tokens": 30, "completion_tokens": 10,
               "total_tokens": 40, "latency_ms": 5},
        mode="agentic",
    )


def test_collect_arm_off_mode_has_no_grounding():
    state = {
        "det_result": {"counts": {"bus": 1}},
        "vlm_report": "a bus at a stop",
        "grounded": None,
        "vlm_usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    rec = ab._collect_arm(FakeAgent(state), _item(), "off", judge=False)
    assert rec["findings"] == 0
    assert rec["retrieval_calls"] == 0
    assert rec["citation_counts"] is None
    assert rec["deterministic_pass"] is True   # 1 detection within [0,9], "bus" present
    assert rec["vlm_tokens"] == 15


def test_collect_arm_grounded_mode_reads_tool_calls():
    state = {
        "det_result": {"counts": {"bus": 1}},
        "vlm_report": "a bus at a stop",
        "grounded": _grounded(),
        "vlm_usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    rec = ab._collect_arm(FakeAgent(state), _item(), "agentic", judge=False)
    assert rec["findings"] == 1
    assert rec["retrieval_calls"] == 1          # derived from tool_calls, not calls_used
    assert rec["k_values"] == [2]
    assert rec["k_max"] == 2
    assert rec["citation_counts"]["ok"] == 1
    assert rec["grounding_tokens"] == 40


def test_aggregate_excludes_error_rows():
    recs = [
        {"id": "a", "arm": "agentic", "findings": 2,
         "citation_counts": {"ok": 2, "unresolved": 0, "hallucinated": 0},
         "attribution_pass_rate": 1.0, "retrieval_calls": 3,
         "k_values": [2, 2], "k_max": 2, "deterministic_pass": True,
         "vlm_tokens": 100, "grounding_tokens": 50, "parse_failed": False},
        {"id": "b", "arm": "agentic", "error": "boom"},
    ]
    agg = ab._aggregate(recs)
    assert agg["images"] == 1
    assert agg["errors"] == [{"id": "b", "error": "boom"}]
    assert agg["attribution_pass_rate"] == 1.0
    assert agg["retrieval_calls_mean"] == 3.0
    assert agg["k_values"] == [2, 2]
    assert agg["deterministic_pass_rate"] == 1.0


def test_aggregate_empty_arm_is_safe():
    agg = ab._aggregate([])
    assert agg["images"] == 0
    assert agg["attribution_pass_rate"] is None
    assert agg["retrieval_calls_mean"] is None
