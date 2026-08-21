# tests/test_agent.py
"""Unit tests for the LangGraph agent: risk classification, routing logic,
and a full end-to-end graph run with all external services mocked."""

import pytest

import app.agent.graph as graph
from app.agent.graph import (
    InspectionAgent,
    _classify_risk,
    build_graph,
    lower_threshold_node,
    review_detections,
)


# ---------------------------- risk classification ---------------------------- #
class TestClassifyRisk:
    def test_explicit_high_risk(self):
        report = "## Analysis\n### 4. Risk Assessment\nOverall: high risk due to exposed wiring."
        assert _classify_risk(report) == "high"

    def test_explicit_moderate_maps_to_medium(self):
        report = "Risk Assessment\nThe site shows moderate risk of vehicle intrusion."
        assert _classify_risk(report) == "medium"

    def test_critical_maps_to_high(self):
        report = "Risk Assessment\ncritical risk of structural failure."
        assert _classify_risk(report) == "high"

    def test_explicit_statement_beats_keywords(self):
        # 'hazard' keyword appears, but the explicit 'low risk' statement wins
        report = ("Scene is a quiet car park. Risk Assessment\n"
                  "Overall low risk; no hazardous conditions present.")
        assert _classify_risk(report) == "low"

    def test_keyword_fallback_collapse(self):
        report = "The embankment shows early signs of collapse near the foundation."
        assert _classify_risk(report) == "high"

    def test_keyword_fallback_corrosion(self):
        report = "Moderate corrosion observed on the support beams."
        assert _classify_risk(report) == "medium"

    def test_no_signal_defaults_to_low(self):
        report = "A well-maintained pedestrian crossing with clear signage."
        assert _classify_risk(report) == "low"


# ---------------------------- routing / nodes ---------------------------- #
class TestRouting:
    def test_empty_first_pass_retries(self):
        state = {"det_result": {"counts": {}}, "retry_count": 0}
        assert review_detections(state) == "retry"

    def test_empty_second_pass_continues(self):
        state = {"det_result": {"counts": {}}, "retry_count": 1}
        assert review_detections(state) == "continue"

    def test_nonempty_continues(self):
        state = {"det_result": {"counts": {"car": 2}}, "retry_count": 0}
        assert review_detections(state) == "continue"

    def test_lower_threshold_halves_confidence(self):
        out = lower_threshold_node({"conf_thres": 0.25, "retry_count": 0,
                                    "decisions": []})
        assert out["conf_thres"] == 0.15
        assert out["retry_count"] == 1
        assert len(out["decisions"]) == 1

    def test_lower_threshold_floor_at_010(self):
        out = lower_threshold_node({"conf_thres": 0.12, "retry_count": 0,
                                    "decisions": []})
        assert out["conf_thres"] == 0.10  # 0.12*0.6=0.072 -> clamped to 0.10


# ---------------------------- full graph run (all mocks) ---------------------------- #
class _FakeDetector:
    def __init__(self, results):
        self.results = results  # list of det_results, one per detect() call
        self.calls = 0
        self.conf_thres = None

    def detect(self, image_path):
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        return result

    def draw(self, *args, **kwargs):
        pass


class _FakeVLM:
    def analyze(self, image_path, det_result):
        return ("### 1. Scene Description\nA construction site.\n"
                "### 4. Risk Assessment\nOverall high risk: unsecured "
                "scaffolding and workers without helmets.")


class _FakeRetriever:
    def __init__(self):
        self.last_k = None

    def retrieve(self, query, k=3):
        self.last_k = k
        return [{"text": f"standard excerpt #{i}", "standard": f"Std {i}",
                 "source": f"std_{i}.md", "distance": 0.1 * (i + 1)}
                for i in range(k)]


@pytest.fixture
def mocks(monkeypatch, tmp_path):
    det = _FakeDetector([
        {"image": "img.jpg", "inference_ms": 5.0, "counts": {},
         "detections": []},  # first pass: empty -> triggers retry
        {"image": "img.jpg", "inference_ms": 7.0,
         "counts": {"person": 2},
         "detections": [
             {"class": "person", "confidence": 0.6, "bbox": [1, 2, 3, 4]},
             {"class": "person", "confidence": 0.5, "bbox": [5, 6, 7, 8]}]},
    ])
    vlm = _FakeVLM()
    retriever = _FakeRetriever()
    monkeypatch.setattr(graph, "_get_detector", lambda path: det)
    monkeypatch.setattr(graph, "_get_vlm", lambda: vlm)
    monkeypatch.setattr(graph, "_get_retriever", lambda: retriever)
    return det, retriever


def test_full_flow_adaptive_retry_and_risk_depth(mocks, tmp_path):
    det, retriever = mocks
    agent = InspectionAgent()
    state = agent.run("some_image.jpg", conf_thres=0.25,
                      save=False, save_dir=str(tmp_path))

    # adaptive re-detection happened exactly once
    assert det.calls == 2
    assert state["retry_count"] == 1
    assert any("lowering threshold" in d for d in state["decisions"])

    # high-risk report -> retrieval depth 5 (not the default 3)
    assert state["risk_level"] == "high"
    assert retriever.last_k == 5
    assert len(state["standards"]) == 5

    # detection result comes from the (successful) second pass
    assert state["det_result"]["counts"] == {"person": 2}
    assert state["vlm_report"].startswith("### 1. Scene Description")


def test_full_flow_no_retry_when_objects_found(mocks, monkeypatch, tmp_path):
    det, retriever = mocks
    det.results = [{"image": "img.jpg", "inference_ms": 5.0,
                    "counts": {"car": 1},
                    "detections": [{"class": "car", "confidence": 0.9,
                                    "bbox": [0, 0, 10, 10]}]}]
    agent = InspectionAgent()
    state = agent.run("some_image.jpg", conf_thres=0.25,
                      save=False, save_dir=str(tmp_path))
    assert det.calls == 1
    assert state["retry_count"] == 0
    assert not any("lowering threshold" in d for d in state["decisions"])


def test_build_graph_compiles():
    compiled = build_graph()
    assert compiled is not None
