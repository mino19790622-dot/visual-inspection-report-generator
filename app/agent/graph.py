# app/agent/graph.py
"""LangGraph agent orchestrating the inspection pipeline with decision logic.

Graph:
    START -> detect -> review_detections (conditional)
                |-- "retry"    -> lower_threshold -> detect   (max 1 retry)
                |-- "continue" -> analyze -> assess_risk -> retrieve -> export -> END

Decision logic:
  1. Adaptive re-detection: if YOLO finds 0 objects, lower confidence
     threshold and re-run once (many inspection images are aerial / unusual).
  2. Risk-based retrieval depth: high-risk VLM reports retrieve more
     standards (k=5) than low-risk ones (k=3).
"""

import re
from functools import lru_cache
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from app.detection.detector import YOLODetector
from app.vlm.client import VLMClient
from app.rag.retriever import StandardsRetriever
from app.reporting.exporter import ReportExporter


class AgentState(TypedDict, total=False):
    image_path: str
    conf_thres: float
    retry_count: int
    det_result: dict
    vlm_report: str
    risk_level: str
    top_k: int
    standards: list
    decisions: list
    saved: dict
    save: bool
    save_dir: str


# ---------- expensive resources: build once, reuse across runs ----------
@lru_cache(maxsize=1)
def _get_detector(onnx_path: str):
    return YOLODetector(onnx_path)


@lru_cache(maxsize=1)
def _get_vlm():
    return VLMClient()


@lru_cache(maxsize=1)
def _get_retriever():
    return StandardsRetriever()


# ---------- risk keywords for retrieval-depth decision ----------
HIGH_RISK_PATTERNS = re.compile(
    r"high risk|critical|immediate|urgent|hazard|unsafe|violation|"
    r"structural (?:failure|damage)|collapse|emergency|severe",
    re.IGNORECASE,
)
MEDIUM_RISK_PATTERNS = re.compile(
    r"moderate risk|caution|monitor|degrad|obstruct|corrosion|crack",
    re.IGNORECASE,
)


# ---------------------------- nodes ---------------------------- #
def detect_node(state: AgentState) -> dict:
    conf = state.get("conf_thres", 0.25)
    detector = _get_detector("yolov8m.onnx")
    detector.conf_thres = conf  # cached instance: update threshold per run
    det_result = detector.detect(state["image_path"])
    n = sum(det_result["counts"].values())
    if state.get("retry_count", 0) > 0:
        print(f"  [agent] re-detection @ conf={conf}: {n} objects")
    else:
        print(f"  [agent] detection @ conf={conf}: {n} objects")
    return {"det_result": det_result}


def lower_threshold_node(state: AgentState) -> dict:
    old = state.get("conf_thres", 0.25)
    new = max(0.10, round(old * 0.6, 3))
    decision = (f"Zero objects detected @ conf={old}; "
                f"lowering threshold to {new} and re-running detection")
    print(f"  [agent] {decision}")
    return {
        "conf_thres": new,
        "retry_count": state.get("retry_count", 0) + 1,
        "decisions": state.get("decisions", []) + [decision],
    }


def analyze_node(state: AgentState) -> dict:
    print("  [agent] VLM analysis")
    vlm = _get_vlm()
    report = vlm.analyze(state["image_path"], state["det_result"])
    return {"vlm_report": report}


def assess_risk_node(state: AgentState) -> dict:
    risk = _classify_risk(state["vlm_report"])
    k = {"high": 5, "medium": 4, "low": 3}[risk]
    decision = f"Risk level: {risk} -> retrieving top {k} standards"
    print(f"  [agent] {decision}")
    return {"risk_level": risk, "top_k": k,
            "decisions": state.get("decisions", []) + [decision]}


def _classify_risk(report: str) -> str:
    """Classify risk. Explicit statement in the Risk Assessment section wins;
    keyword scan is only a fallback — avoids false positives from negated
    phrases like 'no hazardous conditions'."""
    if "Risk Assessment" in report:
        section = report.split("Risk Assessment", 1)[1]
    else:
        section = report
    m = re.search(r"\b(critical|high|moderate|medium|low)[\s-]*risk\b",
                  section[:500], re.IGNORECASE)
    if m:
        lvl = m.group(1).lower()
        return {"critical": "high", "high": "high",
                "moderate": "medium", "medium": "medium", "low": "low"}[lvl]
    if HIGH_RISK_PATTERNS.search(report):
        return "high"
    if MEDIUM_RISK_PATTERNS.search(report):
        return "medium"
    return "low"


def retrieve_node(state: AgentState) -> dict:
    retriever = _get_retriever()
    standards = retriever.retrieve(state["vlm_report"], k=state.get("top_k", 3))
    print(f"  [agent] retrieved {len(standards)} standards")
    return {"standards": standards}


def export_node(state: AgentState) -> dict:
    if not state.get("save", True):
        return {"saved": {}}
    detector = _get_detector("yolov8m.onnx")
    exporter = ReportExporter(state.get("save_dir", "reports"))
    saved = exporter.export(
        state["image_path"], state["det_result"], state["vlm_report"],
        state["standards"], draw_fn=detector.draw,
        decisions=state.get("decisions", []),
    )
    for kind, path in saved.items():
        print(f"  [agent] saved [{kind}] {path}")
    return {"saved": saved}


# ---------------------------- routing ---------------------------- #
def review_detections(state: AgentState) -> str:
    """After detection: retry once on empty results, else continue."""
    n = sum(state["det_result"]["counts"].values())
    if n == 0 and state.get("retry_count", 0) == 0:
        return "retry"
    return "continue"


# ---------------------------- graph assembly ---------------------------- #
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("detect", detect_node)
    g.add_node("lower_threshold", lower_threshold_node)
    g.add_node("analyze", analyze_node)
    g.add_node("assess_risk", assess_risk_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("export", export_node)

    g.add_edge(START, "detect")
    g.add_conditional_edges(
        "detect", review_detections,
        {"retry": "lower_threshold", "continue": "analyze"},
    )
    g.add_edge("lower_threshold", "detect")
    g.add_edge("analyze", "assess_risk")
    g.add_edge("assess_risk", "retrieve")
    g.add_edge("retrieve", "export")
    g.add_edge("export", END)
    return g.compile()


class InspectionAgent:
    """High-level entry point: run the LangGraph inspection agent on an image."""

    def __init__(self):
        self.graph = build_graph()

    def run(self, image_path: str, conf_thres: float = 0.25,
            save: bool = True, save_dir: str = "reports") -> dict:
        initial: AgentState = {
            "image_path": image_path,
            "conf_thres": conf_thres,
            "retry_count": 0,
            "decisions": [],
            "save": save,
            "save_dir": save_dir,
        }
        final = self.graph.invoke(initial, config={"recursion_limit": 15})
        return final
