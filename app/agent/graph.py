# app/agent/graph.py
"""LangGraph agent orchestrating the inspection pipeline with decision logic.

Graph (v2):
    START -> detect -> review_detections (conditional)
                |-- "retry"    -> lower_threshold -> detect   (max 1 retry)
                |-- "continue" -> analyze -> assess_risk -> ground -> export -> END

Decision logic
--------------
1. **Adaptive re-detection** (unchanged from v1.0): if YOLO finds 0 objects,
   lower the confidence threshold and re-run once.
2. **Risk classification** (unchanged): parsed out of the VLM narrative, used
   for routing/audit — no longer decides retrieval depth.
3. **Grounded findings with tools** (new in v2): the ``ground`` node runs a
   text agent that *decides* which standard clauses it needs via
   ``retrieve_standards`` and emits findings whose citations point at chunks it
   actually retrieved.

Why retrieval depth moved from a rule to the model
--------------------------------------------------
v1.0 chose ``k`` from the VLM risk level with a hard-coded dict. v2 puts the
retrieved text *in front of* generation (that is the whole point of grounded
generation), which makes the old rule circular: the risk level comes out of the
VLM report, but the VLM would now need the retrieved context to write it.
Letting the model ask for what it needs resolves the cycle — and is the
textbook case for tool calling rather than a feature added to match a job ad.

The code keeps the guardrails: ``k`` is clamped to ``[1, MAX_K]`` and the
retrieval-call budget is capped, both in ``app/tools.py``.
"""

import re
import time
from functools import lru_cache
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.detection.detector import YOLODetector
from app.grounding import GroundedReport, GroundingAgent, summarise_detections
from app.observability import RequestTrace, log_trace
from app.rag.retriever import StandardsRetriever
from app.reporting.exporter import ReportExporter
from app.vlm.client import VLMClient

DEFAULT_GROUNDING_MODE = "agentic"  # or "fixed" (control arm for the ablation)


class AgentState(TypedDict, total=False):
    image_path: str
    conf_thres: float
    retry_count: int
    det_result: dict
    vlm_report: str
    vlm_usage: dict  # token counts + latency for the VLM call
    risk_level: str
    top_k: int
    standards: list
    grounded: GroundedReport
    decisions: list
    saved: dict
    save: bool
    save_dir: str
    total_latency_ms: int
    grounding_mode: str
    trace: RequestTrace


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


# ---------- risk keywords for the narrative fallback ----------
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
    trace = state.get("trace")
    if trace is None:
        return {"det_result": detector.detect(state["image_path"])}
    with trace.span("detect", conf_thres=conf) as span:
        det_result = detector.detect(state["image_path"])
        span.meta["object_count"] = sum(det_result["counts"].values())
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
    """Perception stage — unchanged from v1.0, including its prompt."""
    print("  [agent] VLM analysis")
    vlm = _get_vlm()
    trace = state.get("trace")
    if trace is None:
        result = vlm.analyze(state["image_path"], state["det_result"])
        return {"vlm_report": result["report"], "vlm_usage": result["usage"]}
    with trace.span("vlm_call", model=getattr(vlm, "MODEL", None)) as span:
        result = vlm.analyze(state["image_path"], state["det_result"])
        trace.set_usage(span, result["usage"])
        span.meta["latency_ms"] = (result["usage"] or {}).get("latency_ms")
    return {"vlm_report": result["report"], "vlm_usage": result["usage"]}


def assess_risk_node(state: AgentState) -> dict:
    """Classify risk for routing/audit.

    v1.0 also derived ``top_k`` here; that now belongs to the grounding agent,
    which chooses retrieval depth per query.
    """
    risk = _classify_risk(state["vlm_report"])
    decision = f"Risk level: {risk}"
    print(f"  [agent] {decision}")
    return {"risk_level": risk,
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


def ground_node(state: AgentState) -> dict:
    """Grounding stage: tool-calling agent -> structured, cited findings."""
    mode = state.get("grounding_mode") or DEFAULT_GROUNDING_MODE

    if mode == "off":
        # Control for the v1.0 regression check: with grounding disabled the
        # narrative path is byte-for-byte the v1.0 pipeline, so the existing
        # 10-image golden set tells us whether Phase 0 broke anything.
        print("  [agent] grounding disabled (mode=off)")
        return {"grounded": None, "standards": [], "top_k": 0,
                "decisions": state.get("decisions", [])
                + ["Grounding disabled (mode=off)"]}

    print(f"  [agent] grounding findings (mode={mode})")
    retriever = _get_retriever()
    agent = GroundingAgent(retriever)
    trace = state.get("trace")

    if trace is None:
        report = agent.run(state["vlm_report"],
                           summarise_detections(state.get("det_result")),
                           mode=mode)
    else:
        with trace.span("ground", model=agent.model, mode=mode) as span:
            report = agent.run(state["vlm_report"],
                               summarise_detections(state.get("det_result")),
                               mode=mode)
            trace.set_usage(span, report.usage)
            span.meta["tool_calls"] = len(report.tool_calls)
            span.meta["findings"] = len(report.findings)
            span.meta["citation_counts"] = (report.citation_report or {}).get(
                "counts")

    n_ok = (report.citation_report or {}).get("counts", {}).get("ok", 0)
    decision = (f"Grounding ({mode}): {len(report.findings)} findings, "
                f"{len(report.tool_calls)} tool calls, "
                f"{n_ok} citations verified"
                + (f", {len(report.unresolved)} unresolved"
                   if report.unresolved else "")
                + (", PARSE FAILED" if report.parse_failed else ""))
    print(f"  [agent] {decision}")
    return {
        "grounded": report,
        # `standards` keeps its v1.0 meaning for the report appendix and for
        # observability: everything the grounding stage actually retrieved.
        "standards": report.retrieved,
        "top_k": report.max_k_requested,
        "decisions": state.get("decisions", []) + [decision],
    }


def export_node(state: AgentState) -> dict:
    t_start = state.get("_t_start")
    total_ms = int((time.time() - t_start) * 1000) if t_start else 0
    trace = state.get("trace")
    if trace is not None:
        log_trace(trace, state, image_path=state.get("image_path"))
    if not state.get("save", True):
        return {"saved": {}, "total_latency_ms": total_ms}
    detector = _get_detector("yolov8m.onnx")
    exporter = ReportExporter(state.get("save_dir", "reports"))
    saved = exporter.export(
        state["image_path"], state["det_result"], state["vlm_report"],
        state.get("standards") or [], draw_fn=detector.draw,
        decisions=state.get("decisions", []),
        grounded=state.get("grounded"),
    )
    for kind, path in saved.items():
        print(f"  [agent] saved [{kind}] {path}")
    return {"saved": saved, "total_latency_ms": total_ms}


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
    g.add_node("ground", ground_node)
    g.add_node("export", export_node)

    g.add_edge(START, "detect")
    g.add_conditional_edges(
        "detect", review_detections,
        {"retry": "lower_threshold", "continue": "analyze"},
    )
    g.add_edge("lower_threshold", "detect")
    g.add_edge("analyze", "assess_risk")
    g.add_edge("assess_risk", "ground")
    g.add_edge("ground", "export")
    g.add_edge("export", END)
    return g.compile()


class InspectionAgent:
    """High-level entry point: run the LangGraph inspection agent on an image."""

    def __init__(self):
        self.graph = build_graph()

    def run(self, image_path: str, conf_thres: float = 0.25,
            save: bool = True, save_dir: str = "reports",
            grounding_mode: str = DEFAULT_GROUNDING_MODE,
            trace: bool = False) -> dict:
        initial: AgentState = {
            "image_path": image_path,
            "conf_thres": conf_thres,
            "retry_count": 0,
            "decisions": [],
            "save": save,
            "save_dir": save_dir,
            "grounding_mode": grounding_mode,
            "_t_start": time.time(),
        }
        if trace:
            initial["trace"] = RequestTrace()
        final = self.graph.invoke(initial, config={"recursion_limit": 15})
        return final
