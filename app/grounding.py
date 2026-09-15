# app/grounding.py
"""Grounded finding generation: a text agent with tools, on top of the VLM.

Why this layer exists
---------------------
v1.0 had the VLM write the whole report, then appended retrieved standards as a
human-readable appendix. The retrieved text never reached the model, so the
system had no grounded generation and nothing to validate a claim against.

This node keeps the VLM output untouched (it stays the perception layer and the
existing 10-image golden set / judge remain a valid regression net) and adds a
second stage: a **text** agent that reads the VLM narrative, decides which
standard clauses it needs, retrieves them through a tool, and emits structured
findings whose citations point at chunks it actually retrieved.

Splitting vision from reasoning is also a cost decision: the tool loop runs on a
cheap text model instead of paying VLM rates for every reasoning round.

Control modes
-------------
``agentic``  the model decides whether / how much to retrieve (default).
``fixed``    the retrieval tool is removed and ``k=MAX_K`` clauses are injected
             up front. This is the control arm for the dynamic-vs-fixed top-k
             experiment (see ``eval/ablation_topk.py``).
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from app.citations import check_findings
from app.tools import (
    AGENTIC_TOOL_SPECS,
    FIXED_MODE_TOOL_SPECS,
    MAX_K,
    ToolRegistry,
    parse_tool_arguments,
)

load_dotenv()
logger = logging.getLogger(__name__)

GROUNDING_MODEL = os.getenv("GROUNDING_MODEL", "qwen-plus")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MAX_ROUNDS = 6          # safety valve: hard cap on loop iterations
DEFAULT_MAX_CALLS = 4   # retrieval-call budget


# --------------------------------------------------------------------- schema
class Citation(BaseModel):
    chunk_id: str
    quote: str = ""
    standard: str | None = None


class Finding(BaseModel):
    id: str
    severity: Literal["low", "medium", "high", "critical"]
    description: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    citation: Citation | None = None
    # Filled in after generation by verify(); never set by the model.
    citation_status: str | None = None
    citation_reason: str | None = None


class GroundedReport(BaseModel):
    findings: list[Finding] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    retrieved: list[dict[str, Any]] = Field(default_factory=list)
    citation_report: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)
    rounds: int = 0
    mode: str = "agentic"
    parse_failed: bool = False
    failure_reason: str | None = None

    @property
    def max_k_requested(self) -> int:
        ks = [c.get("args", {}).get("k", 0)
              for c in self.tool_calls if c.get("tool") == "retrieve_standards"]
        return max(ks) if ks else 0


# --------------------------------------------------------------------- prompt
SYSTEM_PROMPT = """You are the grounding stage of a visual inspection system.

You receive a scene narrative written by a vision model, plus what an object
detector found. Your job is to turn that into findings that a human inspector
can act on — each one backed by a written standard.

Rules:
1. Before writing a finding, decide whether you actually need a written basis.
   Use `retrieve_standards` only for concerns you intend to support with a
   clause. Ask for the smallest `k` that can do the job.
2. Every finding MUST carry a `citation` whose `chunk_id` is one you actually
   got back from `retrieve_standards`, and whose `quote` is copied verbatim
   from that clause.
3. Never invent a chunk_id, a clause number, or a quote. If you cannot find a
   written basis, put the observation in `unresolved` and say why.
4. Do not restate the narrative. Only report things that need a decision.
5. Finish by calling `submit_findings` exactly once."""


def _user_content(narrative: str, detection_summary: str,
                  injected: list[dict] | None) -> str:
    parts = [
        "## Detector summary\n" + (detection_summary or "no objects detected"),
        "\n## Vision narrative\n" + (narrative or "(empty)"),
    ]
    if injected:
        blocks = "\n\n".join(
            f"[{c.get('chunk_id')}] {c.get('standard')}\n{c.get('text', '')}"
            for c in injected)
        parts.append(
            "\n## Standard clauses already retrieved (cite these by chunk_id)\n"
            + blocks
            + "\n\nDo not ask for more clauses; submit findings using only "
              "the chunk_ids above.")
    else:
        parts.append(
            "\nRetrieve the clauses you need, then call `submit_findings`.")
    return "\n".join(parts)


# ---------------------------------------------------------------------- agent
class GroundingAgent:
    """Runs the tool-calling loop and returns a verified GroundedReport."""

    def __init__(self, retriever: Any, client: Any = None,
                 model: str = GROUNDING_MODEL, max_calls: int = DEFAULT_MAX_CALLS):
        self.retriever = retriever
        self.model = model
        self.max_calls = max_calls
        self._client = client

    # -- client -------------------------------------------------------- #
    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI  # imported lazily: keeps tests fast
            self._client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),
                                  base_url=BASE_URL)
        return self._client

    # -- main ---------------------------------------------------------- #
    def run(self, narrative: str, detection_summary: str = "",
            mode: str = "agentic", temperature: float = 0.0) -> GroundedReport:
        registry = ToolRegistry(self.retriever, max_calls=self.max_calls)
        injected: list[dict] | None = None
        specs = AGENTIC_TOOL_SPECS

        if mode == "fixed":
            # control arm: pre-retrieve at max depth, then take the tool away
            injected = [dict(c) for c in self.retriever.retrieve(
                f"{detection_summary}\n{narrative}", k=MAX_K)]
            registry._add_retrieved(injected)
            registry.calls.append(
                {"tool": "retrieve_standards",
                 "args": {"query": "(fixed-arm pre-retrieval)", "k": MAX_K},
                 "n_results": len(injected)})
            specs = FIXED_MODE_TOOL_SPECS

        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": _user_content(narrative, detection_summary, injected)},
        ]

        usage = {"prompt_tokens": 0, "completion_tokens": 0,
                 "total_tokens": 0, "latency_ms": 0}
        submitted: dict | None = None
        rounds = 0
        failure_reason = None

        for _ in range(MAX_ROUNDS):
            rounds += 1
            t0 = time.time()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, tools=specs,
                    tool_choice="auto", temperature=temperature)
            except Exception as exc:
                logger.exception("grounding LLM call failed")
                failure_reason = f"llm call failed: {exc}"
                break
            usage["latency_ms"] += int((time.time() - t0) * 1000)
            _accumulate_usage(usage, getattr(resp, "usage", None))

            msg = resp.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []
            if not tool_calls:
                # Model answered in prose instead of calling a tool. Do NOT
                # try to parse prose into findings — that is how you end up
                # with unverifiable citations.
                failure_reason = "model returned no tool call"
                break

            messages.append(_assistant_message(msg, tool_calls))

            for tc in tool_calls:
                name = tc.function.name
                args, err = parse_tool_arguments(tc.function.arguments)
                if err:
                    registry.record_raw_call(name, {})
                    messages.append(_tool_message(
                        tc.id, {"error": err,
                                "hint": "retry with valid JSON arguments"}))
                    continue

                if name == "submit_findings":
                    submitted = args
                    messages.append(_tool_message(tc.id, {"received": True}))
                    continue

                result = registry.execute(name, args)
                if result.get("error"):
                    registry.record_raw_call(name, args)
                messages.append(_tool_message(tc.id, result))

            if submitted is not None:
                break

        report = self._build(submitted, registry, rounds, mode)
        report.usage = usage
        if failure_reason and submitted is None:
            report.parse_failed = True
            report.failure_reason = failure_reason
        return report

    # -- assembly ------------------------------------------------------ #
    def _build(self, submitted: dict | None, registry: ToolRegistry,
               rounds: int, mode: str) -> GroundedReport:
        if submitted is None:
            return GroundedReport(
                findings=[], unresolved=[], tool_calls=registry.call_log(),
                retrieved=registry.retrieved, rounds=rounds, mode=mode,
                parse_failed=True,
                failure_reason="model never called submit_findings")

        try:
            findings = [Finding(**f) for f in (submitted.get("findings") or [])]
            unresolved = [str(u) for u in (submitted.get("unresolved") or [])]
        except (ValidationError, TypeError) as exc:
            return GroundedReport(
                findings=[], unresolved=[],
                tool_calls=registry.call_log(), retrieved=registry.retrieved,
                rounds=rounds, mode=mode, parse_failed=True,
                failure_reason=f"findings failed schema validation: {exc}")

        report = GroundedReport(
            findings=findings, unresolved=unresolved,
            tool_calls=registry.call_log(), retrieved=registry.retrieved,
            rounds=rounds, mode=mode)

        # Deterministic verification against what was actually retrieved.
        # This is the L1 layer; the LLM judge is a later, softer layer.
        verification = check_findings([f.model_dump() for f in findings],
                                      registry.retrieved)
        for f, detail in zip(findings, verification["detail"], strict=False):
            f.citation_status = detail["status"]
            f.citation_reason = detail["reason"]
        report.citation_report = verification
        return report


# ------------------------------------------------------------------- helpers
def _assistant_message(msg: Any, tool_calls: list) -> dict:
    out: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
    out["tool_calls"] = [
        {"id": tc.id, "type": "function",
         "function": {"name": tc.function.name,
                      "arguments": tc.function.arguments}}
        for tc in tool_calls
    ]
    return out


def _tool_message(tool_call_id: str, payload: dict) -> dict:
    return {"role": "tool", "tool_call_id": tool_call_id,
            "content": json.dumps(payload, ensure_ascii=False)}


def _accumulate_usage(usage: dict, resp_usage: Any) -> None:
    if resp_usage is None:
        return
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        usage[key] += getattr(resp_usage, key, 0) or 0


def summarise_detections(det_result: dict | None) -> str:
    """Compact detector summary handed to the grounding agent."""
    det = det_result or {}
    counts = det.get("counts") or {}
    if not counts:
        return "no objects detected"
    items = ", ".join(f"{k}×{v}" for k, v in sorted(counts.items()))
    dets = det.get("detections") or []
    if dets:
        confs = [d.get("confidence") for d in dets if "confidence" in d]
        if confs:
            items += f" (confidence {min(confs):.2f}–{max(confs):.2f})"
    return items
