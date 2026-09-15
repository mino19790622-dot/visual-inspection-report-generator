# app/tools.py
"""Tool schemas and a guarded execution registry for the grounding agent.

Why tools here at all
---------------------
v1.0 retrieved standards *after* generation, and the retrieval depth was chosen
by a hard-coded rule (``k = 5/4/3`` from the VLM risk level). Moving retrieval
in front of generation creates a circular dependency: the risk level comes out
of the VLM report, but the VLM now needs the retrieved context to write that
report. Letting the model decide *whether* and *how much* to retrieve is the
natural resolution — which is exactly tool calling, not a bolted-on feature.

Guardrails that stay in code (never delegated to the model)
-----------------------------------------------------------
* ``k`` is clamped to ``[1, MAX_K]``.
* The tool-call budget is capped (``max_calls``) so a confused model cannot
  loop and burn money.
* Unknown tools and malformed arguments return a structured error instead of
  raising, so the loop can hand the error back to the model and continue.
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

MAX_K = 5
DEFAULT_MAX_CALLS = 4

# OpenAI-compatible function-calling schema. Kept as plain dicts so it can be
# handed straight to the DashScope/OpenAI client.
RETRIEVE_STANDARDS_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "retrieve_standards",
        "description": (
            "Retrieve inspection-standard clauses relevant to a specific aspect "
            "of the scene. Call it once per distinct concern you need a written "
            "basis for. Ask for more clauses (higher k) only when the aspect is "
            "safety-critical or ambiguous; ask for 1-2 when a single clause is "
            "enough. Do NOT call it for things you already have a basis for."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "What to search for, phrased as the inspection concern "
                        "(e.g. 'workers on scaffolding without head protection'). "
                        "Write it in the language of the standards, not of the "
                        "image."),
                },
                "k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_K,
                    "description": (
                        f"How many clauses to return (1-{MAX_K}). Choose the "
                        "smallest number that can support the finding."),
                },
            },
            "required": ["query"],
        },
    },
}

SUBMIT_FINDINGS_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_findings",
        "description": (
            "Submit the final grounded findings. Call this exactly once, after "
            "you have retrieved the clauses you intend to cite. Every finding "
            "MUST cite a clause you actually retrieved, using its chunk_id and a "
            "verbatim quote from it. If you cannot find a written basis for an "
            "observation, put it in `unresolved` instead of inventing a citation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "severity": {
                                "type": "string",
                                "enum": ["low", "medium", "high", "critical"],
                            },
                            "description": {"type": "string"},
                            "confidence": {
                                "type": "number", "minimum": 0.0, "maximum": 1.0,
                            },
                            "citation": {
                                "type": "object",
                                "properties": {
                                    "chunk_id": {
                                        "type": "string",
                                        "description": (
                                            "The `chunk_id` field returned by "
                                            "retrieve_standards, verbatim."),
                                    },
                                    "quote": {
                                        "type": "string",
                                        "description": (
                                            "A short verbatim excerpt from that "
                                            "clause supporting the finding."),
                                    },
                                },
                                "required": ["chunk_id", "quote"],
                            },
                        },
                        "required": ["id", "severity", "description", "citation"],
                    },
                },
                "unresolved": {
                    "type": "array",
                    "description": (
                        "Observations you noticed but could NOT back with any "
                        "retrieved clause. Honesty here is required; do not "
                        "fabricate a citation to avoid this list."),
                    "items": {"type": "string"},
                },
            },
            "required": ["findings"],
        },
    },
}

RETRIEVAL_TOOL_SPECS = [RETRIEVE_STANDARDS_SPEC]
AGENTIC_TOOL_SPECS = [RETRIEVE_STANDARDS_SPEC, SUBMIT_FINDINGS_SPEC]
# Control arm for the dynamic-vs-fixed top-k experiment: the model gets no
# retrieval tool, and fixed-k chunks are injected into the prompt instead.
FIXED_MODE_TOOL_SPECS = [SUBMIT_FINDINGS_SPEC]


class ToolRegistry:
    """Executes tools with argument validation, clamping and a call budget."""

    def __init__(self, retriever: Any, max_k: int = MAX_K,
                 max_calls: int = DEFAULT_MAX_CALLS):
        self._retriever = retriever
        self.max_k = max_k
        self.max_calls = max_calls
        self.calls: list[dict[str, Any]] = []
        self.retrieved: list[dict[str, Any]] = []  # union across all calls

    # -- introspection ------------------------------------------------- #
    @property
    def calls_used(self) -> int:
        """Only *retrieval* calls count against the budget."""
        return sum(1 for c in self.calls if c["tool"] == "retrieve_standards")

    @property
    def budget_exhausted(self) -> bool:
        return self.calls_used >= self.max_calls

    # -- validation ---------------------------------------------------- #
    def _validate(self, name: str, raw_args: Any) -> tuple[dict, list[str]]:
        issues: list[str] = []
        if not isinstance(raw_args, dict):
            return {}, [f"arguments must be an object, got {type(raw_args).__name__}"]
        args = dict(raw_args)

        if name == "retrieve_standards":
            query = args.get("query")
            if not isinstance(query, str) or not query.strip():
                issues.append("`query` must be a non-empty string")
            k = args.get("k", 3)
            if isinstance(k, bool) or not isinstance(k, int):
                issues.append(f"`k` must be an integer, got {k!r}")
                k = 3
            # clamp rather than fail: the boundary is a code-side guardrail,
            # the model only gets a *say* in the value
            clamped = max(1, min(self.max_k, k))
            if clamped != k:
                logger.info("clamped k=%s -> %s", k, clamped)
            args["k"] = clamped
            args["query"] = query.strip() if isinstance(query, str) else query
        return args, issues

    # -- execution ----------------------------------------------------- #
    def execute(self, name: str, raw_args: Any) -> dict[str, Any]:
        """Run one tool call. Never raises; returns a structured result."""
        if name == "submit_findings":
            return {"submitted": True, "arguments": raw_args}

        if name != "retrieve_standards":
            return {"error": f"unknown tool: {name}",
                    "available": ["retrieve_standards", "submit_findings"]}

        if self.budget_exhausted:
            return {"error": f"tool budget exhausted ({self.max_calls} calls)",
                    "hint": "submit your findings with what you already have"}

        args, issues = self._validate(name, raw_args)
        if issues:
            return {"error": "invalid arguments", "issues": issues,
                    "hint": ("call retrieve_standards with a non-empty `query` "
                             f"and an integer `k` between 1 and {self.max_k}")}

        try:
            chunks = self._retriever.retrieve(args["query"], k=args["k"])
        except Exception as exc:  # never let a tool failure kill the pipeline
            logger.exception("retrieve_standards failed")
            return {"error": f"retrieval failed: {exc}",
                    "hint": "retry with a different query, or submit without it"}

        results = [
            {"chunk_id": c.get("chunk_id"), "standard": c.get("standard"),
             "source": c.get("source"), "text": c.get("text", ""),
             "distance": c.get("distance")}
            for c in chunks
        ]
        self._add_retrieved(results)
        self.calls.append({"tool": name, "args": args,
                           "n_results": len(results)})
        return {"query": args["query"], "k": args["k"], "clauses": results}

    def _add_retrieved(self, chunks: list[dict]) -> None:
        seen = {c.get("chunk_id") for c in self.retrieved}
        for c in chunks:
            if c.get("chunk_id") not in seen:
                self.retrieved.append(c)
                seen.add(c.get("chunk_id"))

    def record_raw_call(self, name: str, args: dict) -> None:
        """Record a call that short-circuited (e.g. a validation error)."""
        self.calls.append({"tool": name, "args": args, "n_results": 0})

    def call_log(self) -> list[dict[str, Any]]:
        return list(self.calls)


def parse_tool_arguments(raw: str | None) -> tuple[dict | None, str | None]:
    """Parse a model's ``function.arguments`` string. Returns (args, error)."""
    if not raw:
        return None, "empty arguments"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"arguments are not valid JSON: {exc}"
    if not isinstance(parsed, dict):
        return None, f"arguments must be an object, got {type(parsed).__name__}"
    return parsed, None


def build_registry(retriever: Any, **kwargs) -> ToolRegistry:
    """Factory kept separate so tests can patch it without importing clients."""
    return ToolRegistry(retriever, **kwargs)
