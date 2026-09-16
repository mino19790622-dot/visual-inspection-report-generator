"""Structured observability for the inspection pipeline.

Two log shapes coexist during the v2 upgrade:

* ``log_request(image_filename, state)`` — the v1.0 one-line summary, kept
  byte-compatible with the existing tests and reports.
* ``log_trace(trace, state)`` — the v2 per-stage trace (spans + token usage),
  written as one JSON line per request to ``logs/inspect_spans.jsonl``.

Both go through :func:`app.observability.redaction.redact` before touching
disk. See ``tests/test_log_redaction.py`` for the proof that no API key,
env value or image payload survives.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .pricing import cost_for
from .redaction import redact
from .spans import RequestTrace, Span  # noqa: F401  (re-exported)

DEFAULT_LOG_DIR = "logs"
LEGACY_LOG_NAME = "inspect.jsonl"
TRACE_LOG_NAME = "inspect_spans.jsonl"


def _log_path(filename: str) -> str:
    log_dir = Path(os.getenv("INSPECT_LOG_DIR", DEFAULT_LOG_DIR))
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        return str(log_dir / filename)
    except OSError:
        return str(Path("/tmp") / filename)


def _write(path: str, record: dict[str, Any]) -> None:
    """Serialise + redact + append. Never let logging fail a request."""
    try:
        line = json.dumps(redact(record), ensure_ascii=False)
    except (TypeError, ValueError):
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


# ----------------------------------------------------------------- v1 shape
def log_request(image_filename: str, state: dict) -> None:
    """Write one JSON line summarizing the inspection request (v1.0 shape)."""
    dr = state.get("det_result") or {}
    usage = state.get("vlm_usage") or {}
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "image": os.path.basename(image_filename) if image_filename else None,
        "risk_level": state.get("risk_level"),
        "detection": {
            "object_count": sum(dr.get("counts", {}).values()),
            "inference_ms": dr.get("inference_ms"),
        },
        "vlm": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "latency_ms": usage.get("latency_ms"),
            "cost_rmb": usage.get("cost_rmb"),
            # Absent means "the producer did not tell us", so we do not claim
            # known=True — an unpriced call must not read as a measured one.
            "cost_known": bool(usage.get("cost_known")),
        },
        "retrieval": {
            "top_k": state.get("top_k"),
            "standards_count": len(state.get("standards") or []),
        },
        "total_latency_ms": state.get("total_latency_ms"),
        "saved": list((state.get("saved") or {}).keys()),
    }
    _write(_log_path(LEGACY_LOG_NAME), record)


# ----------------------------------------------------------------- v2 shape
def log_trace(trace: RequestTrace, state: dict | None = None,
              image_path: str | None = None) -> dict:
    """Write the per-stage trace for one request; returns the record written."""
    state = state or {}
    grounded = state.get("grounded")
    record = trace.to_record()
    record["meta"].update({
        "image": os.path.basename(image_path) if image_path else None,
        "risk_level": state.get("risk_level"),
        "top_k": state.get("top_k"),
        "route_path": state.get("route_path"),
        "retry_count": state.get("retry_count"),
        "tool_calls_count": len(getattr(grounded, "tool_calls", []) or []),
        "finding_count": len(getattr(grounded, "findings", []) or []),
        "citation_counts": (getattr(grounded, "citation_report", {}) or {}
                            ).get("counts"),
    })
    _write(_log_path(TRACE_LOG_NAME), record)
    return record


def cost_breakdown(trace: RequestTrace) -> dict:
    """Attach a cost estimate to each span that recorded token usage."""
    out = {"per_span": [], "total": None, "currency": None, "all_known": True}
    total = 0.0
    for span in trace.spans:
        if span.input_tokens is None and span.output_tokens is None:
            continue
        c = cost_for(span.model, span.input_tokens, span.output_tokens)
        out["per_span"].append({"span": span.name, **c})
        if c["known"]:
            total += c["cost"]
        else:
            out["all_known"] = False
        if out["currency"] is None:
            out["currency"] = c["currency"]
    out["total"] = round(total, 6) if out["all_known"] else None
    return out
