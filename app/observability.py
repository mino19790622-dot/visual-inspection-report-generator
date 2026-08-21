"""Structured observability for the inspection pipeline.

Emits one JSON line per /inspect request to logs/inspect.jsonl with:
  - timestamp (UTC ISO 8601)
  - image filename
  - risk_level
  - detection object_count + inference_ms
  - vlm: prompt_tokens, completion_tokens, total_tokens, latency_ms, cost_rmb
  - retrieval: standards_count
  - total_latency_ms (end-to-end)
  - saved_files paths

Usage:
    from app.observability import log_request
    log_request("uploads/abc.jpg", final_state)
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LOG_DIR = "logs"


def _log_path() -> str:
    """Resolve the log file path lazily (so tests can change env at runtime)."""
    log_dir = Path(os.getenv("INSPECT_LOG_DIR", DEFAULT_LOG_DIR))
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        return str(log_dir / "inspect.jsonl")
    except OSError:
        return "/tmp/inspect.jsonl"


def log_request(image_filename: str, state: dict) -> None:
    """Write one JSON line summarizing the inspection request."""
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
        },
        "retrieval": {
            "top_k": state.get("top_k"),
            "standards_count": len(state.get("standards") or []),
        },
        "total_latency_ms": state.get("total_latency_ms"),
        "saved": list((state.get("saved") or {}).keys()),
    }
    path = _log_path()
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        # never let logging fail the request
        pass
