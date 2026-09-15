# app/observability/spans.py
"""Per-stage latency / token accounting for one inspection request.

Design rules (these matter more than the code):

1. **Token counts come from the API response only.** Never estimated, never
   guessed. Image token billing rules change; a hand-rolled estimate is a
   fabricated number. If the provider returns no usage, the field stays None.
2. **Cost is derived, not measured.** Usage is recorded per stage; money is
   computed afterwards from ``config/pricing.yaml`` so a price change can be
   re-applied to historical runs.
3. **A span always closes**, even on exception — otherwise a failed request
   silently loses its most interesting timing data.
"""
from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator


@dataclass
class Span:
    name: str
    start_offset_ms: float = 0.0
    duration_ms: float = 0.0
    status: str = "ok"
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    retries: int = 0
    error_type: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start_offset_ms": round(self.start_offset_ms, 1),
            "duration_ms": round(self.duration_ms, 1),
            "status": self.status,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "retries": self.retries,
            "error_type": self.error_type,
            "meta": self.meta,
        }


@dataclass
class RequestTrace:
    """Collects spans for one request and serialises to the JSONL schema."""

    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())
    spans: list[Span] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._t0 = time.perf_counter()

    @contextmanager
    def span(self, name: str, **meta: Any) -> Iterator[Span]:
        now = time.perf_counter()
        s = Span(name=name, start_offset_ms=(now - self._t0) * 1000, meta=meta)
        self.spans.append(s)
        try:
            yield s
        except Exception as exc:
            s.status = "error"
            s.error_type = type(exc).__name__
            raise
        finally:
            s.duration_ms = (time.perf_counter() - now) * 1000

    # -- convenience --------------------------------------------------- #
    def set_usage(self, span: Span, usage: dict | None) -> None:
        """Copy token counts out of a provider usage object/dict."""
        if not usage:
            return
        if not isinstance(usage, dict):
            usage = {k: getattr(usage, k, None)
                     for k in ("prompt_tokens", "completion_tokens",
                               "total_tokens")}
        span.input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
        span.output_tokens = (usage.get("completion_tokens")
                              or usage.get("output_tokens"))

    def by_name(self, name: str) -> Span | None:
        for s in self.spans:
            if s.name == name:
                return s
        return None

    def total_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000

    def totals(self) -> dict[str, Any]:
        inp = sum(s.input_tokens or 0 for s in self.spans)
        out = sum(s.output_tokens or 0 for s in self.spans)
        return {
            "latency_ms": round(self.total_ms(), 1),
            "input_tokens": inp or None,
            "output_tokens": out or None,
            "stages": len(self.spans),
            "errors": sum(1 for s in self.spans if s.status == "error"),
        }

    def to_record(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "ts_utc": self.started_at,
            "spans": [s.to_dict() for s in self.spans],
            "totals": self.totals(),
            "meta": self.meta,
        }
