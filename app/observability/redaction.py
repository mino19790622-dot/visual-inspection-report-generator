# app/observability/redaction.py
"""Field-whitelist redaction for anything that leaves the process.

Allow-list, not block-list: a block-list only stops the leaks you already
thought of. Anything not named here is dropped before serialisation, so adding
a new field to a span requires an explicit decision to expose it.

The point is provable, not aspirational: ``tests/test_log_redaction.py`` runs a
real request-shaped record through the filter and asserts that no API key, no
env value and no base64 blob survives.
"""
from __future__ import annotations

import base64
import os
import re
from typing import Any

# Top-level keys allowed in a request log record.
ALLOWED_TOP_LEVEL = {
    "request_id", "ts_utc", "spans", "totals", "meta",
    # legacy v1.0 log_request schema
    "ts", "image", "risk_level", "detection", "vlm", "retrieval",
    "total_latency_ms", "saved",
}

# Keys allowed anywhere inside the record.
ALLOWED_KEYS = {
    "request_id", "ts_utc", "ts", "spans", "totals", "meta",
    "name", "start_offset_ms", "duration_ms", "status", "model",
    "input_tokens", "output_tokens", "retries", "error_type",
    "latency_ms", "stages", "errors",
    "image", "image_sha256_prefix", "image_count", "image_resolutions",
    "risk_level", "top_k", "route_path", "retry_count",
    "detection", "object_count", "inference_ms", "counts",
    "vlm", "prompt_tokens", "completion_tokens", "total_tokens", "cost_rmb",
    "cost_known",
    "retrieval", "standards_count", "n_retrieved", "mode",
    "finding_count", "citation_counts", "attribution_pass_rate",
    "tool", "tool_calls_count", "n_results", "args", "query", "k",
    "report_sha256_prefix", "bytes", "saved", "flags",
    "app_version", "git_sha", "model_versions",
    "cost", "currency", "known", "estimated",
}

# Azure/OpenAI-style keys and DashScope keys both start with a recognisable
# prefix; anything shaped like a long random token is treated as sensitive.
_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"\bDASHSCOPE[_-]?API[_-]?KEY\b", re.IGNORECASE),
    re.compile(r"\b(?:api[_-]?key|authorization|bearer)\b", re.IGNORECASE),
)
_DATA_URI = re.compile(r"data:image/[a-zA-Z0-9.+-]+;base64,")
_LONG_B64 = re.compile(r"^[A-Za-z0-9+/]{200,}={0,2}$")

REDACTED = "[REDACTED]"
_MIN_ENV_VALUE_LEN = 8


def _looks_secret(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if _DATA_URI.search(value) or _LONG_B64.match(value.strip()):
        return True
    return any(p.search(value) for p in _SECRET_PATTERNS)


def _env_values() -> set[str]:
    """Values of any env var whose name hints at a credential."""
    out = set()
    for key, value in os.environ.items():
        if not value or len(value) < _MIN_ENV_VALUE_LEN:
            continue
        if any(t in key.upper() for t in
               ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")):
            out.add(value)
    return out


def redact(record: dict[str, Any], _secrets: set[str] | None = None) -> dict[str, Any]:
    """Return a copy of ``record`` containing only allow-listed fields.

    Values that look like secrets are replaced rather than dropped, so a
    redaction is visible in the log instead of silently vanishing.
    """
    secrets = _secrets if _secrets is not None else _env_values()
    return _clean(record, secrets, top=True)


def _clean(node: Any, secrets: set[str], top: bool = False) -> Any:
    """Recursively copy ``node``, keeping only allow-listed keys and values.

    ``top`` chooses the allow-list: the outermost dict is filtered against
    ``ALLOWED_TOP_LEVEL``, every nested dict against ``ALLOWED_KEYS``. The two
    sets are not the same (``meta``, ``totals``, ``saved`` ... exist only at the
    top level), so this flag is what stops a nested dict from smuggling in a
    key that is legal only at the root. Lists are walked element-wise; scalars
    that are not strings pass through unchanged.
    """
    if isinstance(node, dict):
        allowed = ALLOWED_TOP_LEVEL if top else ALLOWED_KEYS
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key not in allowed:
                continue
            out[key] = _clean(value, secrets)
        return out
    if isinstance(node, list):
        return [_clean(v, secrets) for v in node]
    if isinstance(node, str):
        if node in secrets or _looks_secret(node):
            return REDACTED
        return node
    return node


def assert_no_secrets(text: str, secrets: set[str] | None = None) -> None:
    """Raise AssertionError if ``text`` contains a secret or an image payload.

    Used by tests against the *serialised* log line, i.e. it checks what
    actually lands on disk rather than trusting the redaction function.
    """
    secrets = secrets if secrets is not None else _env_values()
    for value in secrets:
        if value in text:
            raise AssertionError("log contains a value from the environment")
    if _DATA_URI.search(text):
        raise AssertionError("log contains a base64 image data URI")
    for line in text.splitlines():
        for token in re.findall(r"[A-Za-z0-9+/]{200,}={0,2}", line):
            try:
                decoded = base64.b64decode(token, validate=True)
            except Exception:
                continue
            if decoded[:4] in (b"\x89PNG", b"\xff\xd8\xff", b"GIF8", b"RIFF"):
                raise AssertionError("log contains base64-encoded image bytes")
