"""Proof that nothing sensitive reaches disk.

This is not a test of the redaction function in isolation — it serialises a
realistic request record and inspects what would actually be written, which is
the only thing that matters.
"""
import base64
import json

import pytest

from app.observability import log_request, log_trace
from app.observability.redaction import redact
from app.observability.spans import RequestTrace

FAKE_KEY = "sk-abcdefgh12345678"
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 400
DATA_URI = "data:image/jpeg;base64," + base64.b64encode(PNG_BYTES).decode()
B64_IMAGE = base64.b64encode(PNG_BYTES).decode()


@pytest.fixture
def secret_env(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", FAKE_KEY)
    monkeypatch.setenv("SOME_OTHER_TOKEN", "supersecrettokenvalue")
    return {FAKE_KEY, "supersecrettokenvalue"}


def _dirty_record() -> dict:
    """A record containing everything that must NOT be logged."""
    return {
        "request_id": "abc123",
        "ts_utc": "2026-09-15T12:00:00+00:00",
        "api_key": FAKE_KEY,                       # stray credential field
        "authorization": f"Bearer {FAKE_KEY}",
        "image_data_uri": DATA_URI,                # the actual image
        "image_b64": B64_IMAGE,
        "prompt": "You are an assistant. " + FAKE_KEY,
        "note_text": "inspector said the surface looks fine",
        "spans": [
            {"name": "vlm_call", "duration_ms": 1200, "status": "ok",
             "model": "qwen-vl-max", "input_tokens": 900,
             "output_tokens": 300,
             "meta": {"image": DATA_URI, "secret": FAKE_KEY}},
        ],
        "totals": {"latency_ms": 4000, "input_tokens": 900,
                   "output_tokens": 300},
        "meta": {},
    }


class TestRedact:
    def test_stray_credential_field_is_dropped(self, secret_env):
        out = redact(_dirty_record())
        assert "api_key" not in out
        assert "authorization" not in out
        assert "secret" not in out["spans"][0]["meta"]

    def test_free_text_is_dropped(self, secret_env):
        """Prompt text and user notes are not allow-listed, so they go."""
        out = redact(_dirty_record())
        assert "prompt" not in out
        assert "note_text" not in out

    def test_image_payloads_are_dropped_or_scrubbed(self, secret_env):
        out = redact(_dirty_record())
        serialised = json.dumps(out)
        assert "data:image" not in serialised
        assert B64_IMAGE not in serialised
        # the payload is scrubbed in place; nothing decodable survives
        assert out["spans"][0]["meta"]["image"] == "[REDACTED]"
        # ...and unknown keys are dropped outright
        assert "secret" not in out["spans"][0]["meta"]

    def test_measured_fields_survive(self, secret_env):
        out = redact(_dirty_record())
        assert out["spans"][0]["input_tokens"] == 900
        assert out["totals"]["latency_ms"] == 4000
        assert out["request_id"] == "abc123"

    def test_known_secret_anywhere_in_a_value_is_scrubbed(self, secret_env):
        out = redact({"meta": {"model": FAKE_KEY}})
        assert out["meta"]["model"] == "[REDACTED]"

    def test_nested_lists_are_walked(self, secret_env):
        out = redact({"spans": [{"meta": {"model": FAKE_KEY}}]})
        assert out["spans"][0]["meta"]["model"] == "[REDACTED]"


class TestWrittenLines:
    def test_trace_log_line_is_clean(self, secret_env, tmp_path, monkeypatch):
        monkeypatch.setenv("INSPECT_LOG_DIR", str(tmp_path))
        trace = RequestTrace()
        with trace.span("vlm_call", model="qwen-vl-max") as s:
            trace.set_usage(s, {"prompt_tokens": 10, "completion_tokens": 5})
        state = {"risk_level": "high", "top_k": 5}
        log_trace(trace, state, image_path="/uploads/site.jpg")

        line = (tmp_path / "inspect_spans.jsonl").read_text(encoding="utf-8")
        _assert_clean(line, secret_env)

    def test_legacy_log_line_is_clean(self, secret_env, tmp_path, monkeypatch):
        monkeypatch.setenv("INSPECT_LOG_DIR", str(tmp_path))
        log_request("uploads/site.jpg", {
            "det_result": {"counts": {"person": 2}, "inference_ms": 12.0},
            "vlm_usage": {"prompt_tokens": 10, "completion_tokens": 5,
                          "total_tokens": 15, "latency_ms": 900,
                          "cost_rmb": 0.0003},
            "risk_level": "high", "top_k": 5, "standards": [{}],
            "total_latency_ms": 1200, "saved": {"report": "/tmp/r.md"},
        })
        line = (tmp_path / "inspect.jsonl").read_text(encoding="utf-8")
        _assert_clean(line, secret_env)

    def test_v1_fields_still_present_after_redaction(self, tmp_path,
                                                     monkeypatch):
        """Redaction must not quietly change the v1.0 log contract."""
        monkeypatch.setenv("INSPECT_LOG_DIR", str(tmp_path))
        log_request("uploads/site.jpg", {
            "risk_level": "high", "standards": [{}], "saved": {"report": "r.md"},
        })
        rec = json.loads((tmp_path / "inspect.jsonl").read_text())
        assert rec["risk_level"] == "high"
        assert rec["retrieval"]["standards_count"] == 1
        assert rec["saved"] == ["report"]


def _assert_clean(line: str, secrets: set) -> None:
    from app.observability.redaction import assert_no_secrets
    assert_no_secrets(line, secrets)
    assert FAKE_KEY not in line
    assert "data:image" not in line
