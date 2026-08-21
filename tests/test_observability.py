"""Unit tests for the observability module (no LLM calls)."""
import json

from app.observability import log_request


def _fake_state() -> dict:
    return {
        "det_result": {
            "counts": {"car": 3, "person": 2},
            "inference_ms": 47.0,
        },
        "vlm_usage": {
            "prompt_tokens": 1200,
            "completion_tokens": 350,
            "total_tokens": 1550,
            "latency_ms": 1800,
            "cost_rmb": 0.031,
        },
        "risk_level": "high",
        "top_k": 5,
        "standards": [{}, {}, {}, {}, {}],
        "total_latency_ms": 2400,
        "saved": {"report": "/tmp/r.md", "annotated": "/tmp/r.jpg"},
    }


def test_log_request_writes_jsonl(tmp_path, monkeypatch):
    monkeypatch.setenv("INSPECT_LOG_DIR", str(tmp_path))
    log_request("uploads/abc123.jpg", _fake_state())
    log_file = tmp_path / "inspect.jsonl"
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    rec = json.loads(lines[0])
    # structure
    assert rec["image"] == "abc123.jpg"
    assert rec["risk_level"] == "high"
    assert rec["detection"]["object_count"] == 5
    assert rec["vlm"]["total_tokens"] == 1550
    assert rec["vlm"]["cost_rmb"] == 0.031
    assert rec["retrieval"]["standards_count"] == 5
    assert rec["total_latency_ms"] == 2400
    assert rec["saved"] == ["report", "annotated"]
    assert rec["ts"].endswith("+00:00") or rec["ts"].endswith("Z")


def test_log_request_graceful_on_missing_state(tmp_path, monkeypatch):
    """Empty/missing fields should log without raising."""
    monkeypatch.setenv("INSPECT_LOG_DIR", str(tmp_path))
    log_request("uploads/empty.jpg", {})  # all fields missing
    log_file = tmp_path / "inspect.jsonl"
    rec = json.loads(log_file.read_text().strip())
    assert rec["image"] == "empty.jpg"
    assert rec["risk_level"] is None
    assert rec["detection"]["object_count"] == 0
    assert rec["vlm"]["total_tokens"] is None
    assert rec["saved"] == []


def test_log_request_falls_back_to_tmp_when_dir_unwritable(monkeypatch):
    """If the configured log dir can't be created, fall back to /tmp."""
    monkeypatch.setenv("INSPECT_LOG_DIR", "/this/does/not/exist/anywhere")
    # write once — should not raise even if mkdir fails
    log_request("uploads/x.jpg", _fake_state())
    # the fallback path is /tmp/inspect.jsonl; we just verify no exception
