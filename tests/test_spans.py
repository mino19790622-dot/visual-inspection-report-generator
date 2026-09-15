"""Tests for per-stage tracing and the usage-to-cost conversion."""
import time

from app.observability import cost_breakdown
from app.observability.pricing import cost_for, price_for
from app.observability.spans import RequestTrace


class TestSpans:
    def test_span_records_duration_and_offset(self):
        trace = RequestTrace()
        with trace.span("detect") as s:
            time.sleep(0.01)
        assert s.duration_ms >= 10
        assert s.start_offset_ms < s.duration_ms + 1
        assert s.status == "ok"

    def test_span_records_error_and_reraises(self):
        trace = RequestTrace()
        with_trace = trace.span("vlm_call")
        with __import__("pytest").raises(ValueError):
            with with_trace as s:
                raise ValueError("boom")
        assert s.status == "error"
        assert s.error_type == "ValueError"

    def test_usage_is_attached_to_the_span(self):
        trace = RequestTrace()
        with trace.span("vlm_call", model="qwen-vl-max") as s:
            trace.set_usage(s, {"prompt_tokens": 100, "completion_tokens": 20})
        assert s.input_tokens == 100 and s.output_tokens == 20

    def test_usage_accepts_provider_objects(self):
        class U:
            prompt_tokens = 7
            completion_tokens = 3
            total_tokens = 10

        trace = RequestTrace()
        with trace.span("ground", model="qwen-plus") as s:
            trace.set_usage(s, U())
        assert s.input_tokens == 7 and s.output_tokens == 3

    def test_missing_usage_leaves_tokens_none_not_zero(self):
        """None means 'not reported'; 0 would be a measurement."""
        trace = RequestTrace()
        with trace.span("detect") as s:
            trace.set_usage(s, None)
        assert s.input_tokens is None

    def test_totals_aggregate_across_spans(self):
        trace = RequestTrace()
        with trace.span("a") as s:
            trace.set_usage(s, {"prompt_tokens": 10, "completion_tokens": 1})
        with trace.span("b") as s:
            trace.set_usage(s, {"prompt_tokens": 20, "completion_tokens": 4})
        totals = trace.to_record()["totals"]
        assert totals["input_tokens"] == 30
        assert totals["output_tokens"] == 5
        assert totals["stages"] == 2
        assert totals["errors"] == 0

    def test_by_name(self):
        trace = RequestTrace()
        with trace.span("vlm_call"):
            pass
        assert trace.by_name("vlm_call") is not None
        assert trace.by_name("nope") is None

    def test_record_shape(self):
        trace = RequestTrace()
        with trace.span("detect"):
            pass
        rec = trace.to_record()
        assert {"request_id", "ts_utc", "spans", "totals", "meta"} <= set(rec)


class TestPricing:
    def test_unconfigured_rate_is_reported_unknown_not_zero(self):
        """config/pricing.yaml ships with null rates on purpose."""
        result = cost_for("qwen-vl-max", 1000, 1000)
        assert result["known"] is False
        assert result["cost"] is None

    def test_unknown_model_is_unknown(self):
        assert cost_for("not-a-model", 1, 1)["known"] is False

    def test_missing_usage_is_unknown(self):
        assert cost_for("qwen-plus", None, None)["known"] is False

    def test_known_rate_is_applied(self, tmp_path, monkeypatch):
        pricing = tmp_path / "p.yaml"
        pricing.write_text(
            "effective_date: '2026-09-15'\ncurrency: CNY\n"
            "models:\n  qwen-plus:\n    input_per_1k: 0.004\n"
            "    output_per_1k: 0.012\n", encoding="utf-8")
        monkeypatch.setenv("PRICING_PATH", str(pricing))
        result = cost_for("qwen-plus", 1000, 500)
        assert result["known"] is True
        assert result["cost"] == 0.01  # 1*0.004 + 0.5*0.012
        assert result["effective_date"] == "2026-09-15"

    def test_price_for_returns_none_when_absent(self):
        assert price_for(None) is None

    def test_cost_breakdown_marks_unknown(self):
        trace = RequestTrace()
        with trace.span("vlm_call", model="qwen-vl-max") as s:
            trace.set_usage(s, {"prompt_tokens": 10, "completion_tokens": 5})
        out = cost_breakdown(trace)
        assert out["all_known"] is False
        assert out["total"] is None
        assert out["per_span"][0]["known"] is False
