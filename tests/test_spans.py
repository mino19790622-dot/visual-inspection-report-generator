"""Tests for per-stage tracing and the usage-to-cost conversion."""
import time

from app.observability import cost_breakdown
from app.observability.pricing import cost_for, load_pricing, price_for
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
    @staticmethod
    def _null_rated(tmp_path, monkeypatch, model: str = "some-model"):
        """Point PRICING_PATH at a file where `model` is listed but unpriced."""
        pricing = tmp_path / "p.yaml"
        pricing.write_text(
            "effective_date: '2026-09-15'\ncurrency: CNY\n"
            f"models:\n  {model}:\n    input_per_1k: null\n"
            "    output_per_1k: null\n", encoding="utf-8")
        monkeypatch.setenv("PRICING_PATH", str(pricing))

    def test_null_rate_is_reported_unknown_not_zero(self, tmp_path, monkeypatch):
        """A listed-but-unpriced model must be unknown, not free.

        0.0 would read as a measurement; None is the honest "not reported".
        This deliberately uses a fixture rather than a shipped model: it used
        to key off qwen-vl-max having null rates, which stopped being true once
        the real DashScope rates were filled in, and a test whose subject is
        "which real models are priced" cannot guard the mechanism.
        """
        self._null_rated(tmp_path, monkeypatch, model="some-model")
        result = cost_for("some-model", 1000, 1000)
        assert result["known"] is False
        assert result["cost"] is None

    def test_shipped_pricing_file_carries_provenance(self):
        """The rates that ship must say when and in what currency they apply.

        An unpinned rate is worse than a missing one: it looks authoritative
        while silently drifting out of date.
        """
        pricing = load_pricing()
        assert pricing.get("effective_date"), "effective_date is required"
        assert pricing.get("currency") == "CNY"
        models = pricing.get("models") or {}
        for m in ("qwen-vl-max", "qwen-plus", "text-embedding-v2",
                  "qwen-turbo"):
            assert m in models, f"{m} missing from the shipped rate snapshot"
            assert "input_per_1k" in models[m]
            assert "output_per_1k" in models[m]

    def test_unknown_model_is_unknown(self):
        assert cost_for("not-a-model", 1, 1)["known"] is False

    def test_missing_usage_is_unknown(self):
        assert cost_for("qwen-plus", None, None)["known"] is False

    def test_half_known_usage_is_unknown(self):
        """One leg missing must not be priced as zero.

        `entry is None or a is None and b is None` binds as
        `entry is None or (a is None and b is None)`, so a call that reported
        only completion_tokens slipped past the guard, charged the input leg as
        0 and still came back known=True — an under-report wearing the badge of
        a measurement.
        """
        for in_tok, out_tok in ((None, 500), (1000, None)):
            result = cost_for("qwen-plus", in_tok, out_tok)
            assert result["known"] is False, f"{in_tok}/{out_tok} must be unknown"
            assert result["cost"] is None

    def test_a_zero_leg_is_free_not_unknown(self, tmp_path, monkeypatch):
        """Missing is None; a genuinely free leg is 0 and must still price.

        An input-only model (text-embedding-v2) records output_per_1k: 0.
        Collapsing 0 into "unknown" would be its own wrong number.
        """
        pricing = tmp_path / "p.yaml"
        pricing.write_text(
            "effective_date: '2026-09-16'\ncurrency: CNY\n"
            "models:\n  text-embedding-v2:\n    input_per_1k: 0.0007\n"
            "    output_per_1k: 0\n", encoding="utf-8")
        monkeypatch.setenv("PRICING_PATH", str(pricing))
        result = cost_for("text-embedding-v2", 1000, 0)
        assert result["known"] is True
        assert result["cost"] == 0.0007

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

    def test_cost_breakdown_marks_unknown(self, tmp_path, monkeypatch):
        self._null_rated(tmp_path, monkeypatch, model="some-model")
        trace = RequestTrace()
        with trace.span("vlm_call", model="some-model") as s:
            trace.set_usage(s, {"prompt_tokens": 10, "completion_tokens": 5})
        out = cost_breakdown(trace)
        assert out["all_known"] is False
        assert out["total"] is None
        assert out["per_span"][0]["known"] is False
