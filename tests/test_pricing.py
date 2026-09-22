# tests/test_pricing.py
"""Unit tests for app/observability/pricing.py — no network, no LLM.

The module encodes two invariants that are easy to break silently, so they are
pinned here explicitly:

  1. ``None`` token counts mean "not measured" and must yield ``known=False``.
     A *measured* zero is a different fact and must still price (and may price
     to 0.0, as the input-only embedding model does).
  2. ``known=False`` must never be rendered as ``cost=0.0``: "no price
     configured" and "this call was free" are different facts.

Both were live bugs in this module's history, so they get regression tests
rather than incidental coverage.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from app.observability.pricing import cost_for, load_pricing, price_for

SNAPSHOT = textwrap.dedent(
    """
    effective_date: 2026-09-16
    currency: CNY
    models:
      input-only:
        input_per_1k: 0.0007
        output_per_1k: 0
      null-output-rate:
        input_per_1k: 0.001
        output_per_1k: null
    """
)


@pytest.fixture
def snapshot(tmp_path: Path) -> Path:
    p = tmp_path / "pricing.yaml"
    p.write_text(SNAPSHOT, encoding="utf-8")
    return p


# --------------------------------------------------------------- price_for
def test_price_for_unknown_model_is_none(snapshot: Path):
    assert price_for("does-not-exist", str(snapshot)) is None


@pytest.mark.parametrize("model", [None, ""])
def test_price_for_missing_model_name_is_none(snapshot: Path, model):
    assert price_for(model, str(snapshot)) is None


def test_price_for_returns_the_entry(snapshot: Path):
    entry = price_for("input-only", str(snapshot))
    assert entry == {"input_per_1k": 0.0007, "output_per_1k": 0}


# ---------------------------------------------------------------- cost_for
def test_unknown_model_is_not_known_and_has_no_cost(snapshot: Path):
    out = cost_for("does-not-exist", 1000, 1000, str(snapshot))
    assert out["known"] is False
    assert out["cost"] is None, "an unconfigured rate must never read as 0.0"


def test_null_rate_is_not_known(snapshot: Path):
    """A null rate means 'could not be sourced', not 'free'."""
    out = cost_for("null-output-rate", 1000, 1000, str(snapshot))
    assert out["known"] is False
    assert out["cost"] is None


@pytest.mark.parametrize(
    "input_tokens,output_tokens",
    [(None, 1000), (1000, None), (None, None)],
)
def test_none_token_count_is_unknown(snapshot: Path, input_tokens, output_tokens):
    """Regression: a half-known call used to price the missing leg as 0."""
    out = cost_for("input-only", input_tokens, output_tokens, str(snapshot))
    assert out["known"] is False
    assert out["cost"] is None


def test_measured_zero_tokens_is_known_and_free(snapshot: Path):
    """Regression: 0 is measured, so it must price — and price to zero."""
    out = cost_for("input-only", 0, 0, str(snapshot))
    assert out["known"] is True
    assert out["cost"] == 0.0


def test_input_only_model_charges_only_the_input_leg(snapshot: Path):
    out = cost_for("input-only", 1000, 5000, str(snapshot))
    assert out["known"] is True
    # 1000/1000 * 0.0007 + 5000/1000 * 0 = 0.0007
    assert out["cost"] == pytest.approx(0.0007)


def test_cost_is_the_sum_of_both_legs(snapshot: Path):
    out = cost_for("input-only", 2000, 0, str(snapshot))
    assert out["cost"] == pytest.approx(0.0014)


def test_effective_date_is_carried_through(snapshot: Path):
    out = cost_for("input-only", 1000, 0, str(snapshot))
    # YAML turns an unquoted ISO date into datetime.date, so compare on str()
    assert str(out["effective_date"]) == "2026-09-16"


def test_currency_defaults_to_cny_when_absent(tmp_path: Path):
    p = tmp_path / "no-currency.yaml"
    p.write_text("models:\n  m:\n    input_per_1k: 1\n    output_per_1k: 1\n", encoding="utf-8")
    assert cost_for("m", 1000, 0, str(p))["currency"] == "CNY"


# ------------------------------------------------------------ path handling
def test_missing_snapshot_file_yields_unknown_not_a_crash(tmp_path: Path):
    out = cost_for("input-only", 1000, 1000, str(tmp_path / "nope.yaml"))
    assert out["known"] is False
    assert out["cost"] is None


def test_env_override_is_resolved_before_the_cache(tmp_path: Path, monkeypatch):
    """Regression: caching on the raw argument made PRICING_PATH a no-op."""
    p = tmp_path / "override.yaml"
    p.write_text(SNAPSHOT, encoding="utf-8")
    monkeypatch.setenv("PRICING_PATH", str(p))
    assert str(load_pricing("config/pricing.yaml")["effective_date"]) == "2026-09-16"
    assert price_for("input-only") is not None


def test_shipped_snapshot_is_loadable():
    """The real config/pricing.yaml must parse and carry an effective_date."""
    pricing = load_pricing("config/pricing.yaml")
    assert pricing.get("effective_date"), "a price table without a date is not auditable"
    assert pricing.get("models"), "expected the four billed models to be present"
