# app/observability/pricing.py
"""Turn recorded token usage into money — from a versioned price snapshot.

Usage is measured; price is configuration. Keeping them apart means a price
change can be re-applied to every historical run instead of invalidating it.

``config/pricing.yaml`` ships with ``null`` rates on purpose: the file is the
single place to paste the current official list, and an unfilled rate yields
``known=False`` rather than a silently wrong number.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_PRICING_PATH = "config/pricing.yaml"


def _load_raw(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        import yaml  # optional dep: absent -> no costing, never a crash
    except ImportError:
        return {}
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=8)
def _load_cached(path: str) -> dict[str, Any]:
    return _load_raw(path)


def load_pricing(path: str = DEFAULT_PRICING_PATH) -> dict[str, Any]:
    """Load the price snapshot.

    The env override is resolved *before* the cache lookup — caching on the
    raw argument would make ``PRICING_PATH`` useless once anything had been
    loaded, which is exactly the kind of silent wrong-number bug this module
    exists to avoid.
    """
    return _load_cached(os.getenv("PRICING_PATH", path))


def price_for(model: str | None, path: str = DEFAULT_PRICING_PATH) -> dict | None:
    """Price entry for a model, or None if the model/rates are unknown."""
    if not model:
        return None
    pricing = load_pricing(path)
    models = pricing.get("models") or {}
    entry = models.get(model)
    return entry if isinstance(entry, dict) else None


def cost_for(model: str | None, input_tokens: int | None,
             output_tokens: int | None,
             path: str = DEFAULT_PRICING_PATH) -> dict[str, Any]:
    """Convert token counts to cost.

    Returns ``{"cost": float|None, "currency": str, "known": bool}``.
    ``known=False`` means the rate was missing — the caller must not render
    that as 0.0, because "no price configured" and "free" are different facts.
    """
    pricing = load_pricing(path)
    currency = pricing.get("currency", "CNY")
    entry = price_for(model, path)
    result: dict[str, Any] = {"cost": None, "currency": currency,
                              "known": False, "model": model}
    if entry is None or input_tokens is None and output_tokens is None:
        return result
    in_rate = entry.get("input_per_1k")
    out_rate = entry.get("output_per_1k")
    if in_rate is None or out_rate is None:
        return result
    cost = ((input_tokens or 0) / 1000.0 * float(in_rate)
            + (output_tokens or 0) / 1000.0 * float(out_rate))
    result["cost"] = round(cost, 6)
    result["known"] = True
    result["effective_date"] = pricing.get("effective_date")
    return result
