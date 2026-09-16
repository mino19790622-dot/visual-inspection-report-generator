"""Three-arm cost close-out: turn the ablation's token totals into money.

Rates are read from ``config/pricing.yaml`` through
``app.observability.pricing.cost_for``, so the price is written down in exactly
one place and this table cannot silently drift from it. A rate that is still
null stays unknown here too -- never 0.0.

Why a bracket and not a number
------------------------------
``eval/ablation_topk.py`` records ``vlm_tokens`` and ``grounding_tokens`` as
``prompt_tokens + completion_tokens`` summed into one figure. The split is not
in the artifact, and input and output are priced differently (qwen-vl-max is
0.0016 vs 0.004 per 1k), so a point cost is not derivable from what was
recorded. Rather than invent a split, this reports the two ends of the range:
every token priced as input (floor) and every token priced as output (ceiling).
The truth is inside; the width of the bracket is the cost of not having
recorded the split. Narrowing it needs a change in the recorder, not an
assumption here.

Embedding cost is absent for the same class of reason: the artifact counts
retrieval *calls*, not embedding tokens, so the text-embedding-v2 side of a
retrieval call is not derivable. It is reported as a declared gap rather than
guessed.

Usage:
    python -m eval.cost_table --artifact path/to/ablation/ci-run.json
    python -m eval.cost_table --artifact ... --out reports/cost/table.json
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

from app.observability.pricing import cost_for

# Which model pays for which column. These are the production defaults
# (app/grounding.py: GROUNDING_MODEL; the VLM span is the narrative call).
VLM_MODEL = "qwen-vl-max"
GROUNDING_MODEL = "qwen-plus"

ARMS = ("off", "fixed", "agentic")


def load_artifact(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def image_count(artifact: dict[str, Any]) -> int:
    """Distinct images, not rows: the artifact has one row per (image, arm)."""
    return len({r["id"] for r in artifact.get("per_image", [])})


def bracket(model: str, total_tokens: int) -> dict[str, Any]:
    """Cost a combined token total as a [floor, ceiling] range.

    Neither end is a prediction of the split; they are the two prices the same
    total attracts depending on the unknown in/out ratio.
    """
    floor = cost_for(model, total_tokens, 0)
    ceiling = cost_for(model, 0, total_tokens)
    if not (floor["known"] and ceiling["known"]):
        return {"model": model, "tokens": total_tokens,
                "floor": None, "ceiling": None, "known": False,
                "currency": floor["currency"]}
    return {
        "model": model,
        "tokens": total_tokens,
        "floor": round(floor["cost"], 6),
        "ceiling": round(ceiling["cost"], 6),
        "known": True,
        "currency": floor["currency"],
    }


def _sum_tokens(artifact: dict[str, Any], arm: str, field: str) -> int:
    return sum(r.get(field, 0) for r in artifact.get("per_image", [])
               if r.get("arm") == arm)


def build_table(artifact: dict[str, Any]) -> dict[str, Any]:
    n_images = image_count(artifact)
    arms: dict[str, Any] = {}

    for arm in ARMS:
        vlm_tokens = _sum_tokens(artifact, arm, "vlm_tokens")
        gnd_tokens = _sum_tokens(artifact, arm, "grounding_tokens")
        calls = _sum_tokens(artifact, arm, "retrieval_calls")
        findings = _sum_tokens(artifact, arm, "findings")

        vlm = bracket(VLM_MODEL, vlm_tokens)
        gnd = bracket(GROUNDING_MODEL, gnd_tokens)
        known = vlm["known"] and gnd["known"]
        floor = round(vlm["floor"] + gnd["floor"], 6) if known else None
        ceiling = round(vlm["ceiling"] + gnd["ceiling"], 6) if known else None

        arms[arm] = {
            "images": n_images,
            "findings": findings,
            "retrieval_calls_total": calls,
            "retrieval_calls_per_image": round(calls / n_images, 3) if n_images else None,
            "vlm_tokens_total": vlm_tokens,
            "grounding_tokens_total": gnd_tokens,
            "vlm_cost": vlm,
            "grounding_cost": gnd,
            "cost_per_image_floor": round(floor / n_images, 6) if floor is not None else None,
            "cost_per_image_ceiling": round(ceiling / n_images, 6) if ceiling is not None else None,
            "known": known,
        }

    return {
        "source_artifact": artifact.get("git_sha"),
        "artifact_timestamp": artifact.get("timestamp"),
        "temperature": artifact.get("temperature"),
        "images": n_images,
        "currency": arms["fixed"]["vlm_cost"]["currency"],
        "model_for_vlm_tokens": VLM_MODEL,
        "model_for_grounding_tokens": GROUNDING_MODEL,
        "arms": arms,
        "marginal_vs_fixed": marginal_vs_fixed(arms),
        "declared_gaps": [
            "input/output token split is not recorded by ablation_topk.py, so "
            "every cost is a [floor, ceiling] bracket over the unknown split",
            "text-embedding-v2 call cost is not derivable: the artifact counts "
            "retrieval calls, not embedding tokens",
            "judge (qwen-turbo) cost is not derivable from this artifact: "
            "judge_score is optional and mean_judge is null in this run",
        ],
    }


def marginal_vs_fixed(arms: dict[str, Any]) -> dict[str, Any]:
    """What the agentic arm cost on top of the fixed arm, and what it bought.

    A single net finding delta would be misleading: the per-image counts move
    in both directions, so the churn is larger than the net.
    """
    fx, ag = arms["fixed"], arms["agentic"]
    net_findings = ag["findings"] - fx["findings"]

    detail = {
        "net_findings_delta": net_findings,
        "extra_retrieval_calls": ag["retrieval_calls_total"] - fx["retrieval_calls_total"],
        "extra_grounding_tokens": ag["grounding_tokens_total"] - fx["grounding_tokens_total"],
    }
    if fx["known"] and ag["known"]:
        detail["extra_cost_floor"] = round(
            (ag["cost_per_image_floor"] - fx["cost_per_image_floor"]) * ag["images"], 6)
        detail["extra_cost_ceiling"] = round(
            (ag["cost_per_image_ceiling"] - fx["cost_per_image_ceiling"]) * ag["images"], 6)
        if net_findings > 0:
            detail["cost_per_net_extra_finding_floor"] = round(
                detail["extra_cost_floor"] / net_findings, 6)
            detail["cost_per_net_extra_finding_ceiling"] = round(
                detail["extra_cost_ceiling"] / net_findings, 6)
    else:
        detail["extra_cost_floor"] = None
        detail["extra_cost_ceiling"] = None
    detail["caveat"] = (
        "net_findings_delta is a NET figure. Per-image counts differ in both "
        "directions, so it must not be read as 'the agentic arm produced this "
        "many more useful findings'.")
    return detail


def per_image_deltas(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    """Where the two grounded arms disagree, image by image."""
    rows: dict[str, dict[str, Any]] = {}
    for r in artifact.get("per_image", []):
        rows.setdefault(r["id"], {})[r.get("arm")] = r
    out = []
    for img, arms in rows.items():
        fx, ag = arms.get("fixed"), arms.get("agentic")
        if not fx or not ag:
            continue
        out.append({
            "id": img,
            "findings_fixed": fx.get("findings"),
            "findings_agentic": ag.get("findings"),
            "findings_delta": (ag.get("findings") or 0) - (fx.get("findings") or 0),
            "calls_fixed": fx.get("retrieval_calls"),
            "calls_agentic": ag.get("retrieval_calls"),
            "grounding_tokens_fixed": fx.get("grounding_tokens"),
            "grounding_tokens_agentic": ag.get("grounding_tokens"),
        })
    return out


def print_markdown(table: dict[str, Any], deltas: list[dict[str, Any]]) -> None:
    cur = table["currency"]
    n = table["images"]
    print(f"\n== Three-arm cost (n={n} images, {cur}, rates from "
          f"config/pricing.yaml) ==")
    print(f"   vlm_tokens      -> {table['model_for_vlm_tokens']}")
    print(f"   grounding_tokens-> {table['model_for_grounding_tokens']}")
    print()
    print("| arm | findings | retrieval calls/img | VLM tok | grounding tok "
          "| cost/img (floor - ceiling) |")
    print("|---|---|---|---|---|---|")
    for arm in ARMS:
        a = table["arms"][arm]
        print(f"| {arm} | {a['findings']} | {a['retrieval_calls_per_image']} "
              f"| {a['vlm_tokens_total']} | {a['grounding_tokens_total']} "
              f"| {a['cost_per_image_floor']} - {a['cost_per_image_ceiling']} |")

    m = table["marginal_vs_fixed"]
    print()
    print(f"agentic vs fixed: net findings {m['net_findings_delta']:+d}, "
          f"retrieval calls {m['extra_retrieval_calls']:+d}, "
          f"grounding tokens {m['extra_grounding_tokens']:+d}")
    print(f"  extra cost for all {n} images: "
          f"{m['extra_cost_floor']} - {m['extra_cost_ceiling']} {cur}")
    if m.get("cost_per_net_extra_finding_floor") is not None:
        print(f"  per NET extra finding: "
              f"{m['cost_per_net_extra_finding_floor']} - "
              f"{m['cost_per_net_extra_finding_ceiling']} {cur}")
    print(f"  note: {m['caveat']}")

    print("\nper-image finding delta (agentic - fixed):")
    for d in deltas:
        if d["findings_delta"]:
            print(f"   {d['id']:32s} {d['findings_fixed']} -> "
                  f"{d['findings_agentic']}  ({d['findings_delta']:+d})"
                  f"  calls {d['calls_fixed']} -> {d['calls_agentic']}"
                  f"  gnd_tok {d['grounding_tokens_fixed']} -> "
                  f"{d['grounding_tokens_agentic']}")

    print("\ndeclared gaps (not estimated):")
    for g in table["declared_gaps"]:
        print(f"   - {g}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", required=True,
                    help="ablation ci-run.json (mode=ablation)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    artifact = load_artifact(args.artifact)
    if not artifact.get("per_image"):
        print("error: artifact has no per_image rows")
        return 2

    table = build_table(artifact)
    deltas = per_image_deltas(artifact)
    print_markdown(table, deltas)

    out = args.out or os.path.join("reports", "cost", "three-arm.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"table": table, "per_image_deltas": deltas}, fh,
                  indent=2, ensure_ascii=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
