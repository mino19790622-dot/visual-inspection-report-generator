"""Cross-check the qwen judge against an independent rater.

The judge (qwen-turbo) scores reports produced by qwen-plus / qwen-vl-max. Same
model family. If the judge is lenient toward its own family, the 3.7 gate in
``eval/run_eval.py`` passes reports a neutral reader would fail, and every
"overall >= 3.7" claim in the README inherits that error.

Why an agreement rate and not a correlation coefficient
-------------------------------------------------------
The audited subset is small (a few images x 4 rubric dimensions), every score
is an integer on a 1-5 ordinal scale, and ties are everywhere. A correlation
coefficient on that many tied ordinal pairs is unstable, and it is easy to
misread as a quality verdict when all it measures is whether the two orderings
move together. The question the gate actually depends on is narrower: *when
the judge passes a report, does an independent rater agree closely enough that
the pass survives?* So the headline numbers here are an exact-agreement rate,
a within-1 rate, and the mean absolute difference. Spearman rho is computed and
printed as a secondary figure, not used for the decision.

The signed mean difference is reported separately and matters more than any
agreement rate for the bias question: agreement measures noise, the sign
measures direction. A lenient judge shows up as a positive signed difference.

Usage:
    python -m eval.judge_agreement --eval-artifact reports/eval/ci-run.json \
        --hand scores/hand_subset.json --threshold 3.7 \
        --rater "independent rater (non-Qwen)" --out reports/judge/agreement.json
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

DIMENSIONS = ("scene_id", "safety", "domain_awareness", "structure")


def load_judge_scores(path: str) -> tuple[dict[str, dict[str, int]], dict[str, Any]]:
    """Pull the judge's 4-dimension scores out of a run_eval artifact."""
    with open(path, encoding="utf-8") as fh:
        artifact = json.load(fh)

    scores: dict[str, dict[str, int]] = {}
    for r in artifact.get("results", []):
        raw = r.get("judge_raw")
        if not isinstance(raw, dict) or raw.get("skipped"):
            continue
        if not all(isinstance(raw.get(d), int) for d in DIMENSIONS):
            continue
        scores[r["id"]] = {d: raw[d] for d in DIMENSIONS}
    return scores, artifact


def load_hand_scores(path: str) -> tuple[dict[str, dict[str, int]], dict[str, Any]]:
    """Load the independent rater's scores.

    Expected shape::

        {"rater": "...", "blind": true, "scored_at": "...",
         "scores": {"<image id>": {"scene_id": 4, "safety": 3, ...}}}
    """
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    scores = doc["scores"] if "scores" in doc else doc
    return scores, {k: v for k, v in doc.items() if k != "scores"}


def _rank(values: list[float]) -> list[float]:
    """Average ranks, so ties share the mean of the ranks they span."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 2:
        return None
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    dx = sum((a - mx) ** 2 for a in x) ** 0.5
    dy = sum((b - my) ** 2 for b in y) ** 0.5
    if dx == 0 or dy == 0:
        return None  # one side is constant: rho undefined, not zero
    return num / (dx * dy)


def spearman(a: list[float], b: list[float]) -> float | None:
    return _pearson(_rank(a), _rank(b))


def agreement(judge: dict[str, dict[str, int]],
              hand: dict[str, dict[str, int]]) -> dict[str, Any]:
    """Pairwise agreement over dimensions, plus per-dimension breakdown."""
    per_dim: dict[str, Any] = {}
    pairs: list[tuple[int, int]] = []

    for dim in DIMENSIONS:
        j_vals, h_vals = [], []
        exact = within1 = 0
        for img, jd in judge.items():
            if img not in hand or dim not in hand[img]:
                continue
            jv, hv = jd[dim], hand[img][dim]
            j_vals.append(jv)
            h_vals.append(hv)
            pairs.append((jv, hv))
            if jv == hv:
                exact += 1
            if abs(jv - hv) <= 1:
                within1 += 1
        n = len(j_vals)
        per_dim[dim] = {
            "n": n,
            "exact": exact,
            "within_1": within1,
            "exact_rate": round(exact / n, 3) if n else None,
            "within_1_rate": round(within1 / n, 3) if n else None,
            "mean_abs_diff": (round(sum(abs(a - b) for a, b in zip(j_vals, h_vals, strict=True)) / n, 3)
                              if n else None),
            "mean_judge": round(sum(j_vals) / n, 3) if n else None,
            "mean_rater": round(sum(h_vals) / n, 3) if n else None,
            "signed_mean_diff": (round(sum(a - b for a, b in zip(j_vals, h_vals, strict=True)) / n, 3)
                                 if n else None),
            "spearman": (round(spearman(j_vals, h_vals), 3)
                         if spearman(j_vals, h_vals) is not None else None),
        }

    n = len(pairs)
    j_all = [p[0] for p in pairs]
    h_all = [p[1] for p in pairs]
    exact = sum(1 for a, b in pairs if a == b)
    within1 = sum(1 for a, b in pairs if abs(a - b) <= 1)
    rho = spearman(j_all, h_all)

    return {
        "n_pairs": n,
        "n_images_compared": len([i for i in judge if i in hand]),
        "exact_agreement": round(exact / n, 3) if n else None,
        "within_1_agreement": round(within1 / n, 3) if n else None,
        "mean_abs_difference": (round(sum(abs(a - b) for a, b in pairs) / n, 3)
                                if n else None),
        "signed_mean_difference_judge_minus_rater": (
            round(sum(a - b for a, b in pairs) / n, 3) if n else None),
        "spearman_rho_secondary": round(rho, 3) if rho is not None else None,
        "per_dimension": per_dim,
    }


def gate_impact(judge: dict[str, dict[str, int]],
                hand: dict[str, dict[str, int]],
                threshold: float) -> dict[str, Any]:
    """Would the two raters pass/fail the same images at the gate?

    run_eval scores an image as ``5*0.4*det + 0.6*judge_avg``. The deterministic
    part is identical for both raters, so only the judge term moves. With
    det_score == 1.0 (which the ablation shows for every image) the judge term
    must reach ``(threshold - 2.0) / 0.6`` for the image to pass; that is the
    judge average the gate actually demands.
    """
    needed = round((threshold - 2.0) / 0.6, 3)
    flips = []
    for img, jd in judge.items():
        if img not in hand:
            continue
        j_avg = sum(jd.values()) / len(DIMENSIONS)
        h_avg = sum(hand[img][d] for d in DIMENSIONS if d in hand[img]) / len(
            [d for d in DIMENSIONS if d in hand[img]])
        if (j_avg >= needed) != (h_avg >= needed):
            flips.append({
                "id": img,
                "judge_avg": round(j_avg, 2),
                "rater_avg": round(h_avg, 2),
                "judge_passes": j_avg >= needed,
                "rater_passes": h_avg >= needed,
            })
    return {
        "threshold": threshold,
        "judge_avg_required_to_pass": needed,
        "images_flipped": flips,
        "n_flipped": len(flips),
    }


def print_report(agg: dict[str, Any], gate: dict[str, Any],
                 meta: dict[str, Any]) -> None:
    print("\n== Judge credibility cross-check ==")
    print(f"   rater    : {meta.get('rater', 'unspecified')}")
    print(f"   blind    : {meta.get('blind')}")
    print(f"   images   : {agg['n_images_compared']}  "
          f"paired scores: {agg['n_pairs']}")
    print(f"   exact agreement   : {agg['exact_agreement']}")
    print(f"   within-1 agreement: {agg['within_1_agreement']}")
    print(f"   mean |diff|       : {agg['mean_abs_difference']}")
    print(f"   signed mean diff (judge - rater): "
          f"{agg['signed_mean_difference_judge_minus_rater']}")
    print(f"   spearman rho (secondary): {agg['spearman_rho_secondary']}")
    print("\n   per dimension:")
    for dim, d in agg["per_dimension"].items():
        print(f"     {dim:18s} n={d['n']} exact={d['exact_rate']} "
              f"within1={d['within_1_rate']} mad={d['mean_abs_diff']} "
              f"judge={d['mean_judge']} rater={d['mean_rater']} "
              f"signed={d['signed_mean_diff']}")
    print(f"\n   gate: a judge average of {gate['judge_avg_required_to_pass']} "
          f"is needed at threshold {gate['threshold']}")
    print(f"   images where the raters disagree on pass/fail: "
          f"{gate['n_flipped']}")
    for f in gate["images_flipped"]:
        print(f"     {f['id']:32s} judge={f['judge_avg']} "
              f"({'pass' if f['judge_passes'] else 'fail'})  "
              f"rater={f['rater_avg']} "
              f"({'pass' if f['rater_passes'] else 'fail'})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-artifact", required=True)
    ap.add_argument("--hand", required=True)
    ap.add_argument("--threshold", type=float, default=3.7)
    ap.add_argument("--rater", default="unspecified")
    ap.add_argument("--out", default=os.path.join("reports", "judge",
                                                  "agreement.json"))
    args = ap.parse_args()

    judge, _ = load_judge_scores(args.eval_artifact)
    hand, hand_meta = load_hand_scores(args.hand)
    if not judge:
        print("error: no usable judge_raw scores in the eval artifact")
        return 2
    if not hand:
        print("error: hand-score file is empty")
        return 2

    agg = agreement(judge, hand)
    gate = gate_impact(judge, hand, args.threshold)
    meta = {"rater": args.rater, **hand_meta}
    print_report(agg, gate, meta)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"rater_meta": meta, "agreement": agg, "gate_impact": gate},
                  fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
