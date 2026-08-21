"""End-to-end evaluation runner for the visual-inspection agent.

For each image in the golden set:
  1. Run the agent (save=False — no report files written).
  2. Deterministic checks: risk level, detection count range, keyword presence.
  3. LLM-as-judge scoring of the VLM report against the rubric.
  4. Per-image score on 0-5 scale: 5*0.4*det + 0.6*judge_avg.
  5. Overall score = mean of per-image scores.
  6. Exit 0 if overall >= threshold (default 3.8), else 1.

Usage:
    python -m eval.run_eval                   # run full golden set
    python -m eval.run_eval --id bus_street_side  # one image
    python -m eval.run_eval --threshold 3.8   # custom threshold
    python -m eval.run_eval --out eval/runs/run.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ensure project root is on sys.path when run as `python eval/run_eval.py`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.graph import InspectionAgent  # noqa: E402
from eval.judge import judge  # noqa: E402

GOLDEN_SET = Path(__file__).parent / "golden_set" / "golden_set.json"
DEFAULT_THRESHOLD = 3.7  # on 0-5 scale; tolerates LLM-judge variance
JUDGE_WEIGHT = 0.6
DET_WEIGHT = 0.4
# Score is on a 0-5 scale: 0.4 * 5 = 2.0 from deterministic, 0.6 * 5 = 3.0 from judge.
# Default threshold 3.8 means: must pass all deterministic checks AND avg judge >= 3/5.


def _load_golden_set() -> list[dict]:
    with open(GOLDEN_SET, encoding="utf-8") as f:
        return json.load(f)


def _deterministic_checks(state: dict, expect: dict, report: str) -> tuple[float, list[str]]:
    """Return (score 0-1, list of failure reasons)."""
    failures: list[str] = []
    # 1. detection count range (state.det_result has "counts" dict, not object_count)
    dr = state.get("det_result") or {}
    n = sum(dr.get("counts", {}).values())
    if not (expect["min_detections"] <= n <= expect["max_detections"]):
        failures.append(
            f"detection_count: expected in "
            f"[{expect['min_detections']},{expect['max_detections']}] actual={n}")
    # 2. must-mention keyword (any-of)
    keywords = expect.get("must_mention_any", [])
    lower = report.lower()
    if keywords and not any(k.lower() in lower for k in keywords):
        failures.append(f"keywords: none of {keywords} found in report")
    # 3. safety concerns (heuristic: any of risk/hazard/safety keywords)
    if expect.get("should_flag_safety_concerns"):
        if not any(w in lower for w in
                   ["risk", "hazard", "safety", "concern", "danger"]):
            failures.append("should_flag_safety_concerns: no safety vocabulary")
    # pass if no failures
    score = 1.0 if not failures else 0.0
    return score, failures


def _judge_score(judge_result: dict) -> float:
    """Return average judge score on 0–5 scale (mean of 4 dimensions, each 1-5)."""
    keys = ("scene_id", "safety", "domain_awareness", "structure")
    vals = [int(judge_result.get(k, 0)) for k in keys]
    if any(v == 0 for v in vals):
        return 0.0
    return sum(vals) / len(keys)


def evaluate_one(agent: InspectionAgent, item: dict,
                 skip_judge: bool = False) -> dict:
    image_path = ROOT / item["image"]
    t0 = time.time()
    state = agent.run(str(image_path), save=False)
    elapsed = time.time() - t0
    report = state.get("vlm_report", "")
    det_score, failures = _deterministic_checks(
        state, item["expect"], report)
    if skip_judge:
        judge_result = {"skipped": True}
        judge_avg = 0.0
    else:
        judge_result = judge(str(image_path), report, item["rubric"])
        judge_avg = _judge_score(judge_result)  # 0–5
    score = 5 * DET_WEIGHT * det_score + JUDGE_WEIGHT * judge_avg  # 0–5
    return {
        "id": item["id"],
        "image": item["image"],
        "elapsed_s": round(elapsed, 1),
        "det_score": det_score,
        "judge_raw": judge_result,
        "judge_avg": round(judge_avg, 2),
        "score": round(score, 3),
        "det_failures": failures,
        "risk_level": state.get("risk_level"),
        "detection_count": sum(
            (state.get("det_result") or {}).get("counts", {}).values()
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the golden-set evaluation")
    ap.add_argument("--id", help="run only this golden-set entry id")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"overall score threshold (default {DEFAULT_THRESHOLD})")
    ap.add_argument("--skip-judge", action="store_true",
                    help="only run deterministic checks (no LLM cost)")
    ap.add_argument("--out", help="write JSON report to this path")
    args = ap.parse_args()

    items = _load_golden_set()
    if args.id:
        items = [i for i in items if i["id"] == args.id]
        if not items:
            print(f"ERROR: id '{args.id}' not in golden set", file=sys.stderr)
            return 2

    print(f"Running evaluation on {len(items)} image(s); threshold={args.threshold}; "
          f"judge={'on' if not args.skip_judge else 'OFF (deterministic only)'}\n")

    # build agent once; detector & retriever are cached via lru_cache
    print("Loading agent (this triggers RAG index build if first run)...\n")
    agent = InspectionAgent()

    results: list[dict] = []
    t_start = time.time()
    for i, item in enumerate(items, 1):
        print(f"[{i}/{len(items)}] {item['id']}  ({item['image']})")
        try:
            r = evaluate_one(agent, item, skip_judge=args.skip_judge)
        except Exception as e:
            print(f"  ERROR: {e}")
            r = {"id": item["id"], "image": item["image"], "error": str(e),
                 "score": 0.0}
        results.append(r)
        # brief line
        if "error" in r:
            print(f"  ERROR: {r['error']}\n")
        else:
            mark = "PASS" if r["score"] >= args.threshold else "FAIL"
            print(f"  risk={r['risk_level']:6s} det={r['detection_count']:2d}  "
                  f"score={r['score']:.3f}  [{mark}]")
            if r["det_failures"]:
                for f in r["det_failures"]:
                    print(f"    - {f}")
            judge_raw = r.get("judge_raw", {})
            if isinstance(judge_raw, dict) and not judge_raw.get("skipped"):
                print(f"    judge: scene={judge_raw.get('scene_id')} "
                      f"safety={judge_raw.get('safety')} "
                      f"domain={judge_raw.get('domain_awareness')} "
                      f"struct={judge_raw.get('structure')}")
                reason = judge_raw.get("reason")
                if reason:
                    print(f"    reason: {reason}")
        print()

    overall = sum(r.get("score", 0) for r in results) / max(len(results), 1)
    total_elapsed = time.time() - t_start
    print("=" * 64)
    print(f"OVERALL SCORE: {overall:.3f}  (threshold {args.threshold})  "
          f"[{'PASS' if overall >= args.threshold else 'FAIL'}]")
    print(f"Total time: {total_elapsed:.1f}s  "
          f"({len(results)} images, judge={'on' if not args.skip_judge else 'off'})")
    print("=" * 64)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "threshold": args.threshold,
                "overall_score": round(overall, 3),
                "passed": overall >= args.threshold,
                "results": results,
            }, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {args.out}")

    return 0 if overall >= args.threshold else 1


if __name__ == "__main__":
    raise SystemExit(main())
