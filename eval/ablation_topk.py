# eval/ablation_topk.py
"""Three-arm tool-calling ablation for the grounding stage.

EXPERIMENT DESIGN — written before running (phase-C prompt 2.1)
---------------------------------------------------------------
* Population: 10 hand-curated golden-set images
  (`eval/golden_set/golden_set.json`). **Sample size n = 10** — small; all
  reported numbers are directional, never a production benchmark.
* Arms:
    - `off`     : v1.0 narrative path, grounding stage disabled. Regression
                  baseline — proves Phase 0 did not change the VLM narrative.
    - `fixed`   : grounding with k=MAX_K=5 clauses injected up front; the
                  retrieval tool is removed. Control arm for "did the model's
                  own k choice beat a fixed top-k?".
    - `agentic` : grounding with the `retrieve_standards` tool; the model
                  decides k per query (clamped to [1,5] by app/tools.py).
* Randomness / repetitions: grounding runs at temperature=0.0; the VLM runs at
  its configured temperature. **One pass per image** (single repetition) because
  each arm costs VLM calls + (fixed/agentic) grounding calls. If numbers move on
  rerun, that variance is a stated limitation, not hidden.
* Statistics (per arm, over the 10 images), all derived from the agent's own
  returned state — no re-scoring invented here:
    - attribution pass rate = ok / total_findings, plus the ok/unresolved/
      hallucinated triple (from app.citations.check_findings via grounded.citation_report)
    - retrieval calls: total, mean, per-image
    - k distribution: the k values the model actually requested (agentic);
      the fixed value (fixed); N/A (off)
    - token usage: VLM prompt/completion + grounding prompt/completion
    - deterministic-checks pass rate (reused from eval.run_eval)
* Off arm has no grounding -> attribution / retrieval / k are N/A. It only
  confirms the v1.0 narrative path is intact (same deterministic checks).

HONESTY RULES (phase-C prompt 2.2)
----------------------------------
* Sample size n=10 is printed on every output table.
* If agentic beats fixed ONLY because it issued more retrieval calls, the
  summary says "spent more", not "strategy better".
* If agentic is worse than fixed, that is reported too — negative results are
  this repo's signature.
* No recall@k / MRR: those need retrieval ground truth, which phase B produces.
  Computing them here would be meaningless, so they are absent by design.

Usage
-----
    python -m eval.ablation_topk                       # all three arms
    python -m eval.ablation_topk --arms agentic fixed  # subset
    python -m eval.ablation_topk --judge                # also LLM-judge the narrative
    python -m eval.ablation_topk --out reports/ablation/run.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.graph import InspectionAgent  # noqa: E402
from eval.run_eval import _deterministic_checks, _load_golden_set  # noqa: E402

GOLDEN_SET = Path(__file__).parent / "golden_set" / "golden_set.json"
ARMS = ["off", "fixed", "agentic"]


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip() or None
    except Exception:
        return None


def _collect_arm(agent: InspectionAgent, item: dict, mode: str,
                 judge: bool) -> dict:
    """Run one image under one arm; return a flat per-image record."""
    image_path = ROOT / item["image"]
    t0 = time.time()
    state = agent.run(str(image_path), save=False, grounding_mode=mode)
    elapsed = time.time() - t0

    grounded = state.get("grounded")
    vlm_usage = state.get("vlm_usage") or {}
    det = state.get("det_result") or {}
    report = state.get("vlm_report", "")

    det_score, failures = _deterministic_checks(state, item["expect"], report)
    rec: dict[str, object] = {
        "id": item["id"],
        "arm": mode,
        "elapsed_s": round(elapsed, 2),
        "detection_count": sum(det.get("counts", {}).values()),
        "deterministic_pass": not failures,
        "deterministic_score": round(det_score, 3),
    }

    if grounded is None:
        rec.update({
            "findings": 0, "citation_counts": None,
            "attribution_pass_rate": None, "retrieval_calls": 0,
            "k_values": [], "k_max": 0, "parse_failed": False,
            "vlm_tokens": (vlm_usage.get("prompt_tokens") or 0)
                          + (vlm_usage.get("completion_tokens") or 0),
            "grounding_tokens": 0,
        })
        if judge:
            rec["judge_score"] = _judge(image_path, report, item["rubric"])
        return rec

    cit = grounded.citation_report or {}
    counts = cit.get("counts") or {}
    usage = grounded.usage or {}
    rec.update({
        "findings": len(grounded.findings),
        "citation_counts": counts,
        "attribution_pass_rate": cit.get("attribution_pass_rate"),
        "retrieval_calls": sum(1 for c in grounded.tool_calls
                               if c.get("tool") == "retrieve_standards"),
        "k_values": [c.get("args", {}).get("k")
                     for c in grounded.tool_calls
                     if c.get("tool") == "retrieve_standards"],
        "k_max": grounded.max_k_requested,
        "parse_failed": grounded.parse_failed,
        "vlm_tokens": (vlm_usage.get("prompt_tokens") or 0)
                      + (vlm_usage.get("completion_tokens") or 0),
        "grounding_tokens": (usage.get("prompt_tokens") or 0)
                           + (usage.get("completion_tokens") or 0),
    })
    if judge:
        rec["judge_score"] = _judge(image_path, report, item["rubric"])
    return rec


def _judge(image_path: Path, report: str, rubric: str) -> float | None:
    try:
        from eval.judge import judge  # imported lazily: openai not needed otherwise
    except Exception:
        return None
    try:
        res = judge(str(image_path), report, rubric)
        s = res.get("avg") if isinstance(res, dict) else None
        return round(float(s), 3) if s is not None else None
    except Exception:
        return None


def _aggregate(records: list[dict]) -> dict:
    """Per-arm aggregate. `records` are the per-image rows for one arm.

    Rows with an ``error`` key are excluded from the averages (but listed), so a
    single failing image cannot crash the whole ablation run or silently skew it.
    """
    ok_records = [r for r in records if "error" not in r]
    errors = [{"id": r.get("id"), "error": r.get("error")}
              for r in records if "error" in r]
    n = len(ok_records)
    total_findings = sum(r.get("findings", 0) for r in ok_records)
    cit = {"ok": 0, "unresolved": 0, "hallucinated": 0}
    for r in ok_records:
        cc = r.get("citation_counts") or {}
        for k in cit:
            cit[k] += cc.get(k, 0)
    ok = cit["ok"]
    apr = (ok / total_findings) if total_findings else None
    calls = [r.get("retrieval_calls", 0) for r in ok_records]
    kvals = [k for r in ok_records for k in (r.get("k_values") or [])]
    det_pass = sum(1 for r in ok_records if r.get("deterministic_pass"))
    judges = [r["judge_score"] for r in ok_records
              if r.get("judge_score") is not None]
    return {
        "images": n,
        "errors": errors,
        "total_findings": total_findings,
        "citation_counts": cit,
        "attribution_pass_rate": round(apr, 4) if apr is not None else None,
        "retrieval_calls_total": sum(calls),
        "retrieval_calls_mean": round(sum(calls) / n, 3) if n else None,
        "k_values": kvals,
        "k_max": max((r.get("k_max") or 0) for r in ok_records) if ok_records else 0,
        "deterministic_pass_rate": round(det_pass / n, 4) if n else None,
        "vlm_tokens_total": sum(r.get("vlm_tokens", 0) for r in ok_records),
        "grounding_tokens_total": sum(r.get("grounding_tokens", 0)
                                      for r in ok_records),
        "mean_judge": round(sum(judges) / len(judges), 3) if judges else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="3-arm tool-calling ablation")
    ap.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS,
                    help="which arms to run (default: all)")
    ap.add_argument("--judge", action="store_true",
                    help="also LLM-judge the VLM narrative (costs extra calls)")
    ap.add_argument("--out", help="write structured JSON here")
    args = ap.parse_args()

    items = _load_golden_set()
    if not items:
        print("no golden-set items found", file=sys.stderr)
        return 2

    print(f"Tool-calling ablation — {len(items)} images, arms={args.arms}"
          f"{', with judge' if args.judge else ''}\n")

    agent = InspectionAgent()
    per_image: list[dict] = []
    for mode in args.arms:
        print(f"--- arm: {mode} ---")
        for i, item in enumerate(items, 1):
            try:
                rec = _collect_arm(agent, item, mode, args.judge)
            except Exception as e:  # one bad image must not kill the run
                rec = {"id": item["id"], "arm": mode, "error": str(e)}
                print(f"  [{i}/{len(items)}] {item['id']}: ERROR {e}")
            else:
                print(f"  [{i}/{len(items)}] {item['id']}: "
                      f"findings={rec.get('findings')} "
                      f"calls={rec.get('retrieval_calls')} "
                      f"apr={rec.get('attribution_pass_rate')}")
            per_image.append(rec)

    by_arm = {mode: _aggregate([r for r in per_image if r.get("arm") == mode])
              for mode in args.arms}

    # Honesty flag: did agentic "win" only by spending more retrieval?
    honest_note = None
    if "agentic" in by_arm and "fixed" in by_arm:
        a, f = by_arm["agentic"], by_arm["fixed"]
        a_calls = a.get("retrieval_calls_mean") or 0
        f_calls = f.get("retrieval_calls_mean") or 0
        a_apr = a.get("attribution_pass_rate")
        f_apr = f.get("attribution_pass_rate")
        if (a_apr is not None and f_apr is not None and a_apr >= f_apr
                and a_calls > f_calls):
            honest_note = ("agentic >= fixed on attribution but used more "
                           "retrieval calls on average — report as 'spent "
                           "more', not 'strategy better'.")
        elif (a_apr is not None and f_apr is not None and a_apr < f_apr):
            honest_note = ("agentic < fixed on attribution — negative result, "
                           "reported as-is.")

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "sample_size": len(items),
        "arms": args.arms,
        "temperature": "grounding=0.0; vlm=configured; 1 repetition",
        "design": ("off=v1.0 narrative baseline; fixed=k=MAX_K injected; "
                   "agentic=model decides k (clamped 1-5). Metrics from "
                   "app.citations.check_findings. No recall@k/MRR (no "
                   "retrieval ground truth yet — phase B)."),
        "aggregates": by_arm,
        "per_image": per_image,
        "honesty_note": honest_note,
    }

    out = args.out or str(ROOT / "reports" / "ablation" /
                          f"ablation_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    # Console summary
    print("\n" + "=" * 64)
    print("ABLATION SUMMARY  (n=%d, provisional judge threshold)" % len(items))
    print("=" * 64)
    for mode in args.arms:
        a = by_arm[mode]
        print(f"\n[{mode}]")
        print(f"  findings={a['total_findings']}  citation={a['citation_counts']}")
        print(f"  attribution_pass_rate={a['attribution_pass_rate']}")
        print(f"  retrieval_calls mean={a['retrieval_calls_mean']} "
              f"total={a['retrieval_calls_total']}")
        print(f"  k_values={a['k_values']}  k_max={a['k_max']}")
        print(f"  deterministic_pass_rate={a['deterministic_pass_rate']}")
        print(f"  tokens vlm={a['vlm_tokens_total']} "
              f"grounding={a['grounding_tokens_total']}")
        if a.get("mean_judge") is not None:
            print(f"  mean_judge={a['mean_judge']}")
    if honest_note:
        print(f"\nHONESTY NOTE: {honest_note}")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
