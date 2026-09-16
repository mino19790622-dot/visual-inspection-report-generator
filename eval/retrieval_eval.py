"""Retrieval-side evaluation: recall@k, MRR, and the retrieval/generation split.

Why this file exists
--------------------
Phase C measured *attribution* — whether a finding cited a chunk that really was
retrieved. That answers "did the model invent a reference?", not "was the clause
the finding needed retrievable at all?". The latter is undefined without
retrieval ground truth, which is why recall@k / MRR were deliberately withheld
until now.

Truth lives in ``eval/golden_set/golden_set.json`` under the ``retrieval_truth``
field, annotated under ``eval/golden_set/ANNOTATION.md``. This module consumes it
and never invents it.

Citation verification is **not** reimplemented here. ``app.citations.check_findings``
is imported and reused, so the evaluation criterion and the (future) runtime
criterion cannot drift apart.

Two layers, reported separately
------------------------------
``R`` (retrieval)  ranking quality only — needs the embedding model, no LLM.
``G`` (generation) optional; runs the grounding agent on the golden ``scene``
                   text and asks which findings cite a *relevant* chunk.

Splitting them is what makes a failure localisable, per item:

    retrieval_miss      no truth clause appeared in top-k
    retrieved_unused    a truth clause was in top-k, but no finding cited it
    used                a finding cited a truth clause

Query source — a stated limitation
----------------------------------
By default the query is the golden item's ``scene`` string (a human-written
description). Production queries are the VLM *narrative*. The two are not
identical: the numbers here therefore measure "can the retriever surface the
right clause for an accurate scene description", which is a cleaner and cheaper
question but not literally the deployed path. It is reported as such.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from app.citations import check_findings
from app.rag.retriever import StandardsRetriever

GOLDEN_SET = os.path.join("eval", "golden_set", "golden_set.json")
TRUTH_META = os.path.join("eval", "golden_set", "retrieval_truth_meta.json")
STANDARDS_DIR = os.path.join("data", "standards")

# Must cover the `fixed` arm (k=5) and the `agentic` arm's observed mode (k=2).
K_GRID: tuple[int, ...] = (1, 2, 3, 5, 10)
MAX_K: int = max(K_GRID)


# --------------------------------------------------------------------- loading
def load_golden_set(path: str = GOLDEN_SET) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_truth_meta(path: str = TRUTH_META) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def truth_of(item: dict) -> list[str]:
    return list((item.get("retrieval_truth") or {}).get("chunk_ids") or [])


# ------------------------------------------------------------ integrity checks
def verify_fingerprint(meta: dict) -> list[str]:
    """Re-hash the standards files and compare with the recorded fingerprint.

    The truth is only valid for the chunking it was annotated against. If a
    standards file changed, its digest changes and every chunk_id may have
    shifted — so this must fail loudly rather than silently mis-score.
    """
    import hashlib

    problems: list[str] = []
    recorded = (meta.get("chunking_fingerprint") or {}).get("standards_sha256", {})
    for name, want in recorded.items():
        path = os.path.join(STANDARDS_DIR, name)
        if not os.path.exists(path):
            problems.append(f"{name}: missing")
            continue
        with open(path, "rb") as fh:
            got = hashlib.sha256(fh.read()).hexdigest()
        if got != want:
            problems.append(f"{name}: sha256 {got[:12]} != recorded {want[:12]}")
    return problems


def verify_truth_ids(retriever: StandardsRetriever, items: list[dict],
                     meta: dict) -> None:
    """Assert every truth chunk_id still resolves to the anchored chunk.

    Layer 3 of the protocol: a renumbering must error out, not score 0.
    """
    anchors = meta.get("anchors") or {}
    all_ids = sorted({c for it in items for c in truth_of(it)})
    if not all_ids:
        raise SystemExit("no retrieval truth found in the golden set")
    got = retriever.collection.get(ids=all_ids)
    present = set(got.get("ids") or [])
    missing = [c for c in all_ids if c not in present]
    if missing:
        raise SystemExit(
            "truth chunk_id(s) no longer exist in the index — the standards were "
            f"re-chunked; re-annotate before trusting any score: {missing}")

    by_id = dict(zip(got["ids"], got["documents"], strict=False))
    for cid in all_ids:
        anchor = anchors.get(cid) or {}
        prefix = (anchor.get("prefix40") or "").strip()
        if not prefix:
            continue
        body = " ".join(by_id[cid].split())
        if prefix.split(" ")[0] not in body:
            raise SystemExit(
                f"anchor mismatch for {cid}: expected {prefix!r} in the chunk "
                "body. The chunking or the standards changed; re-annotate.")


# ------------------------------------------------------------------- R: metrics
def recall_at_k(ranked: list[str], truth: list[str], k: int) -> float:
    """Fraction of the unique relevant chunks that appear in the top ``k``.

    The denominator is the **deduplicated** truth size. Using the raw list
    length would let a repeated id inflate the denominator and depress recall.

    Callers must exclude items whose truth set is empty: scoring those would
    manufacture a retrieval failure that does not exist (ANNOTATION.md §3.4),
    so an empty set raises rather than silently returning 0.0.
    """
    truth_set = set(truth)
    if not truth_set:
        raise ValueError(
            "recall_at_k needs a non-empty truth set; empty-truth items must be "
            "excluded from the denominator instead of being scored")
    return len(set(ranked[:k]) & truth_set) / len(truth_set)


def rank_of_first_relevant(ranked: list[str], truth: list[str]) -> int | None:
    truth_set = set(truth)
    for i, cid in enumerate(ranked, start=1):
        if cid in truth_set:
            return i
    return None


def evaluate_retrieval(retriever: StandardsRetriever, items: list[dict],
                       k_grid: tuple[int, ...] = K_GRID) -> dict[str, Any]:
    """Layer R — pure ranking quality over items that have a truth set."""
    max_k = max(k_grid)
    per_item: list[dict] = []
    for it in items:
        truth = truth_of(it)
        query = it["scene"]
        ranked = [c["chunk_id"] for c in retriever.retrieve(query, k=max_k)]
        row: dict[str, Any] = {
            "id": it["id"],
            "n_truth": len(truth),
            "top_k_ranked": ranked,
            "first_relevant_rank": rank_of_first_relevant(ranked, truth),
            "retrieved_any_truth": bool(set(ranked) & set(truth)),
        }
        if truth:
            for k in k_grid:
                row[f"recall@{k}"] = round(recall_at_k(ranked, truth, k), 4)
        per_item.append(row)

    scored = [r for r in per_item if r["n_truth"] > 0]
    excluded = [r["id"] for r in per_item if r["n_truth"] == 0]
    agg: dict[str, Any] = {
        "items_total": len(per_item),
        "items_scored": len(scored),
        "items_excluded_empty_truth": len(excluded),
        "excluded_ids": excluded,
        "corpus_chunks": retriever.collection.count(),
    }
    for k in k_grid:
        vals = [r[f"recall@{k}"] for r in scored]
        agg[f"recall@{k}"] = round(sum(vals) / len(vals), 4) if vals else None
    rrs = [(1.0 / r["first_relevant_rank"]) if r["first_relevant_rank"] else 0.0
           for r in scored]
    agg["mrr"] = round(sum(rrs) / len(rrs), 4) if rrs else None
    agg["items_with_no_truth_in_topk"] = sum(1 for r in scored
                                             if not r["retrieved_any_truth"])
    return {"aggregate": agg, "per_item": per_item}


# ----------------------------------------------------- G: generation (optional)
def classify_verdict(truth: set[str], cited_truth: list[str],
                     retrieved_truth: list[str], retrieval_calls: int) -> str:
    """Name the layer-G outcome for one image.

    The point of splitting R from G is to tell *why* a truth chunk never made
    it into a finding. ``retrieved_unused`` (the retriever surfaced it, the
    agent ignored it) and ``retrieval_miss`` (the retriever never surfaced it)
    are retriever-vs-agent diagnoses. ``not_attempted`` is a third case that
    must not be folded into either: the agent returned without calling the
    retriever at all, so the layer-R ranking was never consulted and blaming
    the retriever would be wrong.
    """
    if not truth:
        return "not_scored_empty_truth"
    if cited_truth:
        return "used"
    if retrieved_truth:
        return "retrieved_unused"
    if retrieval_calls == 0:
        return "not_attempted"
    return "retrieval_miss"


def evaluate_grounding(retriever: StandardsRetriever, items: list[dict],
                       mode: str) -> dict[str, Any]:
    """Layer G — run the grounding agent on the scene text, then check citations.

    Deliberately uses only ``app.citations.check_findings`` for the citation
    verdict; nothing here re-derives what "grounded" means.
    """
    from app.grounding import GroundingAgent

    agent = GroundingAgent(retriever)
    per_item: list[dict] = []
    for it in items:
        truth = set(truth_of(it))
        report = agent.run(narrative=it["scene"], detection_summary="", mode=mode)
        findings = [f.model_dump() for f in report.findings]
        check = check_findings(findings, report.retrieved)
        cited = [f.get("citation", {}).get("chunk_id") for f in findings
                 if f.get("citation")]
        cited_truth = sorted({c for c in cited if c in truth})
        retrieved_ids = {c.get("chunk_id") for c in report.retrieved}
        retrieved_truth = sorted(retrieved_ids & truth)
        calls = sum(1 for c in report.tool_calls
                    if c.get("tool") == "retrieve_standards")

        verdict = classify_verdict(truth, cited_truth, retrieved_truth, calls)

        per_item.append({
            "id": it["id"],
            "mode": mode,
            "findings": len(findings),
            "attribution_pass_rate": check["attribution_pass_rate"],
            "citation_counts": check["counts"],
            "retrieved_truth_chunks": retrieved_truth,
            "cited_truth_chunks": cited_truth,
            "verdict": verdict,
            "parse_failed": report.parse_failed,
            "retrieval_calls": calls,
            "grounding_tokens": (report.usage or {}).get("total_tokens", 0),
        })

    scored = [r for r in per_item if r["verdict"] != "not_scored_empty_truth"]
    dist: dict[str, int] = {}
    for r in scored:
        dist[r["verdict"]] = dist.get(r["verdict"], 0) + 1
    rates = [r["attribution_pass_rate"] for r in scored
             if r["attribution_pass_rate"] is not None]
    return {
        "aggregate": {
            "mode": mode,
            "items_scored": len(scored),
            "verdict_distribution": dist,
            "items_with_zero_retrieval_calls": sum(
                1 for r in scored if r["retrieval_calls"] == 0),
            "mean_attribution_pass_rate": (round(sum(rates) / len(rates), 4)
                                           if rates else None),
            "total_findings": sum(r["findings"] for r in per_item),
            "citations_ok": sum(r["citation_counts"]["ok"] for r in per_item),
            "citations_unresolved": sum(r["citation_counts"]["unresolved"]
                                        for r in per_item),
            "citations_hallucinated": sum(r["citation_counts"]["hallucinated"]
                                          for r in per_item),
        },
        "inputs": {
            "narrative": "golden_set[i]['scene'] (the human-written scene text)",
            "detection_summary": "",
            "caveat": ("This harness has no detector, so layer G runs with an "
                       "empty detection summary. That is not the production "
                       "configuration (image -> YOLO -> narrative -> grounding) "
                       "and it is the reason agents decline to retrieve: with "
                       "nothing detected there is little to ground. Read every "
                       "layer-G verdict as conditional on these inputs, and "
                       "check 'items_with_zero_retrieval_calls' before "
                       "attributing anything to the retriever."),
        },
        "per_item": per_item,
    }


# --------------------------------------------------------------------- reporting
def print_report(retrieval: dict, grounding: dict | None, meta: dict) -> None:
    agg = retrieval["aggregate"]
    print("\n== Retrieval (layer R) — query = golden 'scene' text ==")
    print(f"   items scored  : {agg['items_scored']} "
          f"(excluded, empty truth: {agg['items_excluded_empty_truth']}"
          f" -> {agg['excluded_ids']})")
    print(f"   corpus chunks : {agg['corpus_chunks']}")
    for k in K_GRID:
        print(f"   recall@{k:<3}    : {agg[f'recall@{k}']}")
    print(f"   MRR           : {agg['mrr']}")
    print(f"   no truth in top-{MAX_K}: {agg['items_with_no_truth_in_topk']}"
          f"/{agg['items_scored']}")
    print(f"   annotation disagreement (Jaccard mean): {meta.get('jaccard_mean')}")

    if grounding:
        g = grounding["aggregate"]
        print(f"\n== Generation (layer G) — grounding mode = {g['mode']} ==")
        print(f"   items scored  : {g['items_scored']}")
        print(f"   attribution   : {g['mean_attribution_pass_rate']}")
        print(f"   verdicts      : {g['verdict_distribution']}")
        print(f"   zero retrieval calls: {g['items_with_zero_retrieval_calls']}"
              f"/{g['items_scored']}  (detections are empty in this harness;"
              f" 'not_attempted' is not a retriever failure)")
        print(f"   citations     : ok={g['citations_ok']} "
              f"unresolved={g['citations_unresolved']} "
              f"hallucinated={g['citations_hallucinated']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grounding-mode", choices=["none", "fixed", "agentic"],
                    default="none", help="run layer G too (needs an LLM)")
    ap.add_argument("--out", default=os.path.join("reports", "retrieval",
                                                  "ci-run.json"))
    ap.add_argument("--k-grid", default=",".join(map(str, K_GRID)))
    args = ap.parse_args()

    k_grid = tuple(int(x) for x in args.k_grid.split(","))
    items = load_golden_set()
    meta = load_truth_meta()

    problems = verify_fingerprint(meta)
    if problems:
        raise SystemExit("standards changed since annotation — re-annotate: "
                         + "; ".join(problems))

    retriever = StandardsRetriever(standards_dir=STANDARDS_DIR)
    verify_truth_ids(retriever, items, meta)

    retrieval = evaluate_retrieval(retriever, items, k_grid)
    grounding = None
    if args.grounding_mode != "none":
        grounding = evaluate_grounding(retriever, items, args.grounding_mode)

    payload = {
        "query_source": "golden_scene_text",
        "query_source_caveat": (
            "Production queries are the VLM narrative; this run uses the "
            "human-written scene description, so the figures describe "
            "retrievability given an accurate description, not the deployed path."),
        "k_grid": list(k_grid),
        "grounding_mode": args.grounding_mode,
        "truth_source": "eval/golden_set/golden_set.json#retrieval_truth",
        "annotation_disagreement_jaccard_mean": meta.get("jaccard_mean"),
        "chunking_fingerprint": meta.get("chunking_fingerprint"),
        "retrieval": retrieval,
        "grounding": grounding,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print_report(retrieval, grounding, meta)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
