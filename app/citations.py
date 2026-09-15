# app/citations.py
"""Citation grounding checks.

This module is deliberately shared:

* **Offline** — ``eval/`` uses it to measure citation correctness (Phase 1-B).
* **Online**  — ``guardrails/`` will use it to reject or downgrade ungrounded
  findings at runtime (Phase 2-D).

Both paths MUST call these same functions. If the runtime check and the
evaluation criterion ever drift apart, every published number becomes
meaningless, so there is exactly one implementation.

Status model
------------
``ok``            the cited chunk was retrieved this request, and the quoted
                  text is findable inside it.
``unresolved``    the chunk was retrieved, but the quote does not appear in it
                  (bad extraction / paraphrased quote).
``hallucinated``  the citation is missing, or points at a chunk that was never
                  retrieved this request — the model invented the reference.
"""
from __future__ import annotations

import difflib
import re
from enum import Enum
from typing import Any, Iterable

# Calibrate on the calibration split before trusting it as a gate (see
# docs/V2_GAP_ANALYSIS.md). It is a starting value, NOT a measured result.
DEFAULT_MATCH_THRESHOLD = 0.55


class CitationStatus(str, Enum):
    OK = "ok"
    UNRESOLVED = "unresolved"
    HALLUCINATED = "hallucinated"


def _normalise(text: str) -> str:
    """Lower-case and collapse whitespace; strip markdown quote/heading noise."""
    text = (text or "").replace("\n", " ").replace(">", " ")
    return re.sub(r"\s+", " ", text).strip().lower()


def fuzzy_contains(haystack: str, needle: str,
                   threshold: float = DEFAULT_MATCH_THRESHOLD) -> float:
    """Best similarity (0.0-1.0) between ``needle`` and any window of ``haystack``.

    Sliding-window because a citation quote is a *substring* of a chunk: a
    whole-string comparison would score a short correct quote as a poor match
    against a long chunk.
    """
    h, n = _normalise(haystack), _normalise(needle)
    if not n:
        return 0.0
    if n in h:
        return 1.0
    hw, nw = h.split(), n.split()
    if not hw or not nw:
        return 0.0
    width = len(nw)
    best = 0.0
    matcher = difflib.SequenceMatcher(None, "", n)
    for i in range(0, max(1, len(hw) - width + 1)):
        window = " ".join(hw[i:i + width])
        matcher.set_seq1(window)
        ratio = matcher.ratio()
        if ratio > best:
            best = ratio
            if best >= 0.999:
                break
    return best


def check_citation(citation: dict | None,
                   retrieved_chunks: Iterable[dict],
                   threshold: float = DEFAULT_MATCH_THRESHOLD
                   ) -> tuple[CitationStatus, str]:
    """Verify one citation against the chunks actually retrieved for a request.

    Fully deterministic — no LLM involved. This is the check that can be a hard
    CI gate, unlike the semantic (judge) layer.
    """
    chunks = {c.get("chunk_id"): c for c in (retrieved_chunks or [])}
    if not citation:
        return CitationStatus.HALLUCINATED, "finding has no citation"

    chunk_id = citation.get("chunk_id")
    if not chunk_id:
        return CitationStatus.HALLUCINATED, "citation has no chunk_id"
    if chunk_id not in chunks:
        return (CitationStatus.HALLUCINATED,
                f"chunk_id {chunk_id!r} was not retrieved this request")

    score = fuzzy_contains(chunks[chunk_id].get("text", ""),
                           citation.get("quote", ""), threshold)
    if score < threshold:
        return (CitationStatus.UNRESOLVED,
                f"quote match {score:.2f} < threshold {threshold}")
    return CitationStatus.OK, f"quote match {score:.2f}"


def check_findings(findings: Iterable[dict],
                   retrieved_chunks: Iterable[dict],
                   threshold: float = DEFAULT_MATCH_THRESHOLD) -> dict[str, Any]:
    """Verify every finding's citation. Returns counts + per-finding detail."""
    chunks = list(retrieved_chunks or [])
    detail: list[dict[str, Any]] = []
    counts = {s.value: 0 for s in CitationStatus}
    for f in findings or []:
        status, reason = check_citation(f.get("citation"), chunks, threshold)
        counts[status.value] += 1
        detail.append({"finding_id": f.get("id"), "status": status.value,
                       "reason": reason})
    total = sum(counts.values())
    return {
        "total_findings": total,
        "counts": counts,
        "attribution_pass_rate": (counts[CitationStatus.OK.value] / total
                                  if total else None),
        "detail": detail,
    }
