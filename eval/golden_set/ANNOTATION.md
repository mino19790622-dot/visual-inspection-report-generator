# Retrieval Ground-Truth Annotation Protocol

> **Written before any annotation was performed.** This file exists so that the
> retrieval ground truth has a stated, reviewable definition rather than being
> whatever the retriever happened to return. Committed separately from, and
> earlier than, the truth data (see "Provenance" at the end).
>
> Applies to: `eval/golden_set/golden_set.json` retrieval truth, consumed by
> `eval/retrieval_eval.py`.

---

## 1. What is being annotated, and why

The golden set already carries **generation** expectations (`risk_level`,
`must_mention_any`, `min/max_detections`) but **no retrieval truth**. Without
retrieval truth the only measurable quality signal was the citation
*attribution* pass rate — which tells you a finding cited something that was
really retrieved, not that the *right* clause was retrievable. Recall@k and MRR
are undefined until this file's definitions are applied and the truth exists.

---

## 2. Truth unit: the `chunk_id`

### 2.1 Chosen unit

The unit is the **chunk**, identified by its `chunk_id`, which
`app/rag/retriever.py` builds as:

```
{standards_filename}::{chunk_index}      e.g.  construction_site_safety.md::3
```

Rationale: `chunk_id` is the same identifier a `Finding.citation` must point at,
and the same one `app.citations.check_finding` verifies against. Annotating in
any other unit (clause number, section heading, character span) would require a
lossy mapping before it could be compared with what the system actually
retrieves. **Annotating in the retrieval unit keeps the truth and the metric in
the same coordinate system.**

### 2.2 The fragility problem, and the mitigation

`chunk_index` is a function of chunking parameters. Changing `CHUNK_SIZE`,
`CHUNK_OVERLAP`, the `>50`-char filter, or the `\n(?=##)` split rule
**renumbers chunks and silently invalidates every id**. A truth file that
quietly stops matching is worse than no truth file, because the metric would
still produce a number.

Mitigation — three layers:

1. **Pin the chunking configuration.** Truth is valid *only* for
   `CHUNK_SIZE=600`, `CHUNK_OVERLAP=100`, `min_len=50`, `split=r"\n(?=##)"`.
   These values are recorded in the truth file's `chunking_fingerprint` block
   together with the `sha256` of each standards file. If a standards file
   changes, its digest changes and the truth must be re-checked.
2. **Carry a self-healing anchor.** Each truth entry records, alongside the id,
   the chunk's **heading path** and the **first 40 characters** of its body.
   After a re-chunk, ids can be re-derived by re-matching those anchors — the
   annotation survives a renumbering even though the ids do not.
3. **Fail loudly.** `eval/retrieval_eval.py` re-derives chunks from the live
   retriever and asserts every truth `chunk_id` still resolves to a chunk whose
   anchor matches. A mismatch is a hard error, not a silent zero score.

---

## 3. Relevance definition: **binary**, at *applicable-clause* granularity

### 3.1 The rule

A chunk is **relevant** to a golden-set image iff a competent inspector, given
only the image's scene description, **could be expected to act on or cite that
clause for this scene** — i.e. the clause's subject matter directly governs an
observable condition or a required action in the scene.

It is **not** sufficient that a clause merely mentions a word appearing in the
scene, and it is **not** required that the clause be the single best citation.

### 3.2 Binary, not graded — and why

Graded relevance (e.g. 0/1/2 for irrelevant / related-but-insufficient /
directly-supporting) was rejected for three concrete reasons:

1. **It needs agreement to be meaningful.** A graded scale encodes a
   *distance between grades*. With one annotator there is no way to
   demonstrate that the gap between 1 and 2 means the same thing across items;
   the grader is unfalsifiable. A binary judgement is reproducible and directly
   auditable by a reader of this repo.
2. **It does not buy anything for the metrics used.** The metrics specified
   (recall@k, MRR) are rank-of-first-relevant metrics. They consume a relevance
   *set*; they do not use graded magnitudes unless nDCG is introduced, and nDCG
   is deliberately out of scope here.
3. **Partial credit would hide the failure mode we care about.** The question
   this evaluation must answer is binary in practice: *was the clause the
   finding needed retrievable at rank ≤ k, or not?*

### 3.3 Multi-standard images

An image may legitimately be relevant to **several standards** (a construction
scene touches both site safety and environmental inspection). Truth is therefore
a *set* spanning files, not a single file. The coarse `relevant_standard_files`
list is recorded as well, so a reader can see whether a miss is "wrong clause"
or "wrong standard entirely".

### 3.4 Images with an empty truth set

Some scenes are genuinely not governed by any document in the 6-standard
corpus (e.g. a static marina with no works activity). These are annotated with
`chunk_ids: []` **plus an explicit `no_applicable_clause: true` reason** — never
left blank, so an empty set is a stated finding rather than missing work.

**Empty-truth items are excluded from the recall@k / MRR denominators** and
reported separately by count. Scoring them as recall 0 would manufacture a
retrieval failure that does not exist; scoring them as 1 would inflate the
metric. Their count is always shown next to the denominator.

---

## 4. Data structure

One image → one truth record; one record → many chunk ids.

```jsonc
{
  "id": "construction_foundation_aerial",
  // ... existing generation fields: image / scene / expect / rubric ...

  "retrieval_truth": {
    "chunk_ids": ["construction_site_safety.md::3",
                  "construction_site_safety.md::4"],
    "relevant_standard_files": ["construction_site_safety.md"],
    "no_applicable_clause": false,
    "confidence": "high",          // high = unambiguous; medium = judgement call
    "note": "Exposed rebar + open foundation -> equipment exclusion zones and
             hazard thresholds both govern." 
  }
}
```

`confidence` records the annotator's own uncertainty about the item. It is
**never** used to filter results — it exists so a low-confidence item can be
re-checked (and is surfaced per-item in the output).

### File-level block

The truth file's top-level object carries the fingerprint so staleness is
detectable:

```jsonc
"chunking_fingerprint": {
  "chunk_size": 600, "chunk_overlap": 100, "min_chunk_len": 50,
  "split_rule": "\\n(?=##)",
  "standards_sha256": { "construction_site_safety.md": "...", "...": "..." }
}
```

---

## 5. Single annotator — bias controls

The annotator also designed the retriever and wrote the metrics. That is
"setting the exam and marking it", and it must be constrained rather than
waved away. Four measures:

1. **Pre-registration (this file).** The relevance rule, the truth unit, and the
   handling of empty sets are fixed *and committed* before any annotation. They
   cannot be reshaped after seeing results.
2. **Blind to retrieval output.** Annotation is produced from the scene
   description and the standards corpus **only**. The retriever is not run, and
   no retrieved set is inspected, until the truth file is committed. *(This is
   what makes the truth and not a transcript of the model's output.)*
3. **Two independent passes with a recorded disagreement rate.** The whole set
   is annotated twice, the second pass ≥ 24 h later, re-reading the standards
   but not the first pass. Inter-pass agreement is reported as
   **Jaccard index over the chunk-id sets, averaged per item**. Items that
   disagree are resolved by re-reading the clause, and the resolution is
   recorded in `note`. The measured disagreement rate is published alongside the
   results — it is an error bar on the truth itself.
4. **Per-item publication.** Metrics are reported per item as well as
   aggregate, so a reader can audit any single judgement rather than trusting a
   mean.

**What remains uncontrolled, stated plainly:** there is no second human
annotator and no adjudicator, so an *idiosyncratic* relevance philosophy — one
this file's rule does not capture — cannot be detected. Measure 3 detects
*instability*, not *shared error*.

---

## 6. Is n = 10 enough? — **No, and it is not presented as if it were**

n = 10 images against a **50-chunk** corpus. Assessment:

- A single query against 50 chunks yields a **coarse** recall curve: one rank
  position is 2 % of the corpus, so recall@1 and recall@2 differ by whole
  items, not decimals.
- With n = 10 (minus any empty-truth items) the 95 % interval on a recall
  estimate is roughly ±0.25–0.30 around 0.8. **Differences below ~0.25 between
  two configurations are not resolvable** by this set and must not be reported
  as improvements.
- The k-grid can only meaningfully span small k; recall@k for large k
  saturates simply because the corpus is small.

**Recommendation: keep n = 10 for this round; do not expand now.** Expanding
the golden set is a phase-B task with its own cost (each item needs annotation
*and* a generation rubric), and expanding mid-round would repeat the mistake
this protocol exists to prevent — changing the instrument after use.

**What this costs us, stated up front:** every retrieval number from this set is
**directional**. It can support claims of the form *"the needed clause was
retrievable at rank ≤ k in X of the items that have a truth set"*, and it can
surface a **clear** failure (a clause that is never retrieved at any reasonable
k). It **cannot** support fine-grained rank comparisons between the two
retrievers, and no such comparison will be claimed. Every table produced from
this truth must carry `n` and the excluded-item count beside it.

---

## 7. Split discipline (calibration / tuning / test)

The `openvino-yolo-benchmark` project used three disjoint splits
(calibration / tuning / test). Translated to this RAG setting, the three roles
would be:

| role | what it governs here |
|---|---|
| **calibration** | the citation match threshold — `app.citations.DEFAULT_MATCH_THRESHOLD` (0.55). It decides what counts as a valid quote match, so it must be set before the citation numbers are read. |
| **tuning** | the retrieval *policy*: top-k choice, the `fixed` arm's k, the k-clamp range, the retrieval-call budget. |
| **test** | recall@k / MRR and the attribution rate that get published. |

### Verdict: **three-way splitting is not viable at n = 10.** 

With 10 items, three splits give ~3 items each; a recall estimate over 3 items
is not an estimate. Pretending otherwise would be worse than not splitting.
Forcing the discipline where the sample cannot carry it is exactly the kind of
gesture this protocol is meant to prevent.

**Adopted alternative (as the prompt suggests):** the entire set is the **test
set**, and tuning on it is **prohibited by rule**:

- no retrieval parameter is adjusted after seeing these results and re-run to
  report the better pass (prompt §2.6 already forbids this; the rule is
  restated here as an annotation-level commitment);
- if a parameter *is* later changed — for any reason — the change, its
  motivation, and **both** runs are reported, never only the later one;
- the calibration threshold (0.55) is likewise frozen for this round. Its own
  calibration is deferred to phase B, when more items exist to calibrate on.

---

## 8. Provenance

- Protocol written and committed **before** the truth annotation, in its own
  commit, so `git log` shows the ordering (verified by §4 command 1 of the
  execution prompt).
- Annotation produced blind to retriever output; two passes ≥ 24 h apart, with
  the disagreement rate published.
- This document is the authority for any dispute about a truth label. A label
  that cannot be justified by §3.1 is a bug in the truth file.
