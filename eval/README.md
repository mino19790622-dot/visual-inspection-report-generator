# eval/

Golden-set evaluation and LLM-as-judge for the visual-inspection agent.

## What it is

A second quality gate, on top of unit tests:

- **Unit tests** (`tests/`) verify *code logic* (mocked LLM/CV calls).
- **Golden-set eval** (`eval/`) verifies *AI output quality* (real VLM + real judge).

Without this, a refactor that breaks the VLM prompt would still pass CI because
the LLM call is mocked out. The eval catches prompt/pipeline regressions by
running the agent on a fixed set of 10 images and scoring the report against a
human-written rubric.

## Files

- `golden_set/golden_set.json` — 10 images + per-image expectations
  (detection count range, must-mention keywords, safety vocabulary) + rubric
  used by the judge + **retrieval ground truth** (`retrieval_truth.chunk_ids`).
- `golden_set/ANNOTATION.md` — the annotation protocol for that retrieval truth.
  Committed *before* the truth itself, so `git log` proves the rules were fixed
  before anything was labelled. Read it before trusting a recall number.
- `golden_set/retrieval_truth_meta.json` — chunking fingerprint, per-chunk
  anchors, and the two annotation passes with their disagreement rate.
- `judge.py` — LLM-as-judge. Uses `qwen-turbo` (cheap text model) to score
  the VLM report on 4 dimensions: `scene_id`, `safety`, `domain_awareness`,
  `structure` (each 1–5).
- `run_eval.py` — Runner. Builds the agent, runs it on each golden-set image
  (real VLM cost: ~10 calls), then judges the output. Returns per-image and
  overall scores; exits non-zero if below threshold.
- `ablation_topk.py` — 3-arm tool-calling ablation (off / fixed / agentic).
- `retrieval_eval.py` — retrieval-side metrics: recall@k and MRR against the
  annotated truth, with an optional grounding pass that localises each miss.

## Run locally

```bash
# set API key (required for VLM + judge)
export DASHSCOPE_API_KEY=sk-...

# full run (real VLM + judge, ~3 min, ~¥0.2)
python -m eval.run_eval --threshold 3.7

# deterministic only (no LLM judge, free)
python -m eval.run_eval --skip-judge

# one image for quick iteration
python -m eval.run_eval --id bus_street_side

# save JSON report
python -m eval.run_eval --out eval/runs/run-$(date +%s).json
```

## Retrieval evaluation (recall@k / MRR)

Phase C could only measure *attribution* — whether a finding cited a chunk that
really was retrieved. That answers "did the model invent a reference?", not "was
the clause the finding needed retrievable at all?". `retrieval_eval.py` answers
the second question against the annotated truth.

```bash
export DASHSCOPE_API_KEY=sk-...

# retrieval metrics only (embedding calls only; no VLM, no ONNX model)
python -m eval.retrieval_eval

# also run the grounding layer to localise each miss (extra LLM cost)
python -m eval.retrieval_eval --grounding-mode agentic
```

Two layers are reported **separately**, which is what makes a failure locatable:

| layer | measures | needs |
|---|---|---|
| `R` retrieval | recall@k, MRR — ranking quality | embedding model |
| `G` generation | which findings cite a *relevant* chunk | an LLM |

Per item, the combination yields one of:

- `retrieval_miss` — no truth clause in top-k (the retriever is at fault)
- `retrieved_unused` — a truth clause was available but no finding cited it (the
  generator is at fault)
- `used` — a finding cited a truth clause

### Integrity guards

The truth is only valid for the chunking it was annotated against, so the runner
**fails loudly** (rather than scoring 0) if:

- a standards file's `sha256` differs from the recorded fingerprint, or
- a truth `chunk_id` no longer resolves, or its anchor text is missing.

### Known limits

- Queries are the golden items' `scene` text, **not** the VLM narrative, so the
  figures describe retrievability given an accurate description, not the
  deployed path. Recorded in every output as `query_source_caveat`.
- One item (`marina_aerial`) has an intentionally empty truth set; it is excluded
  from the denominators and reported by id rather than scored.
- The annotation was produced by a single annotator; the two-pass disagreement
  rate is published in `retrieval_truth_meta.json` as an error bar on the truth.

## Run on GitHub Actions

The `eval.yml` workflow runs these commands. It's **`workflow_dispatch` only**
(manual trigger) to control cost — every push would burn money.

Trigger from the Actions tab → "eval" → "Run workflow". Three modes:

| mode | runs | cost |
|---|---|---|
| `eval` | golden-set quality gate (VLM + LLM judge) | VLM + judge calls |
| `ablation` | 3-arm tool-calling comparison | VLM + grounding LLM |
| `retrieval` | recall@k / MRR vs annotated truth | embeddings only (set `grounding` to add the LLM layer) |

**Required secret**: `DASHSCOPE_API_KEY` in
Settings → Secrets and variables → Actions. Add it once (it's already the same
key used locally).

## Score interpretation

- **Overall score** is the mean of per-image scores on a 0–5 scale.
- **Per-image score** = `5 * 0.4 * det_score` (0 or 1) + `0.6 * judge_avg`
  (avg of 4 judge dimensions). So a passing image must:
  1. Pass all deterministic checks (right detection count range, mention the
     right keywords, flag safety when expected), and
  2. Score ≥3.0 average on the LLM judge.
- **Default threshold 3.7** tolerates normal LLM-judge variance. If a CI run
  fails by <0.1, re-run; if consistently below, inspect `reports/eval/ci-run.json`
  to see which dimension lost points.

## Adding images to the golden set

1. Drop the image into `data/test_images/`.
2. Append a new entry to `golden_set/golden_set.json` with: scene description,
   `expect` (detection count range, keywords, safety flag), a `rubric`
   describing what a good report should cover, and a `retrieval_truth` block
   (see `ANNOTATION.md` — annotate against the standards **before** looking at
   what the retriever returns).
3. Add the anchor entries for any new truth chunk ids to
   `golden_set/retrieval_truth_meta.json`.
4. Run `python -m eval.run_eval --id <new_id>` to verify.
5. Commit + push.
