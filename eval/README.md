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
- `golden_set/JUDGE_CROSSCHECK.md` — the judge-credibility protocol, including
  the pre-registered decision rule, and the result. Committed *before* the
  cross-check was scored, so the rule could not be chosen after the fact.
- `golden_set/judge_crosscheck_scores.json` — the independent rater's 4-dimension
  scores for all 10 images, keyed by real image id, plus the label mapping used
  during blinding. Reproduces the agreement numbers below.
- `judge.py` — LLM-as-judge. Uses `qwen-turbo` (cheap text model) to score
  the VLM report on 4 dimensions: `scene_id`, `safety`, `domain_awareness`,
  `structure` (each 1–5).
- `run_eval.py` — Runner. Builds the agent, runs it on each golden-set image
  (real VLM cost: ~10 calls), then judges the output. Returns per-image and
  overall scores; exits non-zero if below threshold. The JSON it writes now
  includes the scored `report` text and `rubric`, because a judge score with no
  record of what was scored cannot be re-scored by anyone else.
- `ablation_topk.py` — 3-arm tool-calling ablation (off / fixed / agentic).
- `retrieval_eval.py` — retrieval-side metrics: recall@k and MRR against the
  annotated truth, a `routing_diagnostic` that separates "wrong standards file"
  from "right file, wrong clause", and an optional grounding pass that localises
  each miss. Reports layer R and layer G separately, never blended.
- `cost_table.py` — prices the ablation arms from `config/pricing.yaml` via
  `cost_for`. Emits a `[floor, ceiling]` bracket rather than a point cost,
  because the artifact records prompt+completion as one number and inventing a
  split would be inventing data.
- `judge_agreement.py` — compares the judge against an independent rater:
  agreement rates, mean absolute difference, and the **signed** mean difference
  (which measures bias direction; agreement only measures noise).

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
- **The grounding layer (`--grounding-mode fixed|agentic`) is confounded in this
  harness.** There is no detector here, so the agent runs with
  `detection_summary=""` — not the production input. With nothing detected it
  declines to retrieve on most images (7/9 on the agentic run) and returns zero
  findings, so a `retrieval_miss` verdict there would blame the retriever for a
  lookup that never happened. That is why `not_attempted` exists as its own
  verdict. Read layer-G output as "the R→G chain is reachable", never as a
  retriever verdict. Use `--grounding-mode none` for the layer-R numbers.

### Routing vs granularity

`recall@k` cannot tell "retrieved the wrong standards file" from "retrieved the
right file and ranked the wrong clause", and the two have different fixes.
`routing_diagnostic` splits them from the *same* ranking the recall numbers come
from, so the two can never disagree about what was retrieved. On the n=9 run it
is decisive: every truth standard appears in the top-10 for **9/9** images while
only **4/9** put a truth-bearing standard at rank 1 — so the problem is
clause-level ordering, not routing.

## Cost table

```bash
# needs an ablation artifact (mode=ablation writes reports/ablation/ci-run.json)
python -m eval.cost_table --artifact reports/ablation/ci-run.json \
  --out reports/cost/three-arm.json
```

Rates come from `config/pricing.yaml` through `cost_for`, so the price lives in
exactly one place. Costs are reported as a **[floor, ceiling] bracket**, not a
point value: `ablation_topk.py` records `prompt_tokens + completion_tokens` as a
single number, and input/output are priced differently, so a point cost is not
derivable from what was recorded. Nothing here estimates a rate — a null rate
stays unknown. Embedding and judge spend are reported as **declared gaps** for
the same reason. The CI workflow runs this automatically in `ablation` mode.

## Judge cross-check

```bash
python -m eval.judge_agreement --eval-artifact reports/eval/ci-run.json \
  --hand golden_set/judge_crosscheck_scores.json --threshold 3.7 \
  --rater "independent rater (non-Qwen)"
```

Compares the judge against an independent rater. Primary metrics are agreement
rates and the **signed** mean difference (judge − rater) — agreement measures
noise, the sign measures bias, and bias is the question. Spearman ρ is secondary
on purpose: the scale is 1–5 ordinal, ties dominate, and a coefficient at this n
is easy to over-read. `gate_impact` converts the threshold into the `judge_avg`
the gate actually demands (`(T − 2.0) / 0.6`) and lists images where the two
raters disagree on pass/fail. Result and the pre-registered decision rule:
`golden_set/JUDGE_CROSSCHECK.md`.

## Run on GitHub Actions

The `eval.yml` workflow runs these commands. It's **`workflow_dispatch` only**
(manual trigger) to control cost — every push would burn money.

Trigger from the Actions tab → "eval" → "Run workflow". Three modes:

| mode | runs | cost |
|---|---|---|
| `eval` | golden-set quality gate (VLM + LLM judge) | VLM + judge calls |
| `ablation` | 3-arm tool-calling comparison, then the cost table | VLM + grounding LLM |
| `retrieval` | recall@k / MRR vs annotated truth | embeddings only (`grounding=none`); set `grounding` to `fixed`/`agentic` to add the LLM layer (extra cost, and confounded — see below) |

Inputs: `threshold` (default 3.7), `only` (**comma-separated image ids** — a cheap
way to hand-audit the judge on a subset), `grounding` (`none` / `fixed` /
`agentic`, retrieval mode only), `arms` and `judge` (ablation mode only).

**Required secret**: `DASHSCOPE_API_KEY` in
Settings → Secrets and variables → Actions. Add it once (it's already the same
key used locally).

## Score interpretation

- **Overall score** is the mean of per-image scores on a 0–5 scale.
- **Per-image score** = `5 * 0.4 * det_score` (0 or 1) + `0.6 * judge_avg`
  (avg of 4 judge dimensions). So a passing image must:
  1. Pass all deterministic checks (right detection count range, mention the
     right keywords, flag safety when expected), and
  2. Score ≥2.833 average on the LLM judge at threshold 3.7.
- **3.7 is a reference value, not a per-image gate.** The cross-check
  (`golden_set/JUDGE_CROSSCHECK.md`) found no evidence of judge leniency — the
  judge was marginally *stricter* than an independent rater (signed mean
  difference −0.175) — but exact agreement was only 0.55 and one image flipped
  pass/fail between raters. So do not treat a single image's pass/fail as an
  acceptance decision, and do not read "overall passed 3.7" as a quality claim.
  Judge credibility is *measured* now, not assumed.

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
