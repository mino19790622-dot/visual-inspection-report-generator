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
  used by the judge.
- `judge.py` — LLM-as-judge. Uses `qwen-turbo` (cheap text model) to score
  the VLM report on 4 dimensions: `scene_id`, `safety`, `domain_awareness`,
  `structure` (each 1–5).
- `run_eval.py` — Runner. Builds the agent, runs it on each golden-set image
  (real VLM cost: ~10 calls), then judges the output. Returns per-image and
  overall scores; exits non-zero if below threshold.

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

## Run on GitHub Actions

The `eval.yml` workflow runs the same command. It's **`workflow_dispatch` only**
(manual trigger) to control LLM cost — every push would burn money.

Trigger from the Actions tab → "eval" → "Run workflow". You can override the
threshold or run a single image.

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
   `expect` (detection count range, keywords, safety flag), and a `rubric`
   describing what a good report should cover.
3. Run `python -m eval.run_eval --id <new_id>` to verify.
4. Commit + push.
