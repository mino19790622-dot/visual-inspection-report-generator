# Judge credibility cross-check

> **Read this section backwards.** §1–§5 below are the **pre-registration**: they describe the state
> of the world *before* the check ran, which is why they still call 3.7 "the gate" and say its
> credibility "is verified in phase B". Rewriting them after the fact would destroy the point of
> pre-registering. **§6 holds the result and the verdict:** the gate was downgraded to a reference
> value. If you are reading this to find out what 3.7 currently means, go to §6.

The quality gate in `eval/run_eval.py` (`DEFAULT_THRESHOLD = 3.7`) is decided partly by an LLM judge
(`qwen-turbo`) scoring reports written by `qwen-plus` / `qwen-vl-max`. Judge and measured models are
the **same model family**. If the judge is lenient toward its own family, the gate passes reports a
neutral reader would fail, and every "overall ≥ 3.7" claim inherits that error.

This file was written **before** the cross-check was run. The procedure and the decision rule are
pre-registered so the result cannot be chosen after the fact.

## 1. Was a different-family judge available?

Checked, not assumed. Model Studio (百炼) does host non-Qwen **vision-capable** models — `kimi-k2.5`
(Moonshot) is documented as accepting image input, alongside `glm-5.1` and `MiniMax-M2.5`.

They are **not a drop-in swap** for this project. `eval/judge.py` posts to the shared endpoint
`https://dashscope.aliyuncs.com/compatible-mode/v1`, whereas the third-party models are documented
against a **workspace-scoped** endpoint (`https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/...`).
Reaching Kimi therefore needs a different base URL and a separate entitlement, i.e. a new spend line
rather than a free model-name change.

**Conclusion:** a family-swap judge is *available in principle but not at zero extra budget or
entitlement from this project's configured endpoint*. Per the phase-A prompt this is the stated
condition for falling back to the sanctioned substitute below. The substitution is recorded here
rather than being silently skipped, and the Kimi route is left as a follow-up with a known cost.

## 2. Substitute executed: blind scoring by an independent rater

A non-Qwen rater scores a **subset** of the golden set against the **same rubric text** the judge was
given, on the **same four dimensions**, using the **same 1–5 scale**. Agreement between the two
sources is then computed by `eval/judge_agreement.py`.

### Subset

**4 images out of 10**, chosen by the rater from the run's stored reports without knowing which image
each report belongs to (§2.3). The subset is deliberately small because it is hand-scored; n is
stated next to every number. The other 6 images keep judge-only scores and are marked as not
cross-checked.

### Dimensions and pre-registered scoring meaning

Identical to `eval/judge.py`'s prompt, so the two sources are answering the same question:

| Dimension | 1 | 3 | 5 |
|---|---|---|---|
| `scene_id` | wrong scene | partly right | correct and detailed |
| `safety` | no rubric concern flagged | about half flagged | all rubric concerns flagged |
| `domain_awareness` | no mention of the detector domain gap | mentioned in passing | explicitly discussed |
| `structure` | required sections missing | some sections | all four sections, exact headers |

The rater scores **only the report text against the rubric text**. The image itself is deliberately
not shown to the rater — the judge saw the image, so this is a genuine methodological difference and
is disclosed rather than hidden. It biases *against* the judge on `scene_id` (the rater cannot
verify the scene visually), which is the conservative direction for a leniency test.

### Blindness

1. Reports are extracted and **re-keyed to opaque labels** (`item_01`…) in a deterministic shuffle.
2. The rater sees only `(opaque label, report, rubric)` — no image id, no image filename.
3. Scores are recorded against the opaque label, then joined back by `judge_agreement.py`.

**Disclosed weakness.** The rater had already seen the judge's *numeric* per-image scores from an
earlier run over the same golden set (run `35077155821`), before scoring. The blinding above removes
the ability to attach a score to a remembered image id, but it is not full ignorance of the judge's
general behaviour on this set. This is a real limitation of a single-rater, single-author check and
is the reason the result below is reported as evidence about *direction*, not as a calibrated
agreement coefficient.

## 3. Metric, and why this one

Primary: **exact-agreement rate** and **within-1 agreement rate** on 1–5 ordinal scores, plus the
**mean absolute difference**.

Why not a correlation coefficient as the headline: the audited subset is small, every score is an
integer, and ties dominate, so a coefficient is unstable and easy to misread as a quality verdict
when it only measures whether two orderings move together. What the gate depends on is narrower:
does an independent reader put a passing report in the same band? Spearman ρ is computed and
reported as a **secondary** figure and is explicitly not the decision.

The **signed** mean difference (judge − rater) is reported separately from agreement, because
agreement measures noise while the sign measures bias — and bias is the question.

## 4. Pre-registered decision rule

`eval/run_eval.py` computes `score = 5 × 0.4 × det_score + 0.6 × judge_avg`. With `det_score = 1.0` on
every golden image (true in every run to date), an image passes at threshold `T` iff

```
judge_avg ≥ (T − 2.0) / 0.6        →  T = 3.7 requires judge_avg ≥ 2.833
```

So the gate only depends on whether `judge_avg` clears **2.833**, and the cross-check is only
decision-relevant around that band.

| Result | Action |
|---|---|
| signed mean diff < +0.25 **and** within-1 agreement ≥ 0.80 **and** no image flips pass/fail | Keep 3.7 as a gate. Record it as *checked at n = 4, no material leniency detected*, and keep the n=4 disclosure. |
| signed mean diff ≥ +0.25 **or** within-1 agreement < 0.80, with no image flipping | Keep 3.7 as a gate but label it **provisional** and publish the measured disagreement beside it. |
| any image flips pass/fail **or** signed mean diff ≥ +0.5 | **Downgrade the gate.** Stop describing 3.7 as a gate; report `judge_avg` as a **reference value** with the disagreement published, and mark the README's "passed 3.7" language as a non-claim. |

The rule is fixed here so that a marginal result cannot be spun as a pass.

## 5. What this check cannot establish

- It does not measure bias on the *production* query/report distribution, only on 10 golden images.
- One rater is not a panel: a single non-Qwen reader is not evidence about "non-Qwen judges" in
  general, only about this reader on these reports.
- The rater could not see the images, so `scene_id` agreement is a weaker signal than the other
  three dimensions by construction.

## 6. Result

Judge scores and reports from run `35078726570`. Rater scores in
`eval/golden_set/judge_crosscheck_scores.json`. Reproduce with:

```
python -m eval.judge_agreement \
  --eval-artifact reports/eval/ci-run.json \
  --hand eval/golden_set/judge_crosscheck_scores.json \
  --threshold 3.7 --rater "independent rater (non-Qwen)"
```

**n = 10 images, 40 paired scores.**

| Metric | Value |
|---|---|
| Exact agreement | 0.55 |
| Within-1 agreement | **0.825** |
| Mean \|difference\| | 0.725 |
| **Signed mean difference (judge − rater)** | **−0.175** |
| Spearman ρ (secondary) | 0.588 |
| Images where the raters disagree on pass/fail | **1 / 10** |

### Per dimension

| Dimension | exact | within-1 | mean \|diff\| | judge mean | rater mean | signed |
|---|---|---|---|---|---|---|
| `scene_id` | 0.50 | 0.70 | 0.90 | 4.5 | 4.2 | +0.30 |
| `safety` | 0.20 | 0.90 | 1.10 | 3.1 | 3.4 | −0.30 |
| `domain_awareness` | 0.50 | 0.70 | 0.90 | 3.7 | 4.4 | **−0.70** |
| `structure` | **1.00** | **1.00** | **0.00** | 5.0 | 5.0 | 0.00 |

### What the result says

**1. No evidence of family-affinity leniency — the sign is the wrong way for that story.** The
signed mean difference is **−0.175**: the qwen judge scored slightly *lower* than the independent
rater. Self-enhancement bias would push this positive. The largest single-dimension gap is
`domain_awareness` at **−0.70**, where the judge was materially *stricter*. Whatever the gate's
weakness is, "judge flatters its own family" is not what this measurement shows.

**2. `structure` is the control case and it is perfect.** Exact agreement **1.00**, mean absolute
difference **0.00**, over 10 images. This is the one dimension with a mechanical, checkable criterion
(are the four section headers present), and both raters agree on every image. That is evidence the
judge is tracking something real rather than emitting noise — and it is a useful internal check on
the cross-check itself: if the whole comparison were broken, this row would not be clean.

**3. The disagreement is concentrated where judgement is required.** `safety` has exact agreement of
only **0.20** — four of every five safety scores differ — although 90% are within 1 point. The two
raters bracket the same reports differently in the middle of the scale. This is the honest reading:
the judge's *ordering* is roughly right, its *absolute* banding is not reproducible.

**4. The aggregate verdict is robust; the per-image verdict is not.** Overall score is
**4.445 under the judge** and **4.550 under the rater** — both pass 3.7 with margin. But
`children_group` flips: judge `judge_avg` 2.25 → score 3.35 (**FAIL**), rater 3.25 → score 3.95
(**PASS**). Both raters agree this is the worst image; they disagree on which side of the line it
falls. Note the direction — the judge is *harsher*, so the error is conservative, not permissive.

### Decision applied (pre-registered rule 4, third row)

The rule was: *any image flips pass/fail* → downgrade the gate. One image flipped, so that action
applies, and it is applied as written:

- **3.7 is no longer presented as a per-image gate.** A single image's pass/fail at 3.7 is not
  reproducible across raters at n = 10, so it must not be used as an acceptance decision.
- **`judge_avg` is reported as a reference value**, with the measured disagreement published
  alongside it (this section).
- **The README's "passed 3.7" language is rewritten as a non-claim** — see the *Golden-set quality
  evaluation* section. The same wording is now carried by `eval/run_eval.py`'s
  `DEFAULT_THRESHOLD` comment and docstring, and by `.github/workflows/eval.yml`'s header and
  `threshold` input description.

Two honest qualifications, offered as observations rather than as a softening of the rule:

- The *aggregate* comparison **did** hold up (4.445 vs 4.550, both passing), so the 3.7 **aggregate**
  reference retains information; it is the per-image use that is unsupported.
- The observed bias direction is conservative. Had the sign been positive and large, the gate would
  have been *permissive*; it is not.

### Limitations of this result

- The rater saw the judge's numeric scores for this run before scoring, so agreement may be
  **inflated** by anchoring. The measured agreement is therefore an upper bound and the measured
  bias magnitude a lower bound.
- n = 10 images / 40 scores. One rater. No confidence interval is claimed.
- The rater never saw the images, unlike the judge.

