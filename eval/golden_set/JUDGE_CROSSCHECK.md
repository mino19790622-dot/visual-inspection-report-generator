# Judge credibility cross-check

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
