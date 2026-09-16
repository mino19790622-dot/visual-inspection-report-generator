# Interview Notes

Phase-by-phase talking points, grown as the work happens (not saved for the end).
Each entry has **one line you can say in English** and a **boundary** — what you still
must not claim. Do not inflate these; the boundary is part of the material.

---

## Phase C — tool-calling grounded findings

### 1. Circular dependency: found it, then resolved it with tool calling

> "While moving retrieval *before* generation, I hit a circular dependency: the dynamic
> top-k was derived from the VLM's risk level, but the VLM report now needed the retrieved
> context to be written — so one value both fed and depended on generation. I broke the
> cycle by letting the model call a `retrieve_standards` tool and choose k itself, with the
> code clamping k to a safe range and capping the per-request call budget."

- **Boundary (do NOT claim):** single-tenant, single-node, offline/batch evaluation. Not a
  distributed or multi-tenant serving system. The guardrails (k-clamp, call cap) are simple
  code bounds, not a policy engine.

### 2. `mode="off"` — a byte-for-byte equivalent path, kept on purpose

> "When I inserted the grounding stage, I kept a `mode='off'` path that reproduces the
> previous pipeline exactly, so the existing 10-image golden set stayed a valid regression
> net — I could *prove* the upgrade didn't silently change the old behavior instead of
> assuming it."

- **Boundary (do NOT claim):** the regression net is 10 images; it is not a large-scale
  production evaluation. The judge shares the measured model's family — a
  self-enhancement-bias risk that was later checked directly (phase A3/A4, *Judge
  credibility cross-check*). That check found no leniency but did find a verdict flip, so
  3.7 was downgraded from a gate to a **reference value**; it is not, and was never, a
  resolved pass standard.

### 3. Usage is measured, price is configuration — and "no price" ≠ "free"

> "I kept token usage and price as separate concerns: usage is measured from the API
> response, price is a versioned snapshot file. When a rate is missing, the cost function
> returns `known=False` instead of `0.0`, because 'no price configured' and 'this call was
> free' are two different facts — a report that silently shows a free call is worse than one
> that says 'unknown'."

- **Boundary (do NOT claim):** at the time of writing the rate table was `null`, so no cost
  figure existed — only the token accounting and the refusal to fabricate a number. The
  rates were filled in later (phase A4, snapshot `2026-09-16`), so per-image CNY costs now
  exist; they are n = 10, exclude embedding and judge spend, and are reported as intervals
  because the run artifact did not record the input/output token split. That is cost
  *measurement*, not cost governance.

### 4. A number with no provenance — I fixed the claim, not the date

> "My README claimed a 'v1.0 baseline': 4.28 / 5.0. That date was months before the
> repository existed, and the score matched no run record I could find. The number that *was*
> real was a 2026-08-21 evaluation at 4.475 / 5.0 (commit `2036d66`, run `32514897867`) — and
> it ran *before* the `v1.0` tag, so 'the v1.0 baseline score' was never a thing: the tag froze
> the code, not a measurement. I deleted the number instead of hunting for a date to put next
> to it, because the honest correction was to the claim, not to the date."

- **Boundary (do NOT claim):** this is a **documentation-accuracy fix, not a new evaluation
  result** — no measurement was produced, and the only figure kept is the one traceable to a
  cited run. It says nothing about model quality. The existing limits are unchanged: n = 10, and
  the judge shares the measured model's family — a risk later checked directly (phase A3/A4),
  which downgraded 3.7 from a gate to a **reference value**.

---

## Phase A3/A4 — retrieval evaluation, judge credibility, cost close-out

### 5. I pre-registered a check that could overturn my own conclusion, then ran it

> "My judge (`qwen-turbo`) and the models under test (`qwen-plus`, `qwen-vl-max`) are all in
> the Qwen family, so the judge could have been lenient toward its own family and quietly
> inflated the score. Instead of assuming it wasn't, I wrote the cross-check — and fixed its
> decision rule, a three-row table, *before* running it. The measurement came back with the
> bias pointing the safe way: a signed mean difference of −0.175, so the judge was slightly
> *stricter*, not lenient. But exact agreement was only 0.55, the `safety` dimension 0.20, and
> one image flipped pass/fail between the two raters. My pre-registered rule said any flip
> downgrades the threshold, so I executed the downgrade: 3.7 is no longer a per-image gate, it
> is a reference value. The point is not that I found a flaw — it is that I designed a check
> that could falsify my own conclusion, and then accepted the result rather than re-aiming the
> standard after seeing it. I also published the two details that weaken the check itself: the
> rater had already seen the judge's scores, so 0.55 is an upper bound on agreement, and the
> pre-registration named four items while ten were scored."

- **Boundary (do NOT claim):** n = 10, a single rater, and the rater was **not fully blind**
  (the pre-registered `blindness_limits` are on the record). This is not a general calibration
  of the judge — it covers this golden set only, and it does not establish that the judge is
  sound for other images or tasks. It is one check, not an evaluation system.

### 6. Marginal cost, not total cost

> "The question I care about isn't 'is this agent expensive' — it's whether the extra finding
> is worth the price. Against the 2026-09-16 rate snapshot the three arms cost 0.0029–0.0072
> (`off`), 0.0046–0.0116 (`fixed`) and 0.0072–0.0179 CNY per image, so the agentic arm buys
> +20% findings for +55% cost — a marginal 0.0254–0.0634 CNY per net additional finding. I
> report that as an interval rather than a number because the run artifact recorded prompt and
> completion tokens as a single figure. I could have assumed a split and produced one clean
> number, but a range that reflects what I actually measured is more honest than a precise
> figure I made up."

- **Boundary (do NOT claim):** n = 10; embedding and judge spend are **not** included. The
  interval comes from the unrecorded input/output token split — it is not measurement noise,
  and it does not narrow with more samples. This is cost *measurement* on one golden set, not
  cost governance, and not a production cost model.
