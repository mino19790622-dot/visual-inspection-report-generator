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

- **Boundary (do NOT claim):** the regression net is 10 images with a **provisional** judge
  threshold; it is not a large-scale production evaluation. The judge shares the measured
  model's family — a self-enhancement-bias risk that is flagged and validated later
  (phase B), not resolved here.

### 3. Usage is measured, price is configuration — and "no price" ≠ "free"

> "I kept token usage and price as separate concerns: usage is measured from the API
> response, price is a versioned snapshot file. When a rate is missing, the cost function
> returns `known=False` instead of `0.0`, because 'no price configured' and 'this call was
> free' are two different facts — a report that silently shows a free call is worse than one
> that says 'unknown'."

- **Boundary (do NOT claim):** the rate table is still `null`, so **no real monetary cost has
  been produced yet** — the CNY cost column is pending (phase A). What exists today is the
  token accounting and the refusal to fabricate a number, not a cost result.

### 4. A number with no provenance — I fixed the claim, not the date

> "My README claimed a 'v1.0 baseline (Feb 2026): 4.28 / 5.0'. The date was before the
> repository existed, and the score matched no run record I could find. The number that *was*
> real was a 2026-08-21 evaluation at 4.475 / 5.0 (commit `2036d66`, run `32514897867`) — and
> it ran *before* the `v1.0` tag, so 'the v1.0 baseline score' was never a thing: the tag froze
> the code, not a measurement. I deleted the number instead of hunting for a date to put next
> to it, because the honest correction was to the claim, not to the date."

- **Boundary (do NOT claim):** this is a **documentation-accuracy fix, not a new evaluation
  result** — no measurement was produced, and the only figure kept is the one traceable to a
  cited run. It says nothing about model quality. The existing limits are unchanged: n = 10, and
  the judge shares the measured model's family with a **provisional** threshold.
