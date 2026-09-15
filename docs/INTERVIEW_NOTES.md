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
