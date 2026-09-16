# Visual Inspection Report Generator

[![regression](https://github.com/mino19790622-dot/visual-inspection-report-generator/actions/workflows/regression.yml/badge.svg)](https://github.com/mino19790622-dot/visual-inspection-report-generator/actions/workflows/regression.yml)

AI-powered visual inspection system: object detection → VLM visual reasoning → RAG standards retrieval → structured inspection report.

**v2:** findings are produced by a text agent that calls a `retrieve_standards` **tool** and cites the exact clauses it retrieved *this request*; a `mode=off` path reproduces the previous pipeline byte-for-byte as a regression baseline; token usage is recorded separately from pricing, which refuses to render an unknown rate as `0.0`.

## Architecture

```
Image → YOLOv8 Detection → Qwen-VL Analysis → RAG Standards Match → Inspection Report
```

| Layer | Technology | Status |
|-------|-----------|--------|
| Detection | YOLOv8m (Ultralytics + ONNX Runtime) | ✅ Complete |
| VLM Analysis | Qwen-VL-Max via Alibaba DashScope | ✅ Complete |
| RAG Standards Retrieval | DashScope Embeddings + ChromaDB | ✅ Complete |
| Report Generation | Markdown + JSON export, timestamped | ✅ Complete |
| Grounded Findings (v2) | tool-calling + citation verification (`app/grounding.py`, `app/citations.py`) | ✅ Complete |
| Agent Orchestration | LangGraph StateGraph | ✅ Complete |
| API Layer | FastAPI + uvicorn | ✅ Complete |
| Deployment | Docker + docker-compose | ✅ Complete |

## Key Features

- **Detection (D1-D2)**: YOLOv8m with hand-written ONNX inference pipeline, PyTorch/ONNX consistency validated
- **VLM Analysis (D3)**: Qwen-VL-Max generates 4-section structured reports (Scene / Inventory / Detector Gaps / Risk). Detection JSON injected into prompt to reduce hallucination; domain-gap awareness catches missed objects and false positives
- **RAG Standards (D4)**: 6 inspection standards (BS EN 1992, ISO 45001, EN 13134, ISO 23953, ISO 14001, pedestrian safety) embedded via DashScope text-embedding-v2, stored in ChromaDB (cosine similarity), retrieved per report
- **Report Export**: timestamped Markdown + JSON + bbox-annotated image saved to `reports/` on every run
- **Agent Orchestration (D5)**: LangGraph StateGraph with real decision logic —
  - *Adaptive re-detection*: zero detections → automatically lower confidence threshold and retry once
  - *Grounded findings via tool-calling (v2)*: after VLM analysis, a text agent (`app/grounding.py`) calls a `retrieve_standards` tool to pull the clauses it needs and emits structured findings whose `citation` points at chunks it actually retrieved **this request**. Retrieval depth is decided by the model (clamped `k∈[1,5]`, call-budgeted in `app/tools.py`) — not a hard-coded risk table. `mode="off"` reproduces the v1.0 path byte-for-byte and serves as the regression baseline.
  - All decisions logged in state and exported in every report
- **API Layer (D6)**: FastAPI service — `POST /inspect` (image upload → full JSON report), `GET /standards`, `GET /health`

## Verified Behaviors

| Scenario | Detection | VLM Output | RAG Retrieval |
|----------|-----------|------------|---------------|
| Urban bus scene | 1 bus + 4 persons ✅ | Correct inventory, no gaps | Public transport + pedestrian standards |
| Group photo (5 people) | 4 persons (missed 1) | Caught the missed 5th person | Pedestrian safety standard |
| Aerial construction site | 3 false positives | Identified all as domain-gap errors | Construction safety + hazard thresholds |

## Testing

217 unit/integration tests (coverage measured at 91%; the CI gate is set to 80%), zero network access required — the ONNX session, DashScope embeddings, and Qwen-VL calls are all mocked at the module boundary:

- `test_detection.py` — letterbox preprocessing, bbox decoding roundtrip, confidence filtering, NMS suppression/clipping
- `test_agent.py` — risk classification (explicit statement vs keyword fallback, negation safety), routing logic, and a **full LangGraph run** verifying adaptive re-detection and tool-calling grounded findings (`mode=off` reproduces v1.0)
- `test_rag.py` — chunking (header split / overlap / fragment filter) and retrieval against a real in-memory ChromaDB with deterministic hash embeddings
- `test_api.py` — FastAPI `TestClient`: schema contract, 415/500 error paths, mocked agent
- `test_exporter.py` — Markdown/JSON report content, graceful annotation failure
- `test_observability.py` — structured JSONL log writer (incl. unwritable-dir fallback)

```bash
pip install -r requirements-ci.txt
ruff check app eval tests
pytest --cov=app --cov-fail-under=80
```

The CI cost split: **every push runs the deterministic regression suite**
(`.github/workflows/regression.yml` — all mocked, zero API cost, and the only per-push gate),
while the **LLM golden-set evaluation is manual** (`.github/workflows/eval.yml` — needs a real
key and spends money). The LLM evaluation produces a score and a reference comparison; it is not
gated, and since the judge cross-check its 3.7 value is explicitly a reference, not an acceptance
bar (see *Golden-set evaluation* below).

## Golden-set Evaluation (manual, judge-scored)

Beyond unit tests, the VLM's *output quality* is regression-tested against a hand-curated golden set:

- `eval/golden_set/golden_set.json` — 10 images (street scenes + aerial construction / port / parking / beach) with expected detection counts, must-mention keywords, safety vocabulary, and a per-image rubric
- `eval/judge.py` — LLM-as-judge using `qwen-turbo` scoring 4 dimensions (scene ID, safety, domain awareness, structure) 1–5 each
- `eval/run_eval.py` — runs the full agent on each image (VLM cost: ~10 calls, fractions of a ¥) + judge, aggregates per-image and overall scores (0-5 scale), exits non-zero if below threshold

```bash
DASHSCOPE_API_KEY=... python -m eval.run_eval --threshold 3.7
# skip the LLM judge (deterministic only, free):
python -m eval.run_eval --skip-judge
# one image at a time for quick iteration:
python -m eval.run_eval --id bus_street_side
```

A separate `eval.yml` workflow lets you run this on demand from the Actions tab (avoids the LLM cost on every push). Most recent pre-v2 evaluation on `main`: **overall 4.475 / 5.0** (2026-08-21, commit `2036d66`, run `32514897867`). The `v1.0` tag (2026-09-15) froze the code baseline but no evaluation was run at the tag, so there is no v1.0-tagged score. The v2 numbers are in [Evaluation](#evaluation) below.

## Evaluation

All numbers below come from actual runs of `eval/run_eval.py` / `eval/ablation_topk.py` on the
10-image golden set (`eval/golden_set/golden_set.json`). **Sample size n = 10** — small; treat the
numbers as directional, not a production benchmark.

### Tool-calling ablation — agentic vs fixed vs off

Run `35018667423` (`eval/ablation_topk.py --arms off fixed agentic`), commit `6b82f8a`, **n = 10**,
grounding temperature 0, one pass per image.

| Arm | Findings | Citations ok/unresolved/hallucinated | Attribution pass rate | Retrieval calls (mean) | k chosen | Grounding tokens | Parse failures | Deterministic checks |
|---|---|---|---|---|---|---|---|---|
| `off` (v1.0 narrative) | 0 | – | – | 0.0 | – | 0 | 0/10 | **10/10** |
| `fixed` (k=5 injected) | 5 | 5 / 0 / 0 | **100%** | 1.0 | 5 (fixed) | 21,305 | 1/10 | 10/10 |
| `agentic` (model decides k) | 6 | 6 / 0 / 0 | **100%** | 1.8 | 2 (modal) | 53,352 | 0/10 | 10/10 |

**What this actually says (no spin):**
- The `off` arm is the regression baseline: the v1.0 narrative path still passes **all 10**
  deterministic checks, so Phase 0 did not break the old pipeline.
- Both `fixed` and `agentic` reached **100% attribution** — no hallucinated citations in this sample.
- `agentic` produced one more finding and used a *smaller* `k` (2 vs 5), **but** it spent ~1.8× the
  retrieval calls and ~2.5× the grounding tokens (53.4k vs 21.3k). Within n=10 this reads as
  **"spent more", not "strategy better"** — the extra cost is the multi-round tool loop. We do **not**
  claim agentic is superior.
- The "+1 finding" is a **net**, and it hides churn in both directions: `agentic` gained 3 findings
  across two images (`large_building_aerial` 0→1, `industrial_lot_aerial` 1→3) and **lost 2** across
  two others (`bus_street_side` 1→0, `construction_foundation_aerial` 3→2). So it is not "one extra
  finding" — it is a different set of findings that happens to be one larger. See *Cost & Latency* for
  what that churn cost.

### Retrieval quality — recall@k / MRR

Run `35079424448` (`eval/retrieval_eval.py`, mode `retrieval`, `grounding=none`), commit `0f2720b`.
Ground truth is the `retrieval_truth` annotation in `eval/golden_set/golden_set.json`; the protocol is
in `eval/golden_set/ANNOTATION.md` and was committed *before* any truth was written. **n = 9 scored**
(`marina_aerial` is excluded — it has no applicable clause, and scoring it would invent either a
success or a failure). Corpus: 50 chunks across 6 standards.

| k | 1 | 2 | 3 | 5 | 10 |
|---|---|---|---|---|---|
| **recall@k** (n = 9) | 0.0556 | 0.1389 | 0.1389 | 0.3426 | 0.5370 |

**MRR = 0.4074** (n = 9). **2 of 9** images have no truth chunk anywhere in the top-10.

`k = 5` is the depth `fixed` uses and `k = 2` the depth `agentic` modal-chooses, so rows 5 and 2 are
the two that correspond to the arms actually in use.

> **What the query was.** These numbers use the golden set's `scene` text as the query string. In
> production the query is the VLM *narrative*, which is longer and differently worded. These figures
> therefore characterise the retriever under a fixed, reproducible query — not the deployed query
> distribution. The artifact records this as `query_source_caveat`.

**Where the failure is — and it is not routing.** recall@k alone cannot distinguish "went to the
wrong standards file" from "went to the right file, ranked the wrong clause", and those need
different fixes. Splitting them (`routing_diagnostic`) is unambiguous at this n:

| Diagnostic | Result |
|---|---|
| Rank 1 comes from a standard that appears in the truth | **4 / 9** |
| *Any* truth standard present anywhere in the top-10 | **9 / 9** |
| *Every* truth standard present in the top-10 | **9 / 9** |

The retriever **always reaches the right standards file** and then ranks the wrong clauses inside it.
Concretely: for `beach_walkers` the truth is `pedestrian_safety.md::3` and `::8`, and 7 of the top-10
chunks come from `pedestrian_safety.md` — but at positions 1, 2, 4, 5, 7, 8, 9, never 3 or 8.

So the fix belongs in **clause-level ordering (chunking, reranking, or a standards-aware score)**,
not in query routing or a discipline filter. That is the single most actionable result in this
section, and it is the opposite of what a bare recall number would suggest.

**Retrieval and generation are reported separately.** Layer R (above) and layer G are computed from
the same run but never blended, so a missing citation can be attributed to one or the other. Layer G
below is from run `35075045463` (`grounding=agentic`), which is the run where the agentic arm was
actually exercised:

| Layer-G verdict | Count | Meaning |
|---|---|---|
| `not_attempted` | 7 / 9 | the agent returned **without calling the retriever at all** |
| `retrieval_miss` | 1 / 9 | it called, and no truth chunk came back |
| `retrieved_unused` | 1 / 9 | a truth chunk came back and was not cited |

> **Layer G is confounded in this harness and must not be read as a retriever verdict.** The harness
> has no detector, so it runs the grounding agent with an empty `detection_summary`, which is *not*
> the production input (image → YOLO → narrative → grounding). With nothing detected, the agent
> declines to retrieve on 7 of 9 images and returns **zero findings in total**. `not_attempted` is
> therefore kept as a separate verdict rather than folded into `retrieval_miss`: blaming the
> retriever for a lookup that never happened would be the opposite of what this split is for. What
> layer G *does* establish is that the full `retrieved_unused` chain is reachable and is observed —
> see the variance note below.

> **Reproducibility, and a limit on it.** Layer R gave **identical** numbers on three independent runs
> (`recall@1 0.0556 … MRR 0.4074` on `35073538882`, `35075045463`, `35079424448`), including once with
> grounding disabled, so the ranking result is stable. Layer G did **not** reproduce: at grounding
> temperature 0, `industrial_lot_aerial` produced 1 finding citing truth (`used`) on run
> `35073538882` and 0 findings with truth retrieved but uncited (`retrieved_unused`) on
> `35075045463`. Treat layer-G per-image verdicts as single-sample observations, not measurements.

> **Annotation instability.** Truth was annotated in two independent passes; mean Jaccard agreement
> was **0.708** (disagreement 29.2%). Both passes ran in the same session rather than ≥24 h apart, so
> 0.292 is a **lower bound** on the instability. This is recorded in
> `eval/golden_set/retrieval_truth_meta.json`, not smoothed over.

> **n = 9 is not enough for a confidence claim.** A 95% interval on a proportion at n = 9 spans
> roughly ±0.25–0.30. Every number above is directional. See `ANNOTATION.md` §6 for the expansion
> recommendation and §7 for why a three-way calibration/tuning/test split is infeasible at this size
> (the whole set is the test set, and tuning retriever parameters against it is prohibited).

### Golden-set quality evaluation

| Metric | Value | Notes |
|---|---|---|
| Overall quality score | **4.445 / 5.0** | eval run `35078726570` (agentic), judge-scored |
| Overall, independent rater | **4.550 / 5.0** | same run, non-Qwen rater, report-only — see below |
| VLM tokens / 10 images | ~18.1k | consistent across arms (18.1–18.3k) |
| 3.7 comparison value | **reference value, not a gate** | see *Judge credibility* below |

> **Judge credibility is now measured, and 3.7 is downgraded to a reference.** A pre-registered
> cross-check against an independent non-Qwen rater (`eval/golden_set/JUDGE_CROSSCHECK.md`, n = 10
> images / 40 paired scores) found:
>
> - **No evidence of self-enhancement bias.** Signed mean difference (judge − rater) = **−0.175**;
>   the judge is marginally *stricter*, not more lenient. The largest single gap is
>   `domain_awareness` at −0.70, again stricter.
> - **`structure` agrees perfectly** (exact 1.00, mean |diff| 0.00), which is the dimension with a
>   mechanical criterion — evidence the judge is measuring something real.
> - **But absolute banding is not reproducible.** Exact agreement 0.55 overall; `safety` only **0.20**
>   (90% within 1 point). One image, `children_group`, flips verdict: judge 2.25 → score 3.35 (FAIL),
>   rater 3.25 → score 3.95 (PASS). The judge being the harsher one makes the error conservative.
> - The **aggregate** comparison held (4.445 vs 4.550, both well above 3.7), so the aggregate
>   reference still carries information; it is the **per-image** use of 3.7 that is unsupported.
>
> Consequently: do not read a single image's 3.7 pass/fail as an acceptance decision, and do not
> read "overall passed 3.7" as a validated quality claim. `judge_avg` is a reference value with the
> disagreement above published beside it.

> **Variance caveat.** These are single-pass runs. A second agentic pass produced different per-image
> findings (e.g. `large_building_aerial`: 3 findings in the eval run vs 1 in the ablation), and the
> `fixed` arm hit 1/10 parse failures while `agentic` hit 0/10. Treat per-image counts as noisy; the
> aggregate claims above are the ones we stand behind.

> **Judge credibility: checked, and the result changed the gate's status.** The judge and the
> measured models are the same family, so a self-enhancement bias was possible; it is now measured
> rather than assumed. The check found no evidence of leniency (the judge is marginally *stricter*)
> but did find that absolute per-image banding is not reproducible — see the note above and
> `eval/golden_set/JUDGE_CROSSCHECK.md`. The check was pre-registered before scoring, and it used the
> *substitute* route (blind independent scoring) because a different-family vision judge
> (`kimi-k2.5` on Model Studio) needs a separate endpoint and entitlement rather than a free
> model-name swap.

## Cost & Latency

Token usage is measured from the API response and recorded per stage in `logs/inspect_spans.jsonl`.
Money is **derived** from `config/pricing.yaml` (a versioned price snapshot), never estimated in code.

**Rate snapshot: `effective_date: 2026-09-16`, currency CNY, China (Beijing) region, pay-as-you-go**,
taken from the Alibaba Cloud Model Studio (百炼) price tables and cited in `config/pricing.yaml`.

| Model | Input / 1k tokens | Output / 1k tokens |
|---|---|---|
| `qwen-vl-max` (VLM narrative) | 0.0016 | 0.004 |
| `qwen-plus` (grounding) | 0.0008 | 0.002 |
| `text-embedding-v2` (retrieval) | 0.0007 | free (input-only model) |
| `qwen-turbo` (judge) | 0.0003 | 0.0006 |

### Three-arm cost per image

Derived by `eval/cost_table.py` from ablation run `35018667423` (n = 10), which is also a CI step in
ablation mode. **Costs are ranges, not point values** — see the bracket note below.

| Arm | Findings | Retrieval calls / image | VLM tokens | Grounding tokens | Cost / image (CNY) |
|---|---|---|---|---|---|
| `off` | 0 | 0.0 | 18,116 | 0 | 0.002899 – 0.007246 |
| `fixed` | 5 | 1.0 | 18,286 | 21,305 | 0.004630 – 0.011575 |
| `agentic` | 6 | 1.8 | 18,108 | 53,352 | 0.007165 – 0.017914 |

The VLM token count is flat across arms (18.1–18.3k). Everything an arm costs on top of `off` is
grounding: **~2.5×** for `fixed`, **~2.9×** for `agentic` relative to `fixed`'s grounding spend.

> **Why a range.** `eval/ablation_topk.py` records `prompt_tokens + completion_tokens` as one
> number. Input and output are priced differently, and the split is not in the artifact, so a point
> cost is not derivable from what was recorded. The two ends are the same token total priced
> entirely as input and entirely as output; the truth is inside. Narrowing it needs a change in the
> recorder, not an assumption here. **No rate in this table was estimated.**

### What the extra agentic finding cost

`agentic` produced 6 findings against `fixed`'s 5 — a **net** +1. That figure is a net, and reading
it as "one more useful finding" would be wrong:

| Image | Findings `fixed → agentic` | Retrieval calls | Grounding tokens |
|---|---|---|---|
| `bus_street_side` | 1 → 0 (**−1**) | 1 → 3 | 1,960 → 7,900 |
| `construction_foundation_aerial` | 3 → 2 (**−1**) | 1 → 6 | 2,754 → 5,173 |
| `large_building_aerial` | 0 → 1 (**+1**) | 1 → 5 | 2,021 → 19,421 |
| `industrial_lot_aerial` | 1 → 3 (**+2**) | 1 → 1 | 2,141 → 4,326 |

Per-image counts move in **both directions** — 3 gained across two images, 2 lost across two others.
The honest unit price is therefore *cost per net finding*, not cost per extra finding:

- **+8 retrieval calls** and **+32,047 grounding tokens** across the 10 images;
- **+0.02535 – 0.06339 CNY** for the whole set, i.e. **+0.002535 – 0.006339 CNY per image**;
- **0.0254 – 0.0634 CNY per net extra finding** (≈ +55% per-image cost for a +20% finding count).

**Verdict: not demonstrated to be worth it.** No measured quality axis improved — attribution pass
rate is 100% for both arms and deterministic checks are 10/10 for both. The one axis that moved is
a raw count that also fell on other images, and the cost is concentrated: `large_building_aerial`
alone spent 19,421 grounding tokens (~10× `fixed`) to turn 0 findings into 1. On this evidence the
agentic arm's extra spend is not justified; the `fixed` arm is the better default until an
end-to-end metric (not a finding count) shows otherwise.

### Declared gaps (not estimated)

- **Input/output split** — not recorded, hence the brackets above.
- **Embedding cost** — `text-embedding-v2` calls are not priced into these figures: the artifact
  counts retrieval *calls*, not the embedding tokens a call consumes. So the retrieval side of every
  arm is understated by an unknown amount. At 0.0007 CNY/1k input this is small relative to the
  grounding spend, but it is unmeasured, not assumed to be zero.
- **Judge cost** — `qwen-turbo` scores are optional and `mean_judge` is `null` in this run, so judge
  spend is not derivable from the artifact.

| Metric | Value | Notes |
|---|---|---|
| VLM tokens / 10 images | ~18.1k | nearly identical across arms (18.1–18.3k) — grounding does not change the VLM call |
| Grounding tokens / 10 images | `off` 0 · `fixed` 21.3k · `agentic` 53.4k | agentic's multi-round tool loop is ~2.5× `fixed` |
| Cost in CNY | `off` 0.0029–0.0072 · `fixed` 0.0046–0.0116 · `agentic` 0.0072–0.0179 per image | snapshot 2026-09-16; ranges because the in/out split is unrecorded |
| Latency / stage | measured | detect / VLM / ground / export per request in the trace JSONL |

## Limitations

Stated up front — not as a disclaimer but as the boundary of what the numbers above mean:

- **Sample size n = 10.** Every measured number is directional, not a benchmark. There is no held-out test split and no stratified sampling.
- **Judge credibility is measured, and it constrains how the gate may be used.** The self-enhancement
  bias that worried us was **not** found — signed mean difference (judge − rater) is −0.175, i.e. the
  judge is marginally stricter. But exact agreement with an independent rater is only 0.55
  (`safety`: 0.20) and one image flips pass/fail, so **3.7 is a reference value, not a per-image
  acceptance gate**, and "overall passed 3.7" is not a quality claim. Single rater, n = 10, and the
  rater had seen the judge's numbers beforehand, so agreement may be **inflated** — the measured
  agreement is an upper bound.
- **Single-node, single-tenant, offline batch evaluation.** No orchestration/scheduling layer, no horizontal scaling, no multi-tenant isolation — the API serves one process.
- **Not deployed on Kubernetes.** Container orchestration stops at AWS ECS Fargate; there are no K8s manifests, autoscalers, or service mesh.
- **Cost is now filled, but as a range and not for the whole pipeline.** `config/pricing.yaml` carries
  real rates as of 2026-09-16 (CNY), so `known=True` for the four billed models. Two gaps remain and
  are declared rather than covered: costs are reported as `[floor, ceiling]` brackets because the
  ablation artifact does not record the input/output token split, and embedding + judge spend is not
  derivable from it at all. See *Cost & Latency*.
- **Retrieval quality is measured, and it is weak.** recall@k / MRR now exist over annotated ground
  truth — see *Retrieval quality* — and the headline is that recall@10 is ~0.54 while every truth
  standard is reached. Read the section before quoting a single number: n = 9, one annotator, and
  ground truth is pinned to a specific chunking configuration.
- **Retrieval ground truth is pinned to the current chunking.** Truth is stored as `chunk_id`
  (`<file>.md::<index>`), which is a pure function of `CHUNK_SIZE` / `CHUNK_OVERLAP` / the split
  rule. Change any of those and the ids renumber silently. `eval/golden_set/retrieval_truth_meta.json`
  records a fingerprint and `eval/retrieval_eval.py` refuses to run when it no longer matches.

## Observability

Every `/inspect` call appends one JSON line to `logs/inspect.jsonl` (path overridable via `INSPECT_LOG_DIR`):

```json
{"ts":"2026-09-15T...","image":"abc.jpg","risk_level":"high",
 "detection":{"object_count":3,"inference_ms":47},
 "vlm":{"prompt_tokens":1200,"completion_tokens":350,"total_tokens":1550,
        "latency_ms":1800,"cost_rmb":null},
 "retrieval":{"top_k":2,"standards_count":2},
 "total_latency_ms":2400,"saved":["report","annotated"]}
```

A per-stage trace (spans + tokens) is written to `logs/inspect_spans.jsonl`. `top_k` is now the depth the grounding agent chose for this request (not a risk-level lookup). Cost is **derived** from `config/pricing.yaml` (a versioned price snapshot, real rates as of 2026-09-16), not hard-coded in code. A stage whose model has no rate reports `known=False` with `cost: null` and is **never rendered as 0.0** — see `app/observability/pricing.py`.

## Quick Start
```bash
# Clone
git clone https://github.com/mino19790622-dot/visual-inspection-report-generator.git
cd visual-inspection-report-generator

# Setup
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure API key (Alibaba Cloud DashScope)
echo "DASHSCOPE_API_KEY=sk-your-key" > .env

# Run full pipeline (detection + VLM + RAG + report export)
python run_pipeline.py data/test_images/bus.jpg

# Or run the LangGraph agent (adds adaptive re-detection + tool-calling grounded findings)
python run_agent.py data/test_images/bus.jpg

# Start the API service
uvicorn app.api.server:app --port 8000
curl -F "image=@data/test_images/bus.jpg" http://localhost:8000/inspect
```

### Docker

```bash
# One command: build image + start container
docker compose up --build -d

# Test
curl http://localhost:8000/health
curl -F "image=@data/test_images/bus.jpg" http://localhost:8000/inspect

# Stop
docker compose down
```

Docker notes:
- **Slim runtime image**: `requirements.docker.txt` excludes torch/ultralytics (build-time only) — serving needs ONNX Runtime only, cutting image size by ~2 GB
- **Secrets stay out of the image**: `DASHSCOPE_API_KEY` injected via `env_file` at runtime
- **Volumes**: `reports/` and `uploads/` are bind-mounted (results visible on host); the ChromaDB vector index lives in a named volume and persists across rebuilds
- OpenAPI docs at `http://localhost:8000/docs`

### Cloud Deployment (AWS)

The Docker image is cloud-ready and **deployed live on AWS ECS Fargate** (eu-west-1) — the same image also runs on AWS App Runner (fully managed, public HTTPS URL, autoscaling; blocked only by new-account activation lag).

```bash
# Option A — App Runner (fully managed)
export AWS_ACCOUNT_ID=123456789012
export APP_RUNNER_SERVICE_ARN=$(aws apprunner create-service \
  --cli-input-yaml file://deploy/apprunner.yaml \
  --query 'Service.ServiceArn' --output text)
./scripts/deploy.sh   # build → push to ECR → roll out

# Option B — ECS Fargate (works on brand-new accounts)
aws ecs create-cluster --cluster-name default
aws ecs register-task-definition --cli-input-json file://deploy/ecs-task-definition.json
aws ecs run-task --cluster default --launch-type FARGATE \
  --task-definition visual-inspection-api:2 \
  --network-configuration "awsvpcConfiguration={subnets=[<subnet-id>],assignPublicIp=ENABLED,securityGroups=[<sg-id>]}"
# Full walkthrough: docs/AWS_DEPLOYMENT.md
```

- `deploy/apprunner.yaml` — App Runner service definition (health check on `/health`, secret from AWS Secrets Manager, 1 vCPU / 2 GB).
- `deploy/ecs-task-definition.json` — Fargate task definition (awsvpc networking, secret injected from Secrets Manager, CloudWatch logs, container health check).
- `scripts/deploy.sh` — login → build → push to ECR → start deployment.
- `.github/workflows/deploy.yml` — on every push to `main`, authenticates via OIDC (no stored keys), builds, pushes to ECR, and triggers App Runner. The ONNX model is fetched from S3 at build time (it is gitignored).

> **Note:** `yolov8m.onnx` (~100 MB) is gitignored. For local deploys keep it in the project root; for CI, store it in S3 and set `MODEL_S3_URI` (see the workflow file).

```bash
# Custom options
python run_pipeline.py your_image.jpg --conf 0.35 --k 5 --save-dir my_reports

# Detection only
python run_detection.py --model yolov8m.pt --source data/test_images/bus.jpg
```

Each run exports to `reports/`: `{image}_{timestamp}.md` (human-readable report), `.json` (machine-readable), `_det.jpg` (annotated image).

## Tech Stack

- **Python 3.12** | PyTorch 2.2.2 | Ultralytics 8.4 | ONNX Runtime 1.23
- **VLM**: Qwen-VL-Max (Alibaba DashScope, OpenAI-compatible API)
- **RAG**: ChromaDB 1.5 + DashScope text-embedding-v2
- **Deployment**: Docker (python:3.12-slim, runtime-only deps) + docker-compose; deployed live on AWS ECS Fargate + ECR (CI/CD via GitHub Actions OIDC)
- **Quality**: pytest 217 tests (91% measured coverage; CI gate at 80%) + LLM-as-judge golden-set eval (10 images, manually triggered; its score is compared against a **reference** value, not an acceptance gate) + JSONL cost/latency observability on every `/inspect`

## Project Structure

```
├── app/
│   ├── agent/graph.py             # LangGraph StateGraph: adaptive re-detection + ground node
│   ├── api/server.py              # FastAPI: POST /inspect, GET /standards, GET /health
│   ├── detection/detector.py      # YOLODetector (ONNX, letterbox + NMS)
│   ├── vlm/client.py              # VLMClient (Qwen-VL-Max, image auto-resize)
│   ├── rag/retriever.py           # StandardsRetriever (embed + ChromaDB); returns stable chunk_id
│   ├── reporting/exporter.py      # ReportExporter (Markdown + JSON + annotated image + grounded findings)
│   ├── citations.py               # v2: shared L1 citation verifier (eval + future guardrails)
│   ├── tools.py                   # v2: tool schemas + ToolRegistry (k clamp [1,5], call budget)
│   ├── grounding.py               # v2: tool-calling GroundingAgent -> structured, cited findings
│   └── observability/             # v2: per-request spans + allow-list redaction + price-snapshot costing
│       ├── spans.py               #   RequestTrace / Span (latency + token accounting)
│       ├── redaction.py           #   field allow-list; no API key / base64 / image lands in logs
│       └── pricing.py             #   cost_for(); a null rate yields known=False, never 0.0
├── config/
│   └── pricing.yaml               # v2: versioned price snapshot (rates unset on purpose)
├── eval/                          # Golden-set + LLM-as-judge + ablation
│   ├── golden_set/golden_set.json
│   ├── judge.py                   # qwen-turbo judge, 4 dimensions
│   ├── run_eval.py                # eval runner (deterministic + judge, --grounding-mode)
│   ├── ablation_topk.py           # v2: 3-arm ablation (off / fixed / agentic)
│   └── README.md
├── tests/                         # pytest suite (217 tests, 91% coverage, no network)
│   ├── fakes.py                   # v2: ScriptedClient — fake LLM for grounding tests
│   ├── test_citations.py          # v2
│   ├── test_tools.py              # v2
│   ├── test_grounding.py          # v2
│   ├── test_spans.py              # v2
│   ├── test_log_redaction.py      # v2
│   ├── test_ablation.py           # v2
│   └── …                          # detection / vlm / rag / agent / api / exporter / observability
├── .github/workflows/
│   ├── regression.yml             # the per-push gate: ruff + full mocked suite + coverage (zero API cost)
│   ├── eval.yml                   # manual: golden-set eval OR 3-arm ablation
│   └── deploy.yml                 # OIDC → ECR → ECS on push to main
├── data/                          # 6 inspection standards + sample test images
├── docs/                          # USER_GUIDE(_zh) / AWS_DEPLOYMENT(_zh) / V2_STATUS / INTERVIEW_NOTES / V2_GAP_ANALYSIS
├── deploy/  scripts/              # AWS task definitions + deploy.sh
├── run_pipeline.py  run_agent.py  export_onnx.py  compare_d1_d2.py
├── CHANGELOG.md
├── Dockerfile / docker-compose.yml / .dockerignore
└── requirements.txt / requirements.docker.txt / requirements-ci.txt
```

## Author

**Mino Zhang** — Builds LLM-agent systems with tool calling, retrieval-grounded citations and a zero-cost CI regression gate, on a 3-year computer-vision foundation
- GitHub: [@mino19790622-dot](https://github.com/mino19790622-dot)
- Background: 3 years CV algorithm engineering (PyTorch/YOLOX/TensorRT)
- MSc Computer Science candidate @ Maynooth University, Ireland

## Documentation

- [docs/USER_GUIDE.md](docs/USER_GUIDE.md) — step-by-step operation manual (install, three run modes, Docker, troubleshooting)
- [docs/USER_GUIDE_zh.md](docs/USER_GUIDE_zh.md) — 中文操作手册（安装 / 三种运行方式 / Docker / 常见问题排查）
- [docs/AWS_DEPLOYMENT.md](docs/AWS_DEPLOYMENT.md) — AWS deployment runbook (ECR + App Runner + GitHub OIDC)
- [docs/AWS_DEPLOYMENT_zh.md](docs/AWS_DEPLOYMENT_zh.md) — AWS 部署操作手册（中文，上云全流程）
- [docs/V2_STATUS.md](docs/V2_STATUS.md) — v2 upgrade status (pre-flight Q1–Q3 + phase-C measured numbers)
- [docs/INTERVIEW_NOTES.md](docs/INTERVIEW_NOTES.md) — interview talking points + explicit boundaries
- [docs/V2_GAP_ANALYSIS.md](docs/V2_GAP_ANALYSIS.md) — v1.0 vs upgrade-plan gap analysis (why phase 0 came first)
- [eval/README.md](eval/README.md) — golden-set, LLM-judge and 3-arm ablation how-to
