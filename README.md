# Visual Inspection Report Generator

[![CI](https://github.com/mino19790622-dot/visual-inspection-report-generator/actions/workflows/ci.yml/badge.svg)](https://github.com/mino19790622-dot/visual-inspection-report-generator/actions/workflows/ci.yml)
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

153 unit/integration tests (coverage measured at 91%; the CI gate is set to 80%), zero network access required — the ONNX session, DashScope embeddings, and Qwen-VL calls are all mocked at the module boundary:

- `test_detection.py` — letterbox preprocessing, bbox decoding roundtrip, confidence filtering, NMS suppression/clipping
- `test_agent.py` — risk classification (explicit statement vs keyword fallback, negation safety), routing logic, and a **full LangGraph run** verifying adaptive re-detection and tool-calling grounded findings (`mode=off` reproduces v1.0)
- `test_rag.py` — chunking (header split / overlap / fragment filter) and retrieval against a real in-memory ChromaDB with deterministic hash embeddings
- `test_api.py` — FastAPI `TestClient`: schema contract, 415/500 error paths, mocked agent
- `test_exporter.py` — Markdown/JSON report content, graceful annotation failure
- `test_observability.py` — structured JSONL log writer (incl. unwritable-dir fallback)

```bash
pip install -r requirements-ci.txt
ruff check app tests
pytest --cov=app --cov-fail-under=80
```

CI runs the same suite on every push/PR (`.github/workflows/ci.yml`), and `regression.yml` re-runs the
deterministic grounded-generation tests on every push with zero API cost.

## Golden-Set Evaluation (MLOps quality gate)

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

### Golden-set quality gate

| Metric | Value | Notes |
|---|---|---|
| Overall quality score | **4.33 / 5.0** | threshold 3.7; eval run `35008709697` (agentic) |
| VLM tokens / 10 images | ~18.1k | consistent across arms (18.1–18.3k) |

> **Variance caveat.** These are single-pass runs. A second agentic pass produced different per-image
> findings (e.g. `large_building_aerial`: 3 findings in the gate run vs 1 in the ablation), and the
> `fixed` arm hit 1/10 parse failures while `agentic` hit 0/10. Treat per-image counts as noisy; the
> aggregate claims above are the ones we stand behind.

> **Judge threshold is provisional.** 3.7 is a starting gate, not a calibrated threshold. Judge
> credibility (incl. self-enhancement bias, judge + measured model both Qwen-family) is verified in
> phase B. Do not read "passed 3.7" as "validated for production."

## Cost & Latency

Token usage is measured from the API response and recorded per stage in `logs/inspect_spans.jsonl`.
Money is **derived** from `config/pricing.yaml` (a versioned price snapshot), never estimated in code.

| Metric | Value | Notes |
|---|---|---|
| VLM tokens / 10 images | ~18.1k | nearly identical across arms (18.1–18.3k) — grounding does not change the VLM call |
| Grounding tokens / 10 images | `off` 0 · `fixed` 21.3k · `agentic` 53.4k | agentic's multi-round tool loop is ~2.5× `fixed` |
| Cost in CNY | **TBD** | `config/pricing.yaml` rates are `null` → `known=False`; filled in phase A. Never rendered as 0.0 |
| Latency / stage | measured | detect / VLM / ground / export per request in the trace JSONL |

## Limitations

Stated up front — not as a disclaimer but as the boundary of what the numbers above mean:

- **Sample size n = 10.** Every measured number is directional, not a benchmark. There is no held-out test split and no stratified sampling.
- **Judge bias unverified.** The LLM judge (`qwen-turbo`) and the measured models (`qwen-plus` / `qwen-vl-max`) are the same family, so a self-enhancement bias cannot be ruled out. Scheduled for the phase-B credibility check, not resolved here.
- **Single-node, single-tenant, offline batch evaluation.** No orchestration/scheduling layer, no horizontal scaling, no multi-tenant isolation — the API serves one process.
- **Not deployed on Kubernetes.** Container orchestration stops at AWS ECS Fargate; there are no K8s manifests, autoscalers, or service mesh.
- **Cost is CNY and currently unfilled.** `config/pricing.yaml` ships with `null` rates, so cost reports `known=False`; no real monetary figure has been produced yet (phase A).
- **Retrieval quality is unmeasured.** No retrieval ground truth exists yet, so no recall@k / MRR is reported.

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

A per-stage trace (spans + tokens) is written to `logs/inspect_spans.jsonl`. `cost_rmb` stays `null` until the price snapshot is filled, and `top_k` is now the depth the grounding agent chose for this request (not a risk-level lookup). Cost is **derived** from `config/pricing.yaml` (a versioned price snapshot), not hard-coded in code; until rates are filled it reports `known=False` and is **never rendered as 0.0** — see `app/observability/pricing.py`.

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
- **Quality**: pytest 153 tests (91% measured coverage; CI gate at 80%) + LLM-as-judge golden-set eval (10 images, threshold-gated, manually triggered) + JSONL cost/latency observability on every `/inspect`

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
├── tests/                         # pytest suite (153 tests, ~91% coverage, no network)
│   ├── fakes.py                   # v2: ScriptedClient — fake LLM for grounding tests
│   ├── test_citations.py          # v2
│   ├── test_tools.py              # v2
│   ├── test_grounding.py          # v2
│   ├── test_spans.py              # v2
│   ├── test_log_redaction.py      # v2
│   ├── test_ablation.py           # v2
│   └── …                          # detection / vlm / rag / agent / api / exporter / observability
├── .github/workflows/
│   ├── ci.yml                     # ruff + pytest + coverage gate
│   ├── regression.yml             # v2: deterministic regression on every push (zero API cost)
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

**Mino Zhang** — AI Engineer building LLM-agent systems (tool calling, RAG, grounded citations) on a computer-vision foundation
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
