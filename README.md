# Visual Inspection Report Generator

[![CI](https://github.com/mino19790622-dot/visual-inspection-report-generator/actions/workflows/ci.yml/badge.svg)](https://github.com/mino19790622-dot/visual-inspection-report-generator/actions/workflows/ci.yml)

AI-powered visual inspection system: object detection → VLM visual reasoning → RAG standards retrieval → structured inspection report.

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
  - *Risk-based retrieval depth*: risk level parsed from the VLM's Risk Assessment section (negation-safe) controls how many standards are retrieved (low=3, medium=4, high=5)
  - All decisions logged in state and exported in every report
- **API Layer (D6)**: FastAPI service — `POST /inspect` (image upload → full JSON report), `GET /standards`, `GET /health`

## Verified Behaviors

| Scenario | Detection | VLM Output | RAG Retrieval |
|----------|-----------|------------|---------------|
| Urban bus scene | 1 bus + 4 persons ✅ | Correct inventory, no gaps | Public transport + pedestrian standards |
| Group photo (5 people) | 4 persons (missed 1) | Caught the missed 5th person | Pedestrian safety standard |
| Aerial construction site | 3 false positives | Identified all as domain-gap errors | Construction safety + hazard thresholds |

## Testing

58 unit/integration tests (85% coverage), zero network access required — the ONNX session, DashScope embeddings, and Qwen-VL calls are all mocked at the module boundary:

- `test_detection.py` — letterbox preprocessing, bbox decoding roundtrip, confidence filtering, NMS suppression/clipping
- `test_agent.py` — risk classification (explicit statement vs keyword fallback, negation safety), routing logic, and a **full LangGraph run** verifying adaptive re-detection and risk-based retrieval depth
- `test_rag.py` — chunking (header split / overlap / fragment filter) and retrieval against a real in-memory ChromaDB with deterministic hash embeddings
- `test_api.py` — FastAPI `TestClient`: schema contract, 415/500 error paths, mocked agent
- `test_exporter.py` — Markdown/JSON report content, graceful annotation failure
- `test_observability.py` — structured JSONL log writer (incl. unwritable-dir fallback)

```bash
pip install -r requirements-ci.txt
ruff check app tests
pytest --cov=app --cov-fail-under=80
```

CI runs the same suite on every push/PR (`.github/workflows/ci.yml`).

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

A separate `eval.yml` workflow lets you run this on demand from the Actions tab (avoids the LLM cost on every push). Baseline run on 10 images: **overall 4.28 / 5.0** (Feb 2026).

## Observability

Every `/inspect` call appends one JSON line to `logs/inspect.jsonl` (path overridable via `INSPECT_LOG_DIR`):

```json
{"ts":"2026-08-21T...","image":"abc.jpg","risk_level":"high",
 "detection":{"object_count":3,"inference_ms":47},
 "vlm":{"prompt_tokens":1200,"completion_tokens":350,"total_tokens":1550,
        "latency_ms":1800,"cost_rmb":0.031},
 "retrieval":{"top_k":5,"standards_count":5},
 "total_latency_ms":2400,"saved":["report","annotated"]}
```

Cost is estimated at ¥0.02 / 1K tokens (qwen-vl-max public price); update `_RMB_PER_1K_TOKENS` in `app/vlm/client.py` if pricing changes.

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

# Or run the LangGraph agent (adds adaptive re-detection + risk-based retrieval)
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

The Docker image is cloud-ready and **deployed live on AWS ECS Fargate** (eu-west-1) — the same image also runs on AWS App Runner (fully managed, public HTTPS URL, autoscaling; blocked only by new-account activation lag). This is also directly relevant to the AWS SAA-C03 certification path.

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
- **Quality**: pytest 58 tests (85% coverage, CI-gated) + LLM-as-judge golden-set eval (10 images, threshold-gated, manually triggered) + JSONL cost/latency observability on every `/inspect`

## Project Structure

```
├── app/
│   ├── detection/
│   │   └── detector.py      # YOLODetector (ONNX, letterbox + NMS)
│   ├── vlm/
│   │   └── client.py        # VLMClient (Qwen-VL-Max, image auto-resize)
│   └── rag/
│       └── retriever.py     # StandardsRetriever (embed + ChromaDB)
│   └── reporting/
│       └── exporter.py      # ReportExporter (Markdown + JSON + annotated image)
│   └── agent/
│       └── graph.py         # LangGraph StateGraph (adaptive retry + risk routing)
│   └── api/
│       └── server.py        # FastAPI: POST /inspect, GET /standards, GET /health
│   └── observability.py     # Structured JSONL logging (latency / tokens / cost)
├── eval/                    # Golden-set + LLM-as-judge evaluation
│   ├── golden_set/golden_set.json
│   ├── judge.py             # qwen-turbo judge scoring 4 dimensions
│   └── run_eval.py          # Eval runner (deterministic + judge, threshold gate)
├── data/
│   ├── standards/           # 6 inspection standards (markdown)
│   └── test_images/         # Sample inspection images
├── reports/                 # Generated reports (timestamped, gitignored)
├── run_pipeline.py          # Linear pipeline: detect → VLM → RAG
├── run_agent.py             # LangGraph agent CLI
├── export_onnx.py           # Export PyTorch model to ONNX
├── compare_d1_d2.py         # PyTorch vs ONNX consistency check
├── docs/
│   ├── USER_GUIDE.md           # Operation manual (English)
│   ├── USER_GUIDE_zh.md        # Operation manual (Chinese)
│   ├── AWS_DEPLOYMENT.md       # AWS deployment runbook (English)
│   └── AWS_DEPLOYMENT_zh.md    # AWS deployment runbook (Chinese)
├── deploy/
│   ├── apprunner.yaml          # AWS App Runner service definition
│   ├── ecs-task-definition.json  # AWS ECS Fargate task definition
│   ├── ecs-execution-trust.json  # ECS execution role trust policy
│   ├── ecs-execution-secrets-policy.json  # ECS secret-read policy
│   └── gh-oidc-*.json          # GitHub OIDC trust + permission policies
├── scripts/
│   └── deploy.sh               # Build + push to ECR + deploy (local)
├── tests/                      # pytest suite (58 tests, 85% coverage, no network)
├── .github/workflows/
│   ├── ci.yml                  # CI: ruff + pytest + coverage on push/PR
│   ├── deploy.yml              # CI: OIDC → ECR → App Runner on push to main
│   └── eval.yml                 # Manual: golden-set LLM-as-judge evaluation
├── Dockerfile / docker-compose.yml / .dockerignore
└── requirements.txt / requirements.docker.txt / requirements-ci.txt
```

## Author

**Mino Zhang** — Computer Vision Engineer transitioning to AI Engineering
- GitHub: [@mino19790622-dot](https://github.com/mino19790622-dot)
- Background: 3 years CV algorithm engineering (PyTorch/YOLOX/TensorRT)
- MSc Computer Science candidate @ Maynooth University, Ireland

## Documentation

- [docs/USER_GUIDE.md](docs/USER_GUIDE.md) — step-by-step operation manual (install, three run modes, Docker, troubleshooting)
- [docs/USER_GUIDE_zh.md](docs/USER_GUIDE_zh.md) — 中文操作手册（安装 / 三种运行方式 / Docker / 常见问题排查）
- [docs/AWS_DEPLOYMENT.md](docs/AWS_DEPLOYMENT.md) — AWS deployment runbook (ECR + App Runner + GitHub OIDC)
- [docs/AWS_DEPLOYMENT_zh.md](docs/AWS_DEPLOYMENT_zh.md) — AWS 部署操作手册（中文，上云全流程）
