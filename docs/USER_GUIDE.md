# Visual Inspection Report Generator · Operation Manual

> A "run-it-and-ship-it" manual: from zero-install to three run modes, Docker deployment, and troubleshooting for known pitfalls.

## Table of Contents

- [0. What This Project Is](#0-what-this-project-is)
- [1. System Requirements](#1-system-requirements)
- [2. First-Time Setup](#2-first-time-setup)
- [3. Run Modes at a Glance](#3-run-modes-at-a-glance)
- [4. Mode A: Linear Pipeline — `run_pipeline.py`](#4-mode-a-linear-pipeline--run_pipelinepy)
- [5. Mode B: LangGraph Agent — `run_agent.py`](#5-mode-b-langgraph-agent--run_agentpy)
- [6. Mode C: FastAPI Service (local uvicorn)](#6-mode-c-fastapi-service-local-uvicorn)
- [7. Mode D: Docker Deployment](#7-mode-d-docker-deployment)
- [8. API Reference](#8-api-reference)
- [9. Report Output](#9-report-output)
- [10. CLI Parameter Tables](#10-cli-parameter-tables)
- [11. Troubleshooting](#11-troubleshooting)
- [12. Project Layout](#12-project-layout)

---

## 0. What This Project Is

Turn a single field photo into a risk-graded inspection report with cited standards:

```
Photo → YOLOv8 Detection → Qwen-VL Visual Reasoning → RAG Standards Retrieval → Markdown/JSON Report
                                                        ↑
                                              LangGraph decision orchestration
```

Typical use cases: construction sites, streets, workshops, ports, assembly lines — anywhere that needs "look at image + check against standards + produce report".

---

## 1. System Requirements

| Component | Minimum | Notes |
|-----------|---------|-------|
| Python | 3.10+ (3.12 recommended) | Older versions hit numpy/torch compatibility walls |
| OS | macOS / Linux | Verified on Intel macOS |
| RAM | 8 GB+ | Peak ~1–2 GB during VLM calls |
| Disk | 4 GB+ | Including venv and model weights |
| Docker (optional) | Docker Desktop or Colima | For containerised deployment |
| Network | Access to `dashscope.aliyuncs.com` | Both VLM and embeddings go through this |

API credentials: Alibaba Cloud DashScope `DASHSCOPE_API_KEY`. Create one at the [DashScope console](https://bailian.console.aliyun.com/) → API-Key page.

---

## 2. First-Time Setup

```bash
# 1) Clone
git clone https://github.com/mino19790622-dot/visual-inspection-report-generator.git
cd visual-inspection-report-generator

# 2) Create a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 3) Install dependencies (includes PyTorch / Ultralytics / ONNX)
pip install -r requirements.txt

# 4) Configure the API key (local only — never commit it)
echo "DASHSCOPE_API_KEY=sk-your-real-key" > .env
```

Verify the installation:

```bash
python -c "from app.detection.detector import YOLODetector; print('OK')"
```

> Tip: if you don't want to install PyTorch (only running the FastAPI service), use `pip install -r requirements.docker.txt` instead — the same slim dependency set used inside the Docker image.

---

## 3. Run Modes at a Glance

| Mode | Entry point | Best for | Speed |
|------|-------------|----------|-------|
| **A. Linear pipeline** | `run_pipeline.py` | Local end-to-end runs, debugging | Fastest, ~5–10s per image |
| **B. LangGraph agent** | `run_agent.py` | See agent decision logs, adaptive re-detection | Slightly slower (may re-detect) |
| **C. FastAPI service** | `uvicorn app.api.server:app` | Serve to frontends/systems, Swagger testing | Network IO + inference |

For a first try, start with Mode A.

---

## 4. Mode A: Linear Pipeline — `run_pipeline.py`

The straightforward path: detection → VLM → RAG → report files.

```bash
# Simplest (uses the bundled sample image)
python run_pipeline.py data/test_images/bus.jpg

# Use your own image
python run_pipeline.py /path/to/your/site_photo.jpg

# Custom options
python run_pipeline.py your_image.jpg \
  --conf 0.35 \            # detection confidence threshold
  --k 5 \                  # retrieve top-k standards
  --save-dir my_reports    # report output directory
```

Console output:

```
============================================================
  Visual Inspection Pipeline (D1-D4)
============================================================
[1/3] Running YOLOv8 detection...
  Detected 5 objects in 87ms
  Counts: {'person': 4, 'bus': 1}
[2/3] Calling Qwen-VL-Max for structured analysis...
  VLM report generated (1247 chars)
[3/3] Retrieving applicable inspection standards...
  Retrieved 3 relevant standards
============================================================
  VLM Inspection Report
============================================================
...
============================================================
  Applicable Standards (RAG Retrieved)
============================================================
--- [1] ISO 45001 ... (relevance: 0.86) ---
...
```

Report files land in `reports/` (see Section 9).

---

## 5. Mode B: LangGraph Agent — `run_agent.py`

Adds two layers of **real decision logic** on top of Mode A:

1. **Adaptive re-detection**: if the first pass detects 0 objects, automatically lower the confidence threshold (floor 0.10) and retry once.
2. **Risk-driven retrieval depth**: the VLM's risk level (low/medium/high) decides how many standards RAG retrieves (3/4/5). Decisions are **written into the report**.

```bash
python run_agent.py data/test_images/bus.jpg
python run_agent.py your.jpg --conf 0.3 --save-dir reports/
```

The run ends with a decision log, e.g.:

```
Decisions:
  - initial detection: 5 objects at conf=0.25
  - risk_level parsed as: high
  - retrieving 5 standards
```

Agent decisions also appear in the exported report under the "Agent Decisions" section, for later review.

---

## 6. Mode C: FastAPI Service (local uvicorn)

Turns the project into a REST API callable from frontends or scripts.

```bash
# Start
uvicorn app.api.server:app --host 0.0.0.0 --port 8000
```

Once running:
- Swagger UI (interactive testing): <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/health>

Test `/inspect`:

```bash
# Simplest: just the image
curl -F "image=@data/test_images/bus.jpg" http://127.0.0.1:8000/inspect

# Custom parameters
curl -F "image=@data/test_images/bus.jpg" \
     -F "conf=0.3" \
     -F "save=true" \
     http://127.0.0.1:8000/inspect | python -m json.tool
```

Returns JSON: `detection` (detection results) + `vlm_report` (visual report) + `risk_level` + `standards` (cited standards with relevance) + `agent_decisions` (decision log) + `saved_files` (paths of exported files).

> ⚠️ macOS note: use `127.0.0.1` instead of `localhost` — uvicorn binds IPv4 while curl may resolve `localhost` to IPv6 `::1`.

---

## 7. Mode D: Docker Deployment

For environment isolation, CI/CD, or sharing with a team.

### 7.1 Start Colima (macOS without Docker Desktop)

```bash
colima start
docker context use colima
```

### 7.2 One-command launch

```bash
# Make sure .env exists (with DASHSCOPE_API_KEY)
ls .env   # must exist

# Build image + start in background
docker compose up --build -d

# Watch logs (first build pulls python:3.12-slim, ~1–2 min)
docker compose logs -f api
# When you see "Application startup complete." you can Ctrl+C out of logs

# Health check
curl http://127.0.0.1:8000/health
# {"status":"ok"}

# Full inspection run
curl -F "image=@data/test_images/bus.jpg" http://127.0.0.1:8000/inspect
```

### 7.3 Containers & data

- `reports/` and `uploads/` are bind-mounted to the host — **reports and uploads appear in your local directory**
- The ChromaDB vector index lives in the named volume `chroma_index` — **survives restarts and rebuilds**
- Image size is about **1.78 GB** (build-only deps such as torch/ultralytics excluded)

### 7.4 Stop & clean up

```bash
docker compose down              # stop service, keep volume
docker compose down -v           # also remove volume (loses ChromaDB index)
```

### 7.5 After a machine reboot

```bash
colima start                     # Colima does not auto-start like Docker Desktop
docker compose up -d             # bring the container back
```

---

## 8. API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe, returns `{"status":"ok"}` |
| `GET` | `/standards` | List standards currently indexed in the RAG knowledge base |
| `POST` | `/inspect` | Upload an image, run the agent, return the full report JSON |
| `GET` | `/docs` | Swagger UI (built into FastAPI) |

`POST /inspect` parameters (form-data):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | file | required | jpg/png/webp/bmp |
| `conf` | float | 0.25 | Detection confidence threshold |
| `save` | bool | true | Whether to export report files to disk |

Allowed file extensions: `.jpg .jpeg .png .webp .bmp`. Anything else returns 415.

---

## 9. Report Output

Every run produces three timestamped files (older runs of the same image are kept, not overwritten):

```
reports/
├── bus_20260821_114832.md      # human-readable
├── bus_20260821_114832.json    # machine-readable
└── bus_20260821_114832_det.jpg # annotated image
```

**`*.md` structure**:

```markdown
# Visual Inspection Report

- Image: data/test_images/bus.jpg
- Generated: 2026-08-21 11:48:32
- Detector: yolov8m (ONNX, 87ms, 5 objects)

## VLM Analysis
> Scene description ...
> Inventory ...
> Detector Gaps ...
> Risk Assessment: high risk ...

## Applicable Standards
1. ISO 45001 — ...
2. BS EN 1992 — ...
3. EN 13134 — ...

## Agent Decisions
- initial detection: 5 objects at conf=0.25
- risk_level parsed as: high
- retrieving 5 standards
```

**`*.json`**: structured fields with the same keys as the API response (`detection` / `vlm_report` / `risk_level` / `standards` / `agent_decisions`).

**`*_det.jpg`**: original image + red bounding boxes + class labels + confidence scores.

---

## 10. CLI Parameter Tables

### `run_pipeline.py`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image` (positional) | `data/test_images/bus.jpg` | Input image path |
| `--onnx` | `yolov8m.onnx` | ONNX model path |
| `--conf` | `0.25` | Detection confidence threshold |
| `--k` | `3` | Number of standards to retrieve |
| `--save-dir` | `reports` | Report output directory |
| `--no-save` | off | Print to console only, don't write files |

### `run_agent.py`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image` | `data/test_images/bus.jpg` | Input image path |
| `--conf` | `0.25` | Detection confidence threshold |
| `--save-dir` | `reports` | Report output directory |
| `--no-save` | off | Print to console only |

### `uvicorn app.api.server:app`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--host` | `127.0.0.1` | Listen address (use `0.0.0.0` inside a container) |
| `--port` | `8000` | Port |
| `--reload` | off | Dev-mode hot reload |

---

## 11. Troubleshooting

### 11.1 `ProxyError` / `TunnelError` even though no proxy is set

Leftover proxy environment variables (corporate VPN / Clash not cleaned up). httpx/requests picks them up automatically.

```bash
env | grep -i proxy       # inspect
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
# or bypass once
curl --noproxy '*' http://127.0.0.1:8000/health
```

**Permanent fix**: remove or comment out these exports in `~/.zshrc` / `~/.bash_profile`.

### 11.2 `curl localhost:8000` refused while the server is clearly running

One of two causes:

- uvicorn binds `127.0.0.1` (IPv4) but `localhost` resolves to `::1` (IPv6) → **use `127.0.0.1`**
- The proxy variables above hijack the request → add `--noproxy '*'`

### 11.3 ChromaDB shows `relevance: 0.01` or negative values

The collection was created without an explicit distance metric (defaults to L2). **Changing the metric requires deleting and rebuilding the store**:

```bash
rm -rf .chroma_db
# rerun any command; the retriever rebuilds with cosine
```

The code already creates collections with `metadata={"hnsw:space": "cosine"}`, so this only bites if you **manually edited the retriever or switched embedding dimensions**.

### 11.4 VLM gives different conclusions for the same image across runs

Early versions used `temperature=0.2`, which caused drift ("no hazardous" ↔ "moderate"). The project now pins `temperature=0`. For any reproducibility issue, **don't touch the temperature**; check first:

- Did the prompt change? (template lives in `app/vlm/client.py`)
- Did the model change? (default `qwen-vl-max`)

### 11.5 Large images crash the VLM call

`VLMClient` already auto-resizes images (shrink to max 1024px before base64). **If you disabled that logic**, restore it:

```python
# app/vlm/client.py
img = resize_if_needed(img, max_side=1024)
```

### 11.6 Docker pull fails with `docker-credential-desktop not found`

Leftover Docker Desktop config:

```bash
# Edit ~/.docker/config.json
# Remove the "credsStore" and "credHelpers" keys entirely
# Keep only "experimental": "enabled" (if present)
```

### 11.7 `docker ps` still can't find the daemon after Colima starts

```bash
colima start
docker context use colima
docker ps
```

### 11.8 `pip install torch` fails halfway

On Intel Macs, don't use the latest torch. **Stick to the versions pinned in `requirements.txt`**:

```text
torch==2.2.2
numpy==1.26.4
opencv-python<5
```

If these three drift out of alignment you hit the numpy 2.x ABI conflict.

### 11.9 `risk_level` is always "n/a" in reports

Either the VLM output lacks a proper "Risk Assessment" section, or the parser fell through to the keyword fallback. Check the report `.md` — the expected format is:

```
> Risk Assessment: (high|moderate|low) risk ...
```

If the VLM output format changed, adjust the regex in `parse_risk_level` in `app/agent/graph.py`.

---

## 12. Project Layout

```
visual-inspection-report-generator/
├── app/
│   ├── detection/detector.py     # Hand-written YOLODetector (ONNX, letterbox, NMS)
│   ├── vlm/client.py             # Qwen-VL-Max client + image auto-resize
│   ├── rag/retriever.py          # ChromaDB collection (cosine)
│   ├── agent/graph.py            # LangGraph StateGraph
│   ├── reporting/exporter.py     # Markdown + JSON + annotated image export
│   └── api/server.py             # FastAPI
├── data/
│   ├── standards/                # 6 inspection standards (markdown)
│   └── test_images/              # Sample images
├── docs/                         # Documentation
│   ├── USER_GUIDE.md             # ← this file (English)
│   └── USER_GUIDE_zh.md          # Chinese manual
├── reports/                      # Report output (gitignored)
├── uploads/                      # API upload staging
├── .chroma_db/                   # Vector index (gitignored)
├── run_pipeline.py               # Mode A
├── run_agent.py                  # Mode B
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements.docker.txt       # Slim image dependencies
└── README.md
```

---

## Appendix: 30-Second Quickstart

```bash
git clone https://github.com/mino19790622-dot/visual-inspection-report-generator.git
cd visual-inspection-report-generator
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "DASHSCOPE_API_KEY=sk-your-key" > .env
python run_pipeline.py data/test_images/bus.jpg
open reports/   # macOS: view the freshly generated report
```

Try the agent:

```bash
python run_agent.py data/test_images/bus.jpg
```

Run it as a service:

```bash
uvicorn app.api.server:app --port 8000 &
open http://127.0.0.1:8000/docs
```

Or go straight to Docker:

```bash
docker compose up --build -d
sleep 30 && curl http://127.0.0.1:8000/health
```
