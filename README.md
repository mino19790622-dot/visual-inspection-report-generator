# Visual Inspection Report Generator

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
- **Deployment**: Docker (python:3.12-slim, runtime-only deps) + docker-compose

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
├── data/
│   ├── standards/           # 6 inspection standards (markdown)
│   └── test_images/         # Sample inspection images
├── reports/                 # Generated reports (timestamped, gitignored)
├── run_pipeline.py          # Linear pipeline: detect → VLM → RAG
├── run_agent.py             # LangGraph agent CLI
├── export_onnx.py           # Export PyTorch model to ONNX
├── compare_d1_d2.py         # PyTorch vs ONNX consistency check
└── requirements.txt
```

## Author

**Mino Zhang** — Computer Vision Engineer transitioning to AI Engineering
- GitHub: [@mino19790622-dot](https://github.com/mino19790622-dot)
- Background: 3 years CV algorithm engineering (PyTorch/YOLOX/TensorRT)
- MSc Computer Science candidate @ Maynooth University, Ireland
