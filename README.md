# Visual Inspection Report Generator

AI-powered visual inspection system that combines object detection with vision-language model analysis to generate structured inspection reports.

## Architecture

```
Image Input → YOLOv8 Detection → VLM Analysis → Structured Report
```

| Layer | Technology | Status |
|-------|-----------|--------|
| Detection | YOLOv8m (Ultralytics + ONNX Runtime) | ✅ Complete |
| VLM Analysis | Qwen2.5-VL via OpenRouter API | 🚧 In Progress |
| RAG Standards Retrieval | LangChain + Vector DB | 📋 Planned |
| Agent Orchestration | LangGraph | 📋 Planned |
| Report Generation | Structured JSON + PDF | 📋 Planned |
| API Layer | FastAPI + Docker | 📋 Planned |

## Current Features

- **D1: PyTorch Inference** — YOLOv8m detection with 237ms latency on Intel Mac
- **D2: ONNX Pipeline** — Exported to ONNX format, custom inference pipeline, D1/D2 consistency validation
- Supports 80 COCO classes with confidence threshold tuning (default: 0.25)

## Quick Start

```bash
# Clone
git clone https://github.com/mino19790622-dot/visual-inspection-report-generator.git
cd visual-inspection-report-generator

# Setup
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run detection
python run_detection.py --model yolov8m.pt --source data/test_images/bus.jpg

# Compare PyTorch vs ONNX
python compare_d1_d2.py
```

## Tech Stack

- **Python 3.12** | PyTorch 2.2.2 | Ultralytics 8.3 | ONNX Runtime 1.16
- **Models**: YOLOv8m (53M params) for detection, Qwen2.5-VL for visual reasoning
- **Deployment**: FastAPI + Docker (planned)

## Project Structure

```
├── app/
│   └── detection/
│       └── detector.py      # YOLODetector class (PyTorch + ONNX)
├── data/test_images/         # Sample inspection images
├── export_onnx.py            # Export PyTorch model to ONNX
├── run_detection.py          # Run detection on images
├── compare_d1_d2.py          # Compare PyTorch vs ONNX outputs
├── quick_test.py             # Quick environment validation
└── requirements.txt
```

## Author

**Mino Zhang** — Computer Vision Engineer transitioning to AI Engineering
- GitHub: [@mino19790622-dot](https://github.com/mino19790622-dot)
- Background: 3 years CV algorithm engineering (PyTorch/YOLOX/TensorRT)
- MSc Computer Science candidate @ Maynooth University, Ireland
