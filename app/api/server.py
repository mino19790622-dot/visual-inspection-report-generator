# app/api/server.py
"""FastAPI service exposing the LangGraph inspection agent.

Endpoints:
    GET  /health     — liveness probe
    GET  /standards  — list indexed inspection standards
    POST /inspect    — upload an image, run the agent, return the inspection report

Run:
    uvicorn app.api.server:app --host 0.0.0.0 --port 8000
    curl -F "image=@data/test_images/bus.jpg" http://localhost:8000/inspect
"""

import os
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.agent.graph import InspectionAgent, _get_retriever

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
UPLOAD_DIR = "uploads"

app = FastAPI(
    title="Visual Inspection Report Generator API",
    description="YOLOv8 detection + Qwen-VL analysis + RAG standards, "
                "orchestrated by a LangGraph agent",
    version="0.1.0",
)
agent = InspectionAgent()


def _save_upload(image: UploadFile) -> str:
    ext = os.path.splitext(image.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex[:12]}{ext}")
    with open(path, "wb") as f:
        f.write(image.file.read())
    return path


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/standards")
def list_standards():
    """List the standards currently indexed in the RAG knowledge base."""
    retriever = _get_retriever()
    metas = retriever.collection.get(include=["metadatas"])["metadatas"]
    seen, items = set(), []
    for m in metas:
        key = m["source"]
        if key not in seen:
            seen.add(key)
            items.append({"standard": m["standard"], "source": key})
    return {"count": len(items), "standards": items}


@app.post("/inspect")
def inspect(image: UploadFile = File(...),
            conf: float = Form(0.25),
            save: bool = Form(True)):
    """Run the full inspection agent on an uploaded image.

    Returns detection results, VLM report, risk level, matched standards,
    and the agent's decision log.
    """
    path = _save_upload(image)
    try:
        # sync def endpoint -> FastAPI runs it in a threadpool, no event-loop block
        state = agent.run(path, conf_thres=conf, save=save)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent failed: {e}")

    return {
        "image": image.filename,
        "detection": {
            "object_count": sum(state["det_result"]["counts"].values()),
            "counts": state["det_result"]["counts"],
            "inference_ms": state["det_result"]["inference_ms"],
            "detections": state["det_result"]["detections"],
        },
        "risk_level": state.get("risk_level"),
        "vlm_report": state.get("vlm_report"),
        "standards": [
            {
                "standard": s["standard"],
                "source": s["source"],
                "relevance": round(1 - s["distance"], 3),
                "excerpt": s["text"][:300],
            }
            for s in state.get("standards", [])
        ],
        "agent_decisions": state.get("decisions", []),
        "saved_files": state.get("saved", {}),
    }
