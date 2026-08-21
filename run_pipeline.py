# run_pipeline.py
"""Full pipeline: YOLOv8 detection -> VLM analysis -> RAG standards retrieval."""

import sys
from app.detection.detector import YOLODetector
from app.vlm.client import VLMClient
from app.rag.retriever import StandardsRetriever


def run(image_path: str, onnx_path: str = "yolov8m.onnx",
        classes_path: str = "classes.txt"):
    print(f"\n{'='*60}")
    print(f"  Visual Inspection Pipeline (D1-D4)")
    print(f"{'='*60}")

    # --- Step 1: Detection ---
    print(f"\n[1/3] Running YOLOv8 detection...")
    detector = YOLODetector(onnx_path, classes_path, conf_thres=0.25)
    det_result = detector.detect(image_path)
    print(f"  Detected {sum(det_result['counts'].values())} objects in "
          f"{det_result['inference_ms']}ms")
    print(f"  Counts: {det_result['counts']}")

    # --- Step 2: VLM Analysis ---
    print(f"\n[2/3] Calling Qwen-VL-Max for structured analysis...")
    vlm = VLMClient()
    report = vlm.analyze(image_path, det_result)
    print(f"  VLM report generated ({len(report)} chars)")

    # --- Step 3: RAG Standards Retrieval ---
    print(f"\n[3/3] Retrieving applicable inspection standards...")
    retriever = StandardsRetriever()
    # Use VLM report as query to find relevant standards
    standards = retriever.retrieve(report, k=3)
    print(f"  Retrieved {len(standards)} relevant standards")

    # --- Output ---
    print(f"\n{'='*60}")
    print("  VLM Inspection Report")
    print(f"{'='*60}\n")
    print(report)

    print(f"\n{'='*60}")
    print("  Applicable Standards (RAG Retrieved)")
    print(f"{'='*60}\n")
    for i, s in enumerate(standards, 1):
        print(f"--- [{i}] {s['standard']} (relevance: {1 - s['distance']:.2f}) ---")
        print(f"Source: {s['source']}")
        print(f"{s['text'][:500]}")
        print()

    print(f"{'='*60}")
    print("  Pipeline complete.")
    print(f"{'='*60}\n")

    return {"report": report, "standards": standards}


if __name__ == "__main__":
    image = sys.argv[1] if len(sys.argv) > 1 else "data/test_images/bus.jpg"
    run(image)
