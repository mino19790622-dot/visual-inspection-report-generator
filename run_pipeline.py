# run_pipeline.py
"""D3 pipeline: YOLOv8 detection -> VLM analysis -> structured report."""

import sys
import json
from app.detection.detector import YOLODetector
from app.vlm.client import VLMClient


def run(image_path: str, onnx_path: str = "yolov8m.onnx",
        classes_path: str = "classes.txt"):
    print(f"\n{'='*60}")
    print(f"  Visual Inspection Pipeline")
    print(f"{'='*60}")

    # --- Step 1: Detection ---
    print(f"\n[1/2] Running YOLOv8 detection...")
    detector = YOLODetector(onnx_path, classes_path, conf_thres=0.25)
    det_result = detector.detect(image_path)
    print(f"  Detected {sum(det_result['counts'].values())} objects in "
          f"{det_result['inference_ms']}ms")
    print(f"  Counts: {det_result['counts']}")

    # --- Step 2: VLM Analysis ---
    print(f"\n[2/2] Calling Qwen-VL-Max for structured analysis...")
    vlm = VLMClient()
    report = vlm.analyze(image_path, det_result)

    # --- Output ---
    print(f"\n{'='*60}")
    print("  VLM Inspection Report")
    print(f"{'='*60}\n")
    print(report)
    print(f"\n{'='*60}")
    print("  Pipeline complete.")
    print(f"{'='*60}\n")

    return report


if __name__ == "__main__":
    image = sys.argv[1] if len(sys.argv) > 1 else "data/test_images/bus.jpg"
    run(image)
