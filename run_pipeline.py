# run_pipeline.py
"""Full pipeline: YOLOv8 detection -> VLM analysis -> RAG standards retrieval -> report export."""

import argparse
import sys
from app.detection.detector import YOLODetector
from app.vlm.client import VLMClient
from app.rag.retriever import StandardsRetriever
from app.reporting.exporter import ReportExporter


def run(image_path: str, onnx_path: str = "yolov8m.onnx",
        classes_path: str = "classes.txt", conf_thres: float = 0.25,
        top_k: int = 3, save: bool = True, save_dir: str = "reports"):
    print(f"\n{'='*60}")
    print(f"  Visual Inspection Pipeline (D1-D4)")
    print(f"{'='*60}")

    # --- Step 1: Detection ---
    print(f"\n[1/3] Running YOLOv8 detection...")
    detector = YOLODetector(onnx_path, classes_path, conf_thres=conf_thres)
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
    standards = retriever.retrieve(report, k=top_k)
    print(f"  Retrieved {len(standards)} relevant standards")

    # --- Output to console ---
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

    # --- Export report ---
    saved = {}
    if save:
        print(f"{'='*60}")
        print("  Exporting report...")
        exporter = ReportExporter(save_dir)
        saved = exporter.export(image_path, det_result, report, standards,
                                draw_fn=detector.draw)
        for kind, path in saved.items():
            print(f"  [{kind}] {path}")

    print(f"\n{'='*60}")
    print("  Pipeline complete.")
    print(f"{'='*60}\n")

    return {"report": report, "standards": standards, "saved": saved}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visual inspection: detection + VLM analysis + standards retrieval")
    parser.add_argument("image", nargs="?", default="data/test_images/bus.jpg",
                        help="Path to input image (default: bus.jpg)")
    parser.add_argument("--onnx", default="yolov8m.onnx",
                        help="ONNX model path (default: yolov8m.onnx)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Detection confidence threshold (default: 0.25)")
    parser.add_argument("--k", type=int, default=3,
                        help="Number of standards to retrieve (default: 3)")
    parser.add_argument("--save-dir", default="reports",
                        help="Directory for exported reports (default: reports/)")
    parser.add_argument("--no-save", action="store_true",
                        help="Print to console only, do not export files")
    args = parser.parse_args()

    run(args.image, onnx_path=args.onnx, conf_thres=args.conf,
        top_k=args.k, save=not args.no_save, save_dir=args.save_dir)
