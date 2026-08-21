
# run_detection.py
import json
from app.detection.detector import YOLODetector

det = YOLODetector("yolov8s.onnx")
result = det.detect("data/test_images/bus.jpg")
print(json.dumps(result, indent=2, ensure_ascii=False))
det.draw("data/test_images/bus.jpg", result, "out_onnx.jpg")

