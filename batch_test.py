from pathlib import Path
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
for img in sorted(Path("data/test_images").glob("*.jpg")):
    results = model(str(img), conf=0.25)   # 航拍/监控图上预训练模型偏弱，阈值降到0.25
    results[0].save(f"out_{img.name}")
    print(img.name, "->", results[0].to_json()[:200])
