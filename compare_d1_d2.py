"""比对 D1(官方API) 与 D2(手写ONNX) 的检测结果"""
import json
from pathlib import Path
from ultralytics import YOLO
from app.detection.detector import YOLODetector

IMG_DIR = Path("data/test_images")
images = sorted(IMG_DIR.glob("*.jpg"))[:10]   # 先比10张

api_model = YOLO("yolov8m.pt")
onnx_det = YOLODetector("yolov8m.onnx")

def api_result(img_path):
    """官方API结果 → 统一格式"""
    r = api_model(str(img_path), conf=0.25)[0]
    dets = []
    for b in r.boxes:
        x1, y1, x2, y2 = [round(float(v)) for v in b.xyxy[0]]
        dets.append({"class": api_model.names[int(b.cls)],
                     "confidence": round(float(b.conf), 3),
                     "bbox": [x1, y1, x2, y2]})
    return dets

def iou(a, b):
    """两个框的重叠度"""
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ax1 != bx1, ay1, by1))
    inter = ix * iy
    return inter / (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter

for img in images:
    d1 = api_result(img)
    d2 = onnx_det.detect(str(img))["detections"]

    # 逐个 D1 检测，在 D2 里找同类且 IoU>0.9 的匹配
    matched, diffs = 0, []
    for a in d1:
        for b in d2:
            if b["class"] == a["class"] and iou(a["bbox"], b["bbox"]) > 0.9:
                matched += 1
                diffs.append(abs(a["confidence"] - b["confidence"]))
                break

    status = "✅" if matched == len(d1) == len(d2) else "⚠️"
    print(f"{status} {img.name}: D1={len(d1)}个 D2={len(d2)}个 匹配={matched}"
          + (f" 平均conf差={sum(diffs)/len(diffs):.3f}" if diffs else ""))

print("\n判定：全部 ✅ = D2 通关；个别 ⚠️ 看是边缘小框差异还是系统性偏差")
