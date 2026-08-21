
# export_onnx.py
from ultralytics import YOLO

model = YOLO("yolov8m.pt")
model.export(format="onnx", imgsz=640, opset=13, simplify=True)
# 生成 yolov8s.onnx（约 43MB）
# 同时导出类别名：保存在 model.names，写进 classes.txt 备用
with open("classes.txt", "w") as f:
    f.write("\n".join(model.names.values()))

