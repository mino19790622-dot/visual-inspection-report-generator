# quick_test.py — D1 环境验证：图片进 → 画框图出
from ultralytics import YOLO

model = YOLO("yolov8m.pt")                            # 首次运行自动下载权重(~22MB)
results = model("https://ultralytics.com/images/bus.jpg")  # 官方自带测试图
results[0].save("out_bus.jpg")                        # 保存画框图
print(results[0].to_json())                          # 打印检测 JSON
                           
