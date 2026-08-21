
# app/detection/detector.py
import time

import cv2
import numpy as np
import onnxruntime as ort


class YOLODetector:
    """YOLOv8 ONNX detector — self-contained preprocessing/postprocessing."""

    def __init__(self, onnx_path: str, classes_path: str = "classes.txt",
                 conf_thres: float = 0.35, iou_thres: float = 0.5):
        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        with open(classes_path) as f:
            self.class_names = [line.strip() for line in f]
        self.input_size = self.session.get_inputs()[0].shape[-1]  # 640

    # ---------- 预处理：letterbox ----------
    def _letterbox(self, img: np.ndarray):
        h, w = img.shape[:2]
        scale = self.input_size / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (nw, nh))
        pad_top = (self.input_size - nh) // 2
        pad_left = (self.input_size - nw) // 2
        canvas = np.full((self.input_size, self.input_size, 3),
                         114, dtype=np.uint8)
        canvas[pad_top:pad_top + nh, pad_left:pad_left + nw] = resized
        return canvas, scale, (pad_left, pad_top)

    # ---------- 后处理：decode + NMS ----------
    def _postprocess(self, output: np.ndarray, scale, pad,
                     orig_shape):
        pred = output[0]              # (84, 8400) = 4 bbox + 80 classes
        boxes_cxcywh = pred[:4, :].T
        class_ids = pred[4:, :].argmax(0)
        confs = pred[4:, :].max(0)

        mask = confs > self.conf_thres
        boxes_cxcywh, class_ids, confs = (
            boxes_cxcywh[mask], class_ids[mask], confs[mask])

        # cxcywh -> xyxy（还原图 letterbox）
        boxes = np.zeros((len(boxes_cxcywh), 4))
        boxes[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        boxes[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        boxes[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
        boxes[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / scale

        # NMS
        keep = cv2.dnn.NMSBoxes(
            boxes.tolist(), confs.tolist(),
            self.conf_thres, self.iou_thres)
        keep = keep.flatten() if len(keep) else []

        h, w = orig_shape[:2]
        detections = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i].clip(0, [w, h, w, h])
            detections.append({
                "class": self.class_names[int(class_ids[i])],
                "confidence": round(float(confs[i]), 3),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
            })
        return detections

    # ---------- 主入口 ----------
    def detect(self, image_path: str) -> dict:
        img = cv2.imread(image_path)
        canvas, scale, pad = self._letterbox(img)
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[None]          # (1,3,640,640)

        t0 = time.perf_counter()
        output = self.session.run(None, {self.input_name: blob})[0]
        infer_ms = (time.perf_counter() - t0) * 1000

        detections = self._postprocess(output, scale, pad, img.shape)
        counts = {}
        for d in detections:
            counts[d["class"]] = counts.get(d["class"], 0) + 1
        return {
            "image": image_path,
            "inference_ms": round(infer_ms, 1),
            "counts": counts,
            "detections": detections,
        }

    # ---------- 可视化 ----------
    def draw(self, image_path: str, result: dict, out_path: str):
        img = cv2.imread(image_path)
        for d in result["detections"]:
            x1, y1, x2, y2 = d["bbox"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(img, f'{d["class"]} {d["confidence"]:.2f}',
                        (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 200, 0), 1)
        cv2.imwrite(out_path, img)

