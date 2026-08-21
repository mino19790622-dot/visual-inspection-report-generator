# app/vlm/client.py
"""VLM client — Qwen-VL via Alibaba Cloud DashScope (OpenAI-compatible API)."""

import base64
import io
import json
import cv2
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


class VLMClient:
    """Call Qwen-VL-Max to analyze inspection images with detection context."""

    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    MODEL = "qwen-vl-max"

    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=self.BASE_URL,
        )

    @staticmethod
    def _encode_image(image_path: str, max_size: int = 1024) -> str:
        """Read image, resize if too large, return base64 data URI."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        h, w = img.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        # Encode to JPEG in memory (smaller than raw base64)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise ValueError("Failed to encode image")
        data = base64.b64encode(buf).decode()
        return f"data:image/jpeg;base64,{data}"

    def _build_prompt(self, detection_result: dict) -> str:
        det_json = json.dumps(detection_result, ensure_ascii=False, indent=2)
        return f"""You are a visual inspection assistant. Analyze this image and produce a structured report.

## Detection Context
A YOLOv8m detector (trained on COCO, 80 classes) has already run on this image. Its results are provided below as ground truth for object counts and locations. Use these numbers directly — do not recount or second-guess them.

Detection results:
{det_json}

## Important Context
The detector was trained on standard camera-angle photos. The input image may come from aerial, surveillance, or industrial inspection angles — which may cause domain gap (missed detections or false positives). Use your own visual understanding to identify what the detector likely missed.

## Output Format (exactly these 4 sections, no extra text)

### 1. Scene Description
One paragraph describing the overall scene: environment type, time of day, weather, camera angle, and general purpose.

### 2. Object Inventory
List each object type detected, with count and confidence range from the detection JSON above. State these as facts from the detector, not your own observation.

### 3. Detector Gaps
Based on your own visual analysis of the image, list objects or anomalies that the detector likely missed or misclassified. For each gap, explain why (e.g., "small object at low resolution", "unusual viewing angle", "class not in COCO 80"). If no gaps, state "None identified."

### 4. Risk Assessment
Given this is a visual inspection context, note any safety risks, structural concerns, or anomalies that warrant human review. Be specific and actionable. If low risk, state so clearly."""

    def analyze(self, image_path: str, detection_result: dict) -> str:
        """Send image + detection JSON to VLM, return structured report text."""
        image_data = self._encode_image(image_path)
        prompt = self._build_prompt(detection_result)

        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            temperature=0,  # deterministic: inspection reports must be reproducible
            max_tokens=2000,
        )
        return response.choices[0].message.content
