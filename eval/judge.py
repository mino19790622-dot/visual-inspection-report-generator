"""LLM-as-judge for the inspection agent's VLM report.

Scores a VLM report against a per-image rubric on 4 dimensions (1–5 each):
  1. scene_id        — did the report correctly identify the scene?
  2. safety          — did it flag appropriate safety concerns?
  3. domain_awareness — did it note the COCO detector's domain gap?
  4. structure       — does it follow the required 4-section format?

Uses qwen-turbo via DashScope (text-only, cheap).
"""
import base64
import json
import os
import time

import cv2
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_JUDGE_MODEL = "qwen-turbo"
_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=_BASE_URL,
        )
    return _client


def _image_data_uri(image_path: str, max_size: int = 512) -> str:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read {image_path}")
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        s = max_size / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise ValueError("encode failed")
    return f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"


_JUDGE_PROMPT = """You are an expert evaluator for a visual inspection system. You are given:
1. The original image.
2. The "ground-truth" rubric describing what a good report must cover.
3. The actual VLM report produced by the system.

Score the report on FOUR dimensions, each 1–5 (integers):
- scene_id: Does the report correctly identify the scene? (1=wrong, 5=correct and detailed)
- safety: Does it flag the appropriate safety concerns from the rubric? (1=none, 5=all)
- domain_awareness: Does it acknowledge the detector's domain gap (COCO trained on standard
  camera-angle photos; the input may be aerial / unusual)? (1=no, 5=explicitly discussed)
- structure: Does it follow the 4-section format (Scene Description / Object Inventory /
  Detector Gaps / Risk Assessment) with the exact section headers? (1=missing sections,
  5=perfect structure)

Return ONLY a JSON object, no prose, no markdown fences:
{{"scene_id": <int>, "safety": <int>, "domain_awareness": <int>, "structure": <int>, "reason": "<one sentence>"}}

## Rubric (ground truth)
{rubric}

## VLM Report
{report}
"""


def judge(image_path: str, report: str, rubric: str,
          max_retries: int = 2) -> dict:
    """Run the judge LLM with one retry on empty/invalid response.

    Returns dict with 4 scores (1-5) + reason.
    """
    img_uri = _image_data_uri(image_path)
    prompt = _JUDGE_PROMPT.format(rubric=rubric, report=report)
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = _get_client().chat.completions.create(
                model=_JUDGE_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": img_uri}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                temperature=0,
                max_tokens=400,
            )
            text = resp.choices[0].message.content.strip()
            if not text:
                raise ValueError("empty judge response")
            if text.startswith("```"):
                # strip leading ``` or ```json and trailing ```
                text = text.strip("`")
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # be lenient: find first {...} block
                start, end = text.find("{"), text.rfind("}") + 1
                if start >= 0 and end > start:
                    return json.loads(text[start:end])
                raise ValueError(f"no JSON object in: {text[:80]!r}") from None
        except Exception as e:  # network, empty content, or JSON parse error
            last_err = e
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            break
    raise RuntimeError(f"judge failed after {max_retries+1} attempts: {last_err}")
