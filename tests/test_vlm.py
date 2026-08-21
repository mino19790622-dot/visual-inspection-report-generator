# tests/test_vlm.py
"""Unit tests for VLMClient prompt construction and image encoding (no API calls)."""

import base64

import cv2
import numpy as np
import pytest

from app.vlm.client import VLMClient


@pytest.fixture
def vlm():
    return VLMClient.__new__(VLMClient)  # skip OpenAI client construction


def _write_image(path, w, h):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (0, 128, 255)
    cv2.imwrite(str(path), img)


class TestEncodeImage:
    def test_returns_data_uri(self, vlm, tmp_path):
        p = tmp_path / "img.jpg"
        _write_image(p, 200, 100)
        uri = vlm._encode_image(str(p))
        assert uri.startswith("data:image/jpeg;base64,")
        # payload decodes to a valid JPEG
        raw = base64.b64decode(uri.split(",", 1)[1])
        decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None

    def test_large_image_downscaled(self, vlm, tmp_path):
        p = tmp_path / "big.jpg"
        _write_image(p, 3000, 2000)
        uri = vlm._encode_image(str(p), max_size=1024)
        raw = base64.b64decode(uri.split(",", 1)[1])
        decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        h, w = decoded.shape[:2]
        assert max(h, w) <= 1024

    def test_missing_file_raises(self, vlm):
        with pytest.raises(ValueError, match="Cannot read image"):
            vlm._encode_image("/nonexistent/path.jpg")


class TestBuildPrompt:
    DET = {"image": "x.jpg", "inference_ms": 42.0, "counts": {"person": 2},
           "detections": [{"class": "person", "confidence": 0.9,
                           "bbox": [1, 2, 3, 4]}]}

    def test_contains_detection_context(self, vlm):
        prompt = vlm._build_prompt(self.DET)
        assert "Detection Context" in prompt
        assert '"person"' in prompt  # detection JSON embedded
        assert "42.0" in prompt

    def test_requests_four_sections(self, vlm):
        prompt = vlm._build_prompt(self.DET)
        for section in ["Scene Description", "Object Inventory",
                        "Detector Gaps", "Risk Assessment"]:
            assert section in prompt

    def test_mentions_domain_gap(self, vlm):
        prompt = vlm._build_prompt(self.DET)
        assert "domain gap" in prompt
