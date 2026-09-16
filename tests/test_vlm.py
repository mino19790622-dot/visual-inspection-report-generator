# tests/test_vlm.py
"""Unit tests for VLMClient prompt construction and image encoding (no API calls)."""

import base64

import cv2
import numpy as np
import pytest

from app.vlm.client import VLMClient, _extract_usage


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


class TestExtractUsage:
    """Usage must come from the response, cost from the price snapshot.

    This module used to carry a hard-coded blended rate (¥0.02 per 1k tokens),
    which agreed with neither the real input nor the real output rate, and which
    rendered a call with no usage block as "free". These tests exist so that
    cannot silently come back.
    """

    @staticmethod
    def _resp(**usage):
        from types import SimpleNamespace
        return SimpleNamespace(usage=SimpleNamespace(**usage) if usage else None)

    def test_no_usage_block_is_unknown_not_free(self):
        """Provider reported nothing -> not measured, which is not the same as 0."""
        u = _extract_usage(self._resp(), latency_ms=12, model="qwen-vl-max")
        assert u["prompt_tokens"] is None
        assert u["completion_tokens"] is None
        assert u["total_tokens"] is None
        assert u["cost_rmb"] is None
        assert u["cost_known"] is False

    def test_cost_follows_the_snapshot_not_a_constant(self, tmp_path, monkeypatch):
        pricing = tmp_path / "p.yaml"
        pricing.write_text(
            "effective_date: '2026-09-16'\ncurrency: CNY\n"
            "models:\n  qwen-vl-max:\n    input_per_1k: 0.001\n"
            "    output_per_1k: 0.004\n", encoding="utf-8")
        monkeypatch.setenv("PRICING_PATH", str(pricing))
        u = _extract_usage(self._resp(prompt_tokens=1000, completion_tokens=500),
                           latency_ms=30, model="qwen-vl-max")
        assert u["total_tokens"] == 1500
        assert u["cost_known"] is True
        assert u["cost_rmb"] == 0.003  # 1 * 0.001 + 0.5 * 0.004

    def test_unpriced_model_is_unknown_not_zero(self, tmp_path, monkeypatch):
        pricing = tmp_path / "p.yaml"
        pricing.write_text(
            "effective_date: '2026-09-16'\ncurrency: CNY\n"
            "models:\n  qwen-vl-max:\n    input_per_1k: null\n"
            "    output_per_1k: null\n", encoding="utf-8")
        monkeypatch.setenv("PRICING_PATH", str(pricing))
        u = _extract_usage(self._resp(prompt_tokens=1000, completion_tokens=500),
                           latency_ms=30, model="qwen-vl-max")
        assert u["cost_known"] is False
        assert u["cost_rmb"] is None

    def test_no_rate_constant_leaked_back_into_code(self):
        import app.vlm.client as mod
        leaked = [n for n in vars(mod)
                  if "RMB_PER" in n.upper() or "RATE" in n.upper()]
        assert leaked == [], f"a price constant leaked back into code: {leaked}"
