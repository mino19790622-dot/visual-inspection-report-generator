# tests/test_detection.py
"""Unit tests for YOLODetector preprocessing / postprocessing (no ONNX needed).

The detector is instantiated via __new__ so the real 50MB ONNX session is
never built — we only exercise the pure numpy/cv2 logic.
"""

import numpy as np
import pytest

from app.detection.detector import YOLODetector


@pytest.fixture
def detector():
    d = YOLODetector.__new__(YOLODetector)
    d.conf_thres = 0.35
    d.iou_thres = 0.5
    d.class_names = ["person", "car", "dog", "cell phone"] + [
        f"c{i}" for i in range(76)]
    d.input_size = 640
    return d


# ---------------------------- letterbox ---------------------------- #
class TestLetterbox:
    def test_landscape_image(self, detector):
        img = np.zeros((720, 1280, 3), dtype=np.uint8)  # 16:9
        canvas, scale, pad = detector._letterbox(img)
        assert canvas.shape == (640, 640, 3)
        assert scale == pytest.approx(0.5)
        assert pad == (0, (640 - 360) // 2)  # no x-pad, centered y-pad

    def test_portrait_image(self, detector):
        img = np.zeros((800, 480, 3), dtype=np.uint8)
        canvas, scale, pad = detector._letterbox(img)
        assert scale == pytest.approx(0.8)
        assert pad == ((640 - 384) // 2, 0)  # centered x-pad, no y-pad

    def test_padding_is_gray_114(self, detector):
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        canvas, _, (pad_left, pad_top) = detector._letterbox(img)
        # corner pixel lies in the letterbox padding
        assert (canvas[0, 0] == 114).all()
        # center pixel lies inside the scaled image (zeros)
        assert (canvas[320, 320] == 0).all()

    def test_content_preserved(self, detector):
        img = np.full((640, 640, 3), 200, dtype=np.uint8)  # square, scale=1
        canvas, scale, pad = detector._letterbox(img)
        assert scale == 1.0 and pad == (0, 0)
        assert (canvas == 200).all()


# ---------------------------- postprocess ---------------------------- #
def _make_output(rows, n=8400):
    """Build a (1, 84, 8400) tensor with only `rows` columns non-zero."""
    out = np.zeros((1, 84, n), dtype=np.float32)
    for col, (cx, cy, w, h, cls, conf) in enumerate(rows):
        out[0, :4, col] = [cx, cy, w, h]
        out[0, 4 + cls, col] = conf
    return out


class TestPostprocess:
    def test_bbox_decoding_roundtrip(self, detector):
        # one car, confidence 0.9, in a 1280x720 image (letterbox: scale=.5,
        # pad=(0,140)); letterbox coords (320,320,100,100) -> orig (540,260,740,460)
        out = _make_output([(320, 320, 100, 100, 1, 0.9)])
        dets = detector._postprocess(out, 0.5, (0, 140), (720, 1280))
        assert len(dets) == 1
        d = dets[0]
        assert d["class"] == "car"
        assert d["confidence"] == 0.9
        assert d["bbox"] == [540, 260, 740, 460]

    def test_confidence_filter(self, detector):
        out = _make_output([
            (320, 320, 100, 100, 1, 0.90),  # above 0.35
            (100, 100, 50, 50, 2, 0.10),    # below -> dropped
        ])
        dets = detector._postprocess(out, 1.0, (0, 0), (640, 640))
        assert len(dets) == 1
        assert dets[0]["class"] == "car"

    def test_nms_suppresses_overlap(self, detector):
        # two heavily overlapping boxes, same class -> only the best kept
        out = _make_output([
            (300, 300, 100, 100, 1, 0.90),
            (310, 310, 100, 100, 1, 0.80),
        ])
        dets = detector._postprocess(out, 1.0, (0, 0), (640, 640))
        assert len(dets) == 1
        assert dets[0]["confidence"] == 0.90

    def test_nms_keeps_distant_boxes(self, detector):
        out = _make_output([
            (150, 150, 80, 80, 1, 0.90),
            (500, 500, 80, 80, 2, 0.85),
        ])
        dets = detector._postprocess(out, 1.0, (0, 0), (640, 640))
        assert {d["class"] for d in dets} == {"car", "dog"}

    def test_bbox_clipped_to_image(self, detector):
        # box partially outside the (640,640) frame -> clipped
        out = _make_output([(10, 10, 100, 100, 0, 0.9)])
        dets = detector._postprocess(out, 1.0, (0, 0), (640, 640))
        assert dets[0]["bbox"] == [0, 0, 60, 60]

    def test_empty_output(self, detector):
        out = np.zeros((1, 84, 8400), dtype=np.float32)
        assert detector._postprocess(out, 1.0, (0, 0), (640, 640)) == []

    def test_counts_helper(self, detector):
        # mimic detect()'s counts aggregation
        dets = [{"class": "car"}, {"class": "car"}, {"class": "dog"}]
        counts = {}
        for d in dets:
            counts[d["class"]] = counts.get(d["class"], 0) + 1
        assert counts == {"car": 2, "dog": 1}
