# tests/test_exporter.py
"""Unit tests for ReportExporter — files land in a tmp dir, never in reports/."""

import json

from app.reporting.exporter import ReportExporter

DET = {
    "image": "test.jpg",
    "inference_ms": 42.0,
    "counts": {"person": 1},
    "detections": [{"class": "person", "confidence": 0.9,
                    "bbox": [1, 2, 3, 4]}],
}
STANDARDS = [{
    "text": "Workers must wear helmets in active construction zones.",
    "standard": "Construction Site Safety",
    "source": "construction_site_safety.md",
    "distance": 0.25,
}]
DECISIONS = ["Zero objects detected @ conf=0.25; lowering threshold to 0.15"]


def _export(tmp_path, draw_fn=None):
    exporter = ReportExporter(out_dir=str(tmp_path / "reports"))
    return exporter.export(
        "test.jpg", DET, "Sample VLM report text.", STANDARDS,
        draw_fn=draw_fn, decisions=DECISIONS)


def test_exports_markdown_and_json(tmp_path):
    saved = _export(tmp_path)
    assert "markdown" in saved and "json" in saved

    md = open(saved["markdown"], encoding="utf-8").read()
    assert "# Visual Inspection Report" in md
    assert "## 1. Detection Summary" in md
    assert "| person | 1 |" in md
    assert "## 2. VLM Analysis" in md
    assert "Sample VLM report text." in md
    assert "## 3. Applicable Standards (RAG)" in md
    assert "Construction Site Safety" in md
    assert "## 4. Agent Decisions" in md
    assert "lowering threshold" in md

    payload = json.load(open(saved["json"], encoding="utf-8"))
    assert payload["detection"]["counts"] == {"person": 1}
    assert payload["vlm_report"] == "Sample VLM report text."
    assert payload["agent_decisions"] == DECISIONS
    assert payload["standards"][0]["relevance"] == 0.75  # 1 - distance


def test_annotated_image_when_draw_fn_succeeds(tmp_path):
    def draw(image_path, det_result, out_path):
        with open(out_path, "wb") as f:
            f.write(b"fake-jpeg-bytes")

    saved = _export(tmp_path, draw_fn=draw)
    assert "annotated_image" in saved
    assert open(saved["annotated_image"], "rb").read() == b"fake-jpeg-bytes"


def test_annotation_failure_does_not_kill_report(tmp_path):
    def failing_draw(image_path, det_result, out_path):
        raise RuntimeError("cv2 crashed")

    saved = _export(tmp_path, draw_fn=failing_draw)
    # md/json still produced; annotated image gracefully omitted
    assert "markdown" in saved and "json" in saved
    assert "annotated_image" not in saved


def test_no_decisions_section_when_empty(tmp_path):
    exporter = ReportExporter(out_dir=str(tmp_path / "reports"))
    saved = exporter.export("test.jpg", DET, "Report.", STANDARDS)
    md = open(saved["markdown"], encoding="utf-8").read()
    assert "Agent Decisions" not in md


def test_no_standards_note(tmp_path):
    exporter = ReportExporter(out_dir=str(tmp_path / "reports"))
    saved = exporter.export("test.jpg", DET, "Report.", [])
    md = open(saved["markdown"], encoding="utf-8").read()
    assert "No relevant standards found." in md


def test_empty_detection_summary(tmp_path):
    empty_det = {"image": "test.jpg", "inference_ms": 1.0, "counts": {},
                 "detections": []}
    exporter = ReportExporter(out_dir=str(tmp_path / "reports"))
    saved = exporter.export("test.jpg", empty_det, "Report.", STANDARDS)
    md = open(saved["markdown"], encoding="utf-8").read()
    assert "No objects detected above threshold." in md
