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
    assert "## 3. Grounded Findings" in md
    assert "## 4. Applicable Standards (RAG)" in md
    assert "Construction Site Safety" in md
    assert "## 5. Agent Decisions" in md
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


# ------------------------------------------------------------------ v2
def _grounded():
    from app.grounding import Citation, Finding, GroundedReport
    return GroundedReport(
        findings=[Finding(
            id="f1", severity="high", description="Unsecured scaffolding.",
            citation=Citation(chunk_id="construction_site_safety.md::1",
                              standard="Construction Site Safety",
                              quote="Workers must wear helmets"),
            citation_status="ok", citation_reason="quote match 1.00")],
        unresolved=["Standing water of unknown depth"],
        tool_calls=[{"tool": "retrieve_standards",
                     "args": {"query": "scaffolding", "k": 2}, "n_results": 2}],
        retrieved=STANDARDS, rounds=2, mode="agentic",
        citation_report={"counts": {"ok": 1, "unresolved": 0,
                                    "hallucinated": 0},
                         "total_findings": 1,
                         "attribution_pass_rate": 1.0, "detail": []})


def test_grounded_findings_render_with_status(tmp_path):
    """A finding's verification status must be visible in the report."""
    exporter = ReportExporter(out_dir=str(tmp_path / "reports"))
    saved = exporter.export("test.jpg", DET, "Report.", STANDARDS,
                            grounded=_grounded())
    md = open(saved["markdown"], encoding="utf-8").read()
    assert "| 1 | high | Unsecured scaffolding. |" in md
    assert "`construction_site_safety.md::1`" in md
    assert "| ok |" in md
    assert "Standing water of unknown depth" in md

    payload = json.load(open(saved["json"], encoding="utf-8"))
    assert payload["grounded"]["findings"][0]["id"] == "f1"
    assert payload["grounded"]["citation_report"]["attribution_pass_rate"] == 1.0


def test_unverified_finding_is_printed_not_dropped(tmp_path):
    """Downgrading must stay visible — silence would hide the failure."""
    from app.grounding import Citation, Finding, GroundedReport
    grounded = GroundedReport(
        findings=[Finding(id="f1", severity="low", description="Invented claim.",
                          citation=Citation(chunk_id="nowhere.md::9",
                                            quote="made up"),
                          citation_status="hallucinated")],
        retrieved=[], mode="agentic",
        citation_report={"counts": {"ok": 0, "unresolved": 0,
                                    "hallucinated": 1},
                         "total_findings": 1,
                         "attribution_pass_rate": 0.0, "detail": []})
    exporter = ReportExporter(out_dir=str(tmp_path / "reports"))
    saved = exporter.export("test.jpg", DET, "Report.", STANDARDS,
                            grounded=grounded)
    md = open(saved["markdown"], encoding="utf-8").read()
    assert "hallucinated" in md


def test_parse_failure_is_stated_in_report(tmp_path):
    from app.grounding import GroundedReport
    grounded = GroundedReport(parse_failed=True, mode="agentic",
                              failure_reason="model returned no tool call")
    exporter = ReportExporter(out_dir=str(tmp_path / "reports"))
    saved = exporter.export("test.jpg", DET, "Report.", STANDARDS,
                            grounded=grounded)
    md = open(saved["markdown"], encoding="utf-8").read()
    assert "failed to produce structured findings" in md


def test_no_grounded_output_when_stage_absent(tmp_path):
    exporter = ReportExporter(out_dir=str(tmp_path / "reports"))
    saved = exporter.export("test.jpg", DET, "Report.", STANDARDS)
    md = open(saved["markdown"], encoding="utf-8").read()
    assert "No grounded findings." in md
    assert "grounded" not in json.load(open(saved["json"], encoding="utf-8"))
