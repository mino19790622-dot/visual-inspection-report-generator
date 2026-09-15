# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is manual and tag-based.

## [Unreleased]

Nothing yet.

## [v1.0] — 2026-09-15 (baseline snapshot, tagged at `0d22732`)

Tagged immediately before the v2 upgrade so that every v2 claim can be
diffed against a frozen baseline. No code changed in this entry.

### What v1.0 is

- **Agent**: LangGraph conditional routing (`detect → review → [retry | analyze
  → assess_risk → retrieve → export]`); adaptive confidence-threshold retry on
  zero detections; risk-based retrieval depth (`k = 5/4/3` for high/medium/low).
- **Vision**: YOLOv8m via ONNX Runtime; PyTorch↔ONNX output consistency verified.
  Qwen-VL-Max structured visual analysis at `temperature=0`.
- **RAG**: 6 inspection standards chunked (600 char / 100 overlap), embedded with
  DashScope `text-embedding-v2`, stored in ChromaDB, retrieved by cosine similarity.
- **Output**: timestamped Markdown + JSON report; every agent decision logged and
  exported with the report.
- **Serving**: FastAPI + Swagger, Docker (slim image), deployed on AWS ECS Fargate
  (ECR / Secrets Manager / CloudWatch / GitHub Actions OIDC).
- **Quality**: pytest suite with LLM and vision service boundaries mocked;
  ruff lint + coverage gate in CI; a separate `eval/` golden-set LLM-as-judge
  gate (manual trigger) scoring VLM report quality on 10 images.

### What v1.0 cannot yet show

Honest statement of the limits this tag freezes in place:

- Retrieval quality is unmeasured — there are no retrieval metrics.
- The report cites no standard clauses, so claim correctness is unverifiable.
- Cost and latency per report are not attributed per stage.
- The saving from dynamic `top_k` is a stated design motivation only — never
  measured.
- The model output is free-form Markdown; there is no structured finding
  schema and no citation field, so hallucinated clauses cannot be detected.
- Prompts are inline strings in `app/vlm/client.py` — no versioning, no
  regression comparison, no rollback.

### Known structural limitation (discovered 2026-09-15, drives the v2 plan)

In v1.0 the retrieval step runs **after** generation
(`detect → analyze → assess_risk → retrieve → export`). The retrieved standards
are appended to the report as a human-readable appendix and **never enter the
VLM prompt**. v1.0 therefore performs retrieve-then-append, not
retrieve-then-generate: there is no grounded generation and no citation to
validate. See `docs/V2_GAP_ANALYSIS.md`.

[v1.0]: https://github.com/mino19790622-dot/visual-inspection-report-generator/releases/tag/v1.0
