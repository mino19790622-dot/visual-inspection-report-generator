# Maintenance log

> This file is appended automatically by a scheduled maintenance task, and the
> commits that touch it carry `[skip ci]`. Every entry corresponds to one real
> change that passed the repository's own checks before it was pushed.
>
> The entries are generated rather than hand-written, and they record
> maintenance work only — they are not a measure of manual development effort.

## 2026-09-22 — pin the unknown-vs-free invariants of cost_for()

- **Change**: new tests/test_pricing.py (15 tests) covering: a None token count is unknown rather than 0; a measured zero is known and free; a null rate is unknown; PRICING_PATH resolved before the lru_cache; the shipped config/pricing.yaml still parses
- **Verification**: pytest tests: 240 passed; ruff: all checks passed

## 2026-09-23 — document the None-means-undefined contract of spearman()

- **Change**: eval/judge_agreement.py: add a docstring to the public spearman() helper stating that it returns None when rho is undefined (fewer than two paired samples, or a constant side) rather than 0, and that callers must not read None as 'no agreement'
- **Verification**: pytest tests: 240 passed; ruff: all checks passed

## 2026-09-24 — Document the top-vs-nested allow-list switch in _clean()

- **Change**: Added a docstring to app/observability/redaction.py::_clean explaining that the top flag selects ALLOWED_TOP_LEVEL for the root dict and ALLOWED_KEYS for every nested dict, and that the two sets differ. Docstring only; no code path changed.
- **Verification**: pytest tests -q -> 240 passed; ruff check -> All checks passed

## 2026-09-25 — Sort the import block in run_detection.py

- **Change**: Added the blank line ruff's isort rule (I001) requires between the stdlib import (json) and the first-party import (app.detection.detector). Whitespace only, no runtime change; clears the file's single outstanding lint error.
- **Verification**: pytest tests: 240 passed; ruff check run_detection.py: All checks passed

## 2026-09-30 — docs(spans): 补 by_name/totals 的契约说明

- **Change**: app/observability/spans.py: 给 RequestTrace.by_name() 与 totals() 补 docstring；写明 by_name 返回首个匹配 span 或 None，以及 totals() 在 token 合计为 0 时把字段收敛成 None —— 与模块 docstring 第 1 条 '未上报即 None' 一致，避免被读成实测的 0。
- **Verification**: pytest 240 passed; ruff All checks passed

## 2026-10-01 — docs(eval): rank_of_first_relevant 的 1-based 与 None 契约

- **Change**: eval/retrieval_eval.py: 给 rank_of_first_relevant() 补 docstring。写明三点：(1) 返回的 rank 是 1-based，因为 evaluate_retrieval 的 MRR 聚合直接除以它（改 0-based 会整体位移）；(2) None 不是 rank，而是「未检索到相关块」，调用方必须当 miss（RR=0.0）处理；(3) truth 为空时同样返回 None 而非 1，未标注条目不会被算成 rank 1 命中。仅文档，无行为变更。
- **Verification**: pytest 240 passed；ruff All checks passed（eval/retrieval_eval.py）

## 2026-10-02 — docs(grounding): document the max_k_requested contract

- **Change**: app/grounding.py: 给 GroundedReport.max_k_requested 补 docstring，写明 ① 它是本次运行里 retrieve_standards 被请求的最大 k（= 模型决定看多少条条款），被 graph.py 导出为 top_k span 字段、被 ablation_topk.py 记为 k_max 列；② 返回 0 表示 agent 从未调用检索工具（= 没检索），不是请求了 0 条；③ fixed 模式那一次合成预检索以 k=MAX_K(=5) 记入 tool_calls，所以对照组报告满深度。仅文档，无行为变更。
- **Verification**: pytest 240 passed；ruff All checks passed (app/grounding.py)

## 2026-10-05 — docs(tools): budget_exhausted 与 call_log 契约

- **Change**: app/tools.py: 给 ToolRegistry.budget_exhausted 补 docstring，写明预算只计 retrieval 调用、检查发生在参数校验之前（故耗尽后连合法调用也拒），而 submit_findings 完全绕过该检查；给 call_log() 补 docstring，说明返回的是调用顺序的副本。仅文档，无行为变更。
- **Verification**: pytest 240 passed; ruff All checks passed (app/tools.py)

