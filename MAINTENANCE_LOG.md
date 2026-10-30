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

## 2026-10-06 — letterbox return contract

- **Change**: app/detection/detector.py: 给 YOLODetector._letterbox 补 docstring，写明返回三元组 (canvas, scale, (pad_left, pad_top)) 的契约：scale 作用于最长边故可直接相除；pad 顺序是 (left, top) 即先 x 后 y，_postprocess 按 pad[0]->x / pad[1]->y 消费，写反会让非方形输入的框整体偏移；canvas 填充 114 且用整除居中，奇数余数偏向 bottom/right。仅文档，无行为变更。
- **Verification**: pytest 240 passed; ruff All checks passed (app/detection/detector.py)

## 2026-10-07 — Document that fuzzy_contains ignores its threshold argument

- **Change**: Added a docstring paragraph to app/citations.py::fuzzy_contains noting that the threshold parameter is never read in the body: the raw best similarity is always returned, and the score < threshold decision lives in check_citation. Signature-only parameter, kept for positional forwarding.
- **Verification**: pytest tests: 240 passed (files=17, unchanged); ruff check app/citations.py: All checks passed

## 2026-10-08 — RAG indexing contract

- **Change**: app/rag/retriever.py: docstring for StandardsRetriever._index_standards. Records that it is idempotent off collection.count()>0, so editing data/standards/*.md has no effect until .chroma_db is deleted; only *.md is read; empty dir raises ValueError; and it runs from __init__ so construction hits DashScope when the store is empty. Doc-only, +23/-1, no behaviour change.
- **Verification**: pytest 240 passed; ruff All checks passed on app/rag/retriever.py

## 2026-10-09 — docs(pricing): document _load_raw/_load_cached contracts

- **Change**: Add docstrings to app/observability/pricing.py::_load_raw and ::_load_cached. _load_raw: the three 'no table here' cases (file absent / optional yaml missing / YAML parses to None) all collapse to {}, so an empty mapping is reported as unpriced (known=False), never as free. _load_cached: lru_cache(maxsize=8) is keyed on the already-resolved path, so load_pricing must apply the PRICING_PATH env override first; the cached mapping is one shared object, so callers must treat it as read-only and tests that rewrite the snapshot should call cache_clear(). Docs only, +20 lines, no behaviour change.
- **Verification**: pytest 240 passed; ruff All checks passed (changed file only)

## 2026-10-15 — docs(exporter): pin the export() path-map and timestamp contracts

- **Change**: app/reporting/exporter.py::ReportExporter.export() docstring +19 lines, docs only. Four contracts, each measured with a temp-dir probe before writing: (1) the returned key set is not fixed - markdown/json always present, annotated_image only when draw_fn is given AND succeeds; (2) on draw failure the key is popped, not set to None/'' - callers must test '"annotated_image" in saved' (backed by test_annotation_failure_does_not_kill_report); (3) popping only withdraws the claim - a draw_fn that writes then raises leaves the partial _det.jpg on disk, verified by probe (leftover_det_jpg exists=True), so a later '*_det.jpg' glob can pick up a truncated image; (4) base comes from one clock read shared by all three paths, but generated_at (JSON) and Generated (Markdown) read the clock again, so they are not guaranteed to match the filename stamp; relevance is the same 1-distance rendered at 3 dp in JSON (probe: 0.754) vs :.2f in Markdown (probe: 0.75). No behavior change.
- **Verification**: pytest 240 passed; ruff All checks passed (changed file only)

## 2026-10-16 — docs(api): pin the _save_upload validation and naming contract

- **Change**: Added a docstring to _save_upload in app/api/server.py (pure documentation, +22/-0, no behaviour change). It records four things a reader currently has to infer: (1) validation is filename-only — ext comes from the client-supplied filename, bytes are never sniffed and the declared content type is ignored, so a JPEG mislabelled .png is accepted while a real image with no/unknown extension gets 415 before any file is written (filename=None yields '' and is rejected the same way); (2) the client filename never reaches the filesystem — the stored name is a fresh uuid4().hex[:12] plus the lowercased ext, so traversal in filename is inert and concurrent uploads cannot collide; (3) the body is read whole via image.file.read() with no application-level size cap — Starlette spools large bodies to disk rather than rejecting them; (4) the returned path is what the agent and app.observability.log_request see, and log_request records only its basename, so the original image.filename is echoed in the response but never written to disk or to the log.
- **Verification**: pytest 240 passed (72*3+24, unchanged); ruff check on the changed file: All checks passed

## 2026-10-19 — docs(eval): pin the agreement/gate_impact denominators

- **Change**: eval/judge_agreement.py: document that agreement() pairs on the intersection of both raters (n_images_compared ignores images only in hand; n_pairs != 4*n_images_compared when a dimension is blank; rates are None not 0.0 on empty denominators), and that gate_impact() normalises the judge average by 4 unconditionally while the rater average divides by dimensions present, raising ZeroDivisionError when a hand entry has none of DIMENSIONS. All three behaviours verified by probe before writing.
- **Verification**: pytest 240 passed; ruff All checks passed

## 2026-10-20 — vlm client: 密钥回落与检测器描述硬编码

- **Change**: 为 VLMClient.__init__ 与 _build_prompt 补 docstring（+32/-0，纯文档）。__init__: api_key 显式取自 DASHSCOPE_API_KEY，而 OpenAI SDK 把 None 当作回落到 OPENAI_API_KEY ⇒ 只有 OPENAI_API_KEY 的机器构造会静默成功、把该 key 发往 dashscope 域名，直到首次 analyze 才报错；两者都缺则构造期即抛 OpenAIError。另 load_dotenv() 在模块导入时执行。_build_prompt: 检测器被写成硬编码字面量 YOLOv8m/COCO 80 类，而 onnx_path 与 classes_path 均为运行时参数、run_pipeline.py 还暴露 --onnx ⇒ 换权重或换类别表后 prompt 仍宣称 COCO-80，不报错但第 3 节按错误类空间归因、第 2 节被要求照抄检测计数。
- **Verification**: pytest 240 passed（72x3+24 进度点自洽）；ruff All checks passed（files: app/vlm/client.py）

## 2026-10-21 — docs(judge_agreement): pin load_judge_scores drop and type rules

- **Change**: eval/judge_agreement.py::load_judge_scores docstring, +21 lines, docs only. Records: (1) returns (scores, artifact) where the second element is the whole parsed artifact, not a per-image map; (2) each kept entry holds exactly the four DIMENSIONS, so extra keys like reason are dropped; (3) three silent-drop paths -- missing/non-dict judge_raw, truthy judge_raw.skipped (what run_eval.py:109 writes under skip_judge), or any dimension absent/non-int -- so a skipped image never surfaces as a disagreement; (4) isinstance(x, int) accepts bool, so a true becomes 1 rather than being rejected, while "4" and 4.0 both drop the row. All four verified with a /tmp probe over 8 synthetic rows: only ok and boolval survived, boolval kept scene_id=True.
- **Verification**: pytest 240 passed; ruff All checks passed on eval/judge_agreement.py

## 2026-10-22 — docs(tools): pin _validate reject-vs-clamp asymmetry

- **Change**: app/tools.py::ToolRegistry._validate 补 docstring(+29/-0, 纯文档)。写明两类失败刻意不同：query 非空串/k 非真 int → 拒绝(bool 因 isinstance(True,int) 被显式排除)；类型合法但越界的 k → 静默 clamp(0/-3→1, 99→max_k)，只留一条 INFO 日志。另写明 args 是副本、query 被 strip、以及 clamp 后 execute 落进 call_log ⇒ max_k_requested / top_k span 字段对「要 5」和「要 99」记录相同值。所有断言先用探针实跑取得。
- **Verification**: pytest 240 passed(files=17, 3x72+24 进度点自洽); ruff files=[app/tools.py] All checks passed

## 2026-10-24 — docs(observability): document the redaction heuristics

- **Change**: 补 _looks_secret 的 docstring（两阶段判定：先 payload 形状 data-URI/长 base64，再凭证前缀模式；非字符串一律不判为密钥，避免对 total_tokens 之类数字字段误报）；补 _env_values 的 docstring，说明 8 字符下限是为了不把 1/true/tmp 这类短值塞进密钥集合而误伤整条记录；_clean 的 docstring 补一句 top 不向子节点传播（递归调用固定传默认 False，嵌套 dict 一律按 ALLOWED_KEYS 过滤）。纯文档，无运行时行为变化。
- **Verification**: pytest 240 passed / 0 failed（tests/ 全量，进度点 72x3+24 计数）；ruff check --no-cache 仅 lint 本次 dirty 文件 app/observability/redaction.py → All checks passed!

## 2026-10-26 — per_image_deltas: name the rows that vanish and the dedupe mismatch

- **Change**: eval/cost_table.py: per_image_deltas() 原 docstring 只有一句「where the two grounded arms disagree」，把三条会误导读表人的隐式契约全留空。补写 +31/-2（纯文档）：① 输出按 id 重组，只有同时具备 fixed 与 agentic 行的图片才会出现；只在 off 下跑过、或 grounded 行未记录的图片静默消失 ⇒ 列表短意味着「不可比」而非「无差异」，不能读成一致（探针：img2 只有 fixed+off 行，deltas 仅返回 img1）。② 重复的 (id, arm) 行在此折叠为最后一条，而 _sum_tokens 会把该 arm 的每一行都加进去 ⇒ 同一图片被记录两次时，arm 合计与逐图 delta 会互相矛盾（探针：两条 fixed 行 findings 2、7 ⇒ arm 合计 9，但 findings_fixed=7）。③ findings_fixed/findings_agentic 回显原始值、可为 None，而 findings_delta 把 None 当 0 ⇒ 缺失计数会打印成 None -> 2 (+2)，与真实的 0 -> 2 无法区分。另给 image_count() 补两点：该分母被所有 arm 的 per-image 列复用（行数少的 arm 仍除以全量图片数）；r[id] 是硬取值，缺 id 的行会 KeyError 拖垮整张表。四条均用 /tmp 合成 artifact 探针实测，改前改后探针输出逐字一致 ⇒ 无行为变更。
- **Verification**: pytest 240 passed; ruff All checks passed (eval/cost_table.py)

## 2026-10-27 — docs(agent): pin the always-zero total_latency_ms

- **Change**: Documented export_node(): _t_start is seeded into initial but not declared in AgentState, and LangGraph keeps only declared channels, so state.get('_t_start') is always None and total_ms is always 0. That 0 reaches /inspect metrics.total_latency_ms and logs/inspect.jsonl, while the v2 trace timing stays real. Also noted both timings exclude export() IO. Docs only, zero behaviour change.
- **Verification**: pytest 240 passed; ruff app/agent/graph.py All checks passed

## 2026-10-28 — Document total_ms as a live wall-clock reading

- **Change**: app/observability/spans.py: added a docstring to RequestTrace.total_ms(). Three measured properties: (1) it is a live reading, not a frozen total - each call re-reads perf_counter, so the JSONL latency_ms is fixed at to_record() time; (2) it includes time outside any span (209.2ms trace vs 105.8ms of spans, 103.5ms unattributed) and is not reconcilable with the per-stage breakdown, while summing duration_ms over-reports when spans nest (130.2 vs 86.9); (3) it stops before artifacts are written, since log_trace runs before exporter.export(). Also noted _t0 is monotonic while started_at is wall clock. Docs only, no behaviour change.
- **Verification**: pytest 240 passed; ruff All checks passed (app/observability/spans.py)

## 2026-10-30 — 记录 agent 两个节点的 state 契约

- **Change**: 给 detect_node / lower_threshold_node 补 docstring：前者说明 conf_thres 默认 0.25、只写回 det_result、detector 是 lru_cache 单例因此每次调用前覆盖阈值、有 trace 时包 detect span；后者说明阈值按 0.6 衰减、下限 0.10、router 最多重试一次（retry_count==0 才进），并点明 decisions 条目会进审计轨迹。纯注释，无运行时改动。
- **Verification**: pytest 240 passed（17 个测试文件）；ruff check 通过（仅 app/agent/graph.py 被改）

