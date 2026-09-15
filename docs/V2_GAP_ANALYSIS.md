# v2 升级：现状核查与方案修正

> 核查对象：`v1.0` (commit `0d22732`)，核查日期 2026-09-15。
> 结论：`v2-upgrade-plan.md` 的**三处前提与代码实际不符**，其中 G1/G2 会直接阻断 4.3（引用评测）与 4.5（引用校验）。
> 需要插入一个 **Phase 0**，并把 Phase 1 内部顺序从 `B → C → A` 调整为 `P0 → C → B → A`。

---

## 一、偏差清单

| # | 方案的假设 | 代码实际（v1.0） | 严重度 | 处置 |
|---|---|---|---|---|
| **G1** | 检索结果进入生成上下文（grounded generation） | `graph.py` 的边序是 `detect → analyze → assess_risk → **retrieve** → export`。**检索发生在 VLM 生成之后**；`exporter.py:101-115` 把 standards 作为报告第 3 节附录直接拼进去，**从未进入 VLM prompt** | 🔴 **阻断** | Phase 0 重组图 |
| **G2** | finding 带 citation 字段 | `vlm/client.py` 的 prompt 要求输出 4 段**自由 Markdown**（Scene / Object Inventory / Detector Gaps / Risk Assessment），**没有结构化 finding，没有 citation 概念** | 🔴 **阻断** | Phase 0 引入 schema |
| **G3** | 无 golden set | 有，`eval/golden_set/golden_set.json` 10 条。但字段只有 `risk_level` / `min-max_detections` / `must_mention_any` / `rubric` —— **没有标准条款真值、没有 citation、没有 split、没有 dataset hash** | 🟡 重建 | 扩展而非重写 |
| **G4** | 无 LLM judge | 有，`eval/judge.py` 用 `qwen-turbo` 打 4 个维度 1–5 分。但判的是**整篇报告质量**，不是引用依据 | 🟡 复用+新增 | 保留原 judge，另加 citation judge |
| **G5** | 无成本埋点 | 有，`app/observability.py` 写 `logs/inspect.jsonl`。但只有 VLM 的 token/延迟，**成本是硬编码 `¥0.02/1K tokens`**（`vlm/client.py:108`，注释自认 "rough estimate"），无阶段 span、无脱敏 | 🟡 改造 | 扩写，不重写 |
| **G6** | CI 无 eval | 有，`.github/workflows/eval.yml`。但**仅 `workflow_dispatch` 手动触发**，无 baseline、无 dataset hash、无 PR 门禁 | 🟡 加门禁 | 扩写 |
| **G7** | 覆盖率门禁 87% | CI 实际 `--cov-fail-under=80`；CI 用 py3.11 而 eval.yml 用 py3.12 | ⚪ 记录 | 顺手统一 |
| **G8** | — | `judge` 用 `qwen-turbo`，被测用 `qwen-vl-max`，**同家族** | ⚠️ 已命中 | 正是方案 §7.6 预警的 self-enhancement bias，README 必须声明 |
| **G9** | — | 裸 `open(path,"a")` 写日志，无字段白名单；日志里写的是 `image` 文件名（`basename`），暂无 base64/key 泄露，但**无任何防回归测试** | ⚠️ 补测试 | 加 `test_log_redaction.py` |

---

## 二、G1 详解：为什么它是阻断项

当前数据流：

```
detect → analyze(VLM, 不含标准) → assess_risk → retrieve → export
                                                    ↓
                                        报告第 3 节（给人看的附录）
```

**后果链：**

1. VLM 生成时**看不到**任何标准条款 → 所谓 "grounded response generation" 在 v1.0 不存在
2. 因此报告里**没有引用** → 4.3.3 的"引用正确率"和 4.5.2 的"引用必须落在检索集内"**无处落地**
3. 因此 4.3.4 那个"无上下文对照实验"（`top_k=0` 跑一遍看引用正确率掉不掉）会**必然显示零差异** —— 不是因为系统好，是因为上下文从来就没被用过
4. 更糟的是：这一格如果照原计划跑出来，会给出一个**假的安全信号**

**这不是"缺功能"，是"现有 RAG 只是检索后附录"。** JD 里的 `grounded response generation` 目前无法被证实，也无法被证伪。

---

## 三、G10（新发现）：把检索前置会撞上一个循环依赖

方案原本想当然地认为"把 retrieve 挪到 analyze 前面就行"。实际不行：

```
动态 top-k 的规则是："按 VLM 输出的风险等级决定 k"
        ↑                        ↓
        └──── VLM 现在需要标准上下文才能生成 ────┘
```

**风险等级来自 VLM 输出，而 VLM 输出需要检索结果，而检索深度由风险等级决定 —— 循环了。**

三条解法，各自代价不同：

| 解法 | 做法 | 代价 | 评价 |
|---|---|---|---|
| **A. 两阶段检索** | 先按 detection 粗检 `k=3` → VLM 生成 → 按风险等级补检到 `k=5` | 多一次检索调用，逻辑变复杂 | 可行但笨 |
| **B. 固定 k 前置** | 检索固定 `k` 提前，彻底放弃动态 top-k | 丢掉 v1.0 的一个卖点，且 4.2.3 的对照实验就没了 | 不推荐 |
| **C. tool-calling（推荐）** | 检索做成工具，**由模型在生成过程中按需决定调不调、调几次、k 多大**，代码侧 `clamp(1, MAX_K)` 兜底 | 需要先做 C | ✅ **这正是 JD 点名的能力，而且是本问题的自然解，不是为用而用** |

**结论：C（tool-calling）从"JD 凑条目"升级为"架构必需项"。** 这也让 4.10 里 C 的那句英文更有分量 —— 你不是"为了面试加了个 tool calling"，而是"为了解决检索与生成的循环依赖引入的 tool calling"。

---

## 四、修正后的执行顺序

```
Phase 0（新增，约 6h）→ C（tool-calling，5h）→ B（评测，约 10h）→ A（成本收尾，3h）
                          ↘ B 的 golden set 标注（纯人工）可与 C 并行 ↗
A 的埋点（3h）前置到 Phase 0 —— 呼应原方案 §7.2
```

### Phase 0 — 结构化输出与图重组（约 6h）★新增

| 项 | 内容 | 工时 |
|---|---|---|
| **P0-1** | VLM 输出 schema：`{narrative: str, findings: [{id, severity, description, citation: {standard, clause_id, quote}, confidence}]}`。**保留原 4 段式作为 `narrative` 字段**，不破坏现有报告与 eval | 2h |
| **P0-2** | Pydantic 模型 + 解析 + 解析失败降级（保留 `narrative`，`findings=[]` 并打 `parse_failed`） | 1.5h |
| **P0-3** | 图重组：`detect → retrieve(工具) → analyze(带上下文) → assess_risk → export`；动态 top-k 交给工具决定 + 代码 clamp | 2h |
| **P0-4** | 更新受影响测试（现有 mock 假设 VLM 返回字符串） | 1h |
| **P0-5** | A 的埋点前置（span + `pricing.yaml` + 脱敏白名单 + `test_log_redaction.py`） | 1.5h |

> **P0 的完成标志**：跑一张图，报告里出现至少一条带 `citation` 的 finding，且 `citation.standard` 确实来自本次检索结果 —— 这一条成立，4.3 和 4.5 才有落脚点。

### 工时重估

| | 原方案 | 修正后 | 差 |
|---|---|---|---|
| Phase 0 | — | 8.5h（P0-1…P0-5 含埋点） | **+8.5** |
| A 成本与延迟 | 6h | 2.5h（埋点已前置，只剩对照实验与报表） | −3.5 |
| B 评测 | 16h | 10h（judge 与 golden set 有底子可复用） | −6 |
| C tool-calling | 5h | 5h | 0 |
| **Phase 1 合计** | ~30h | **~26h** | −4 |
| Phase 2（D+E） | 9h | 9h | 0 |
| **总计** | ~39h | **~35h** | −4 |

> 净减 4h，是因为 G4/G5/G6 的存量代码比方案假设的多。**但 Phase 1 内部的依赖关系变了，顺序不能照抄原表。**

---

## 五、G3 的处置：golden set 扩展而非重写

现有 10 条有真实价值（rubric 是人工写的，judge 也调过），**不要丢**。做法：

1. **保留**现有 10 条作为 `regression` split，继续跑现有 judge —— 它是 P0 重构的**安全网**：如果 P0 改完这 10 条的分数掉了，说明重构破坏了报告质量
2. **新增字段**到每条：`must_cite`（标准文件名 + 条款/章节标识）、`nice_to_have`、`hard_negative_chunks`
3. **扩充到 24–30 条**，新增条目从现有 6 份标准里挑可配对的
4. **加 `splits.yaml`**（calibration / tuning / test）+ `dataset_hash`
5. 现有 `run_eval.py` 的 `--skip-judge` 路径（纯确定性检查）可直接复用为 CI 的 smoke 门禁

⚠️ 注意：现有 6 份标准是**通用场地检查**文档（施工安全、道路、行人、公共交通、环境、裂缝），不是焊接/铸件类的工业验收标准（无 ISO 5817 之类）。**citation 真值必须按实际文档标**，不要用方案里举例的条款编号 —— 那只是举例。

---

## 六、待你决策的三件事

1. **Phase 0 是否现在开工？** 它不在原计划里，且会改动已部署的 pipeline 行为（VLM prompt 与图结构都变）。
2. **G10 走解法 C（tool-calling）还是解法 A（两阶段检索）？** 我推荐 C，但它把 5h 的 C 提到了 Phase 1 的最前面。
3. **是否先在 `v1.0` 分支上验证 P0，再合回 main？** 现有 `eval.yml` 是手动触发的，P0 改完需要先手动跑一次现有 10 条确认没退化。
