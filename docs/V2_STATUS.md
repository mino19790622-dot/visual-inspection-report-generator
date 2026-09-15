# V2 升级状态 · Status（阶段 C 含 Phase 0 补正）

> 本文件回答 `prompt_v2升级_阶段C_含Phase0补正.md` 第〇节的三个前置问题。
> 所有结论基于已提交代码与一次真实 CI 运行，非臆测。

---

## Q1 · tool-calling 是真的 function calling 吗？

**结论：(a) 真 function calling，不是 prompt 里描述工具再自己解析 JSON。**

### 证据（代码层）

`app/grounding.py` 的工具循环走的是 OpenAI 兼容的 function-calling 协议：

```python
# GroundingAgent.run()
resp = self.client.chat.completions.create(
    model=self.model, messages=messages,
    tools=specs,            # OpenAI 兼容 function schema: {"type":"function","function":{...}}
    tool_choice="auto", temperature=temperature)
msg = resp.choices[0].message
tool_calls = getattr(msg, "tool_calls", None)   # 服务端返回的结构化调用
```

- 工具 schema 定义在 `app/tools.py`：`RETRIEVE_STANDARDS_SPEC` / `SUBMIT_FINDINGS_SPEC`，格式为
  `{"type":"function","function":{"name":...,"description":...,"parameters":{...}}}`——这是
  OpenAI / DashScope 标准的 function-calling 入参，不是 prompt 文本。
- 模型返回的 `tool_calls` 由 SDK 解析为结构化对象（`tc.function.name` / `tc.function.arguments`），
  循环据此执行 `retrieve_standards` / `submit_findings`，并把结果以 `role:"tool"` 消息回传。
- 客户端：`OpenAI(api_key=..., base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")`，
  模型默认 `GROUNDING_MODEL="qwen-plus"`。DashScope 的 compatible-mode 端点官方支持 function calling。

### 证据（运行层，实测）

分支 `chore/v2-baseline` 上的 golden-set eval（`actions/runs/35008709697`，agentic 模式）实测：

- `parse_failed=False` 于全部 10 张图 → 模型确实通过 `tool_calls` 返回了可被 pydantic 校验的结构化 `submit_findings`，
  而非自由文本 JSON 被正则解析。
- 检索调用真实发生（`tool_calls` 计数 0–5/图），`findings` 带 `citation.chunk_id` 指向本次真实检索到的 chunk。
- 这说明 tool-calling 循环端到端跑通，是简历缺口"从未做过 tool-calling"的实补。

### 边界（仍不能声称）

- 未做"在 prompt 里描述工具 + 自己正则解析 JSON"的退化路径——当前就是 (a)。
- 未对 function-calling 做超时/部分失败的特殊重试策略（循环已有 `MAX_ROUNDS` 安全阀，但非针对 tool 协议）。
- 面试话术不得声称多租户 / 分布式推理 / Azure / K8s / 大规模生产评测 / fine-tuning。

---

## Q2 · 时间账

> 以下为**诚实估算**，非精确工时追踪（本仓库无时间记账工具）。Phase 0 为计划外。

| 阶段 | 预算 | 实际（估） | 累计（估） | 备注 |
|---|---|---|---|---|
| Phase 0 | 未列（计划外） | ~9h | ~9h | P0-1..P0-5：citations/tools/grounding/observability 包 + graph/retriever/exporter/run_eval 改造 + 6 新测试 + 修 numpy/ruff/测试 ≈ 9h |
| C | 5h | — | — | 本阶段进行中 |
| B | 8h | — | — | golden-set 扩展 + 检索真值 |
| A | 6h | — | — | 填 pricing.yaml 真实费率 + token/€ 对比 |
| **合计（含 P0）** | **19h（原 Phase 1 预算，不含 P0）** | | **~28h** | **超预算 ~9h** |

###  overrun 说明与砍范围建议（砍范围，不砍发版时间）

原计划 Phase 1 预算 19h 只覆盖 C/B/A，**不含**计划外的 Phase 0（~9h）。累计 ~28h，超 ~9h。
按"砍范围不砍发版时间"，建议：

1. **B 从 8h → 5h**：golden set 扩展用更小的校准 split（~15 图而非 30），recall@k/MRR 在检索真值就绪前
   本就禁止计算（见 prompt 2.3），顺延，不占时间。
2. **A 从 6h → 3h**：填 `config/pricing.yaml` 费率 + 一次 ablation 运行即出 token/€ 对比，砍掉可选的可视化看板。
3. **D/E 明确移到 Phase 2**：prompt 第〇/六节已把它们标为"现在不做"，Phase 1 范围 = P0 + C + B(最小) + A(核心) ≈ 20h，贴近 19h。

若仍须压到 ≤19h：C 本就轻量（~4h 可收），B 压到 4h → 9+4+4+3 = 20h，基本贴线；再挤 1h 从 C 的文档里省。

---

## Q3 · 现在 README 里有几个可展示的数字？

- **当前（C 开始前）：0 个真实测量数字。** 此前投入全在搭机器（Phase 0），无产出数字。
- **C 阶段结束时目标：≥3 个真实数字**（来自 agentic eval 实测 + ablation）：

  1. golden-set 总体质量分 **4.33 / 5.0**（n=10，阈值 3.7，实测 run 35008709697）
  2. agentic 臂 **attribution pass rate = 100%**（有 finding 的图全部 ok，0 unresolved / 0 hallucinated）
  3. agentic 臂 **平均检索调用 2.1 次/图**（10 图共 21 次，k 由模型自决）
  4. （ablation 跑完补）fixed / off 两臂的同口径数字，用于对照

> 注：以上 1–3 已是真实运行值（eval artifact `ci-run.json`），C 阶段会在 README `## Evaluation`
> 节正式引用，并**标注样本量 n=10**，不夸大分量。
