# Visual Inspection Report Generator · 操作说明手册

> 一份"跑通即交付"的操作手册：从零安装到三种运行方式，再到 Docker 部署和常见踩坑排查。

## 目录

- [0. 这个项目是什么](#0-这个项目是什么)
- [1. 系统要求](#1-系统要求)
- [2. 第一次安装](#2-第一次安装)
- [3. 三种运行方式速览](#3-三种运行方式速览)
- [4. 方式 A：线性管道 `run_pipeline.py`](#4-方式-a线性管道-run_pipelinepy)
- [5. 方式 B：LangGraph Agent `run_agent.py`](#5-方式-blanggraph-agent-run_agentpy)
- [6. 方式 C：FastAPI 服务（本地 uvicorn）](#6-方式-cfastapi-服务本地-uvicorn)
- [7. 方式 D：Docker 部署](#7-方式-ddocker-部署)
- [8. API 端点参考](#8-api-端点参考)
- [9. 报告输出说明](#9-报告输出说明)
- [10. CLI 参数表](#10-cli-参数表)
- [11. 常见问题排查](#11-常见问题排查)
- [12. 目录结构参考](#12-目录结构参考)

---

## 0. 这个项目是什么

把一张现场照片变成一份带风险评级和引用的检验报告：

```
照片 → YOLOv8 检测 → Qwen-VL 视觉推理 → RAG 检索标准 → Markdown/JSON 报告
                                                ↑
                                          LangGraph 决策编排
```

适用场景举例：建筑工地、街道、车间、港口、装配线等需要"看图+对照标准+出报告"的业务。

---

## 1. 系统要求

| 组件 | 最低版本 | 说明 |
|------|---------|------|
| Python | 3.10+（推荐 3.12） | 旧版 numpy/torch 兼容性问题多 |
| 操作系统 | macOS / Linux | 已验证 macOS Intel |
| 内存 | 8 GB+ | 跑 VLM 期间峰值 ~1–2 GB |
| 磁盘 | 4 GB+ | 含虚拟环境和模型权重 |
| Docker（可选） | Docker Desktop 或 Colima | 服务化部署用 |
| 网络 | 能访问 `dashscope.aliyuncs.com` | VLM 和 Embedding 都走这个 |

API 凭据：阿里云百炼（DashScope）`DASHSCOPE_API_KEY`，开通后到 [百炼控制台](https://bailian.console.aliyun.com/) → API-Key 页面生成。

---

## 2. 第一次安装

```bash
# 1) 克隆
git clone https://github.com/mino19790622-dot/visual-inspection-report-generator.git
cd visual-inspection-report-generator

# 2) 创建虚拟环境（managed Python 3.12 路径示例）
/Users/zhangminghao/.workbuddy/binaries/python/versions/3.12/bin/python3 -m venv .venv
source .venv/bin/activate

# 3) 安装依赖（含 PyTorch / Ultralytics / ONNX）
pip install -r requirements.txt

# 4) 配置 API Key（仅本机，绝不入库）
echo "DASHSCOPE_API_KEY=sk-你的真实key" > .env
```

验证安装：

```bash
python -c "from app.detection.detector import YOLODetector; print('OK')"
```

> 提示：如果你不想装 PyTorch（只跑 FastAPI 服务），请改用 `pip install -r requirements.docker.txt` —— 镜像里也是这套精简依赖。

---

## 3. 三种运行方式速览

| 方式 | 入口 | 适用场景 | 速度 |
|------|------|---------|------|
| **A. 线性管道** | `run_pipeline.py` | 本地跑通、看结果、调试 | 最快，单图 ~5–10s |
| **B. LangGraph Agent** | `run_agent.py` | 想看 Agent 决策日志、自适应重检 | 略慢（可能重检） |
| **C. FastAPI 服务** | `uvicorn app.api.server:app` | 部署给前端/系统调用、Swagger 调试 | 网络 IO + 推理 |

第一次试跑，建议从方式 A 开始。

---

## 4. 方式 A：线性管道 `run_pipeline.py`

最直接：检测 → VLM → RAG → 报告文件。

```bash
# 最简单（用仓库自带样例图）
python run_pipeline.py data/test_images/bus.jpg

# 换成自己的图
python run_pipeline.py /path/to/your/site_photo.jpg

# 自定义参数
python run_pipeline.py your_image.jpg \
  --conf 0.35 \            # 检测置信度阈值
  --k 5 \                  # 检索前 k 条标准
  --save-dir my_reports    # 报告输出目录
```

输出（终端）：

```
============================================================
  Visual Inspection Pipeline (D1-D4)
============================================================
[1/3] Running YOLOv8 detection...
  Detected 5 objects in 87ms
  Counts: {'person': 4, 'bus': 1}
[2/3] Calling Qwen-VL-Max for structured analysis...
  VLM report generated (1247 chars)
[3/3] Retrieving applicable inspection standards...
  Retrieved 3 relevant standards
============================================================
  VLM Inspection Report
============================================================
...
============================================================
  Applicable Standards (RAG Retrieved)
============================================================
--- [1] ISO 45001 ... (relevance: 0.86) ---
...
```

报告文件落地到 `reports/`（详见第 9 节）。

---

## 5. 方式 B：LangGraph Agent `run_agent.py`

在方式 A 的基础上多两层**真实决策逻辑**：

1. **自适应重检**：第一次检测返回 0 个目标时，自动把置信度阈值砍半（最低 0.10）再跑一次。
2. **风险驱动检索深度**：VLM 输出的风险等级（low/medium/high）决定 RAG 检索条数（3/4/5），决策**会写进报告**。

```bash
python run_agent.py data/test_images/bus.jpg
python run_agent.py your.jpg --conf 0.3 --save-dir reports/
```

终末尾会打印决策日志，例如：

```
Decisions:
  - initial detection: 5 objects at conf=0.25
  - risk_level parsed as: high
  - retrieving 5 standards
```

Agent 决策同样会出现在导出报告的 "Agent Decisions" 段落里，方便复盘。

---

## 6. 方式 C：FastAPI 服务（本地 uvicorn）

把项目变成可被前端或脚本调用的 REST API。

```bash
# 启动
uvicorn app.api.server:app --host 0.0.0.0 --port 8000
```

启动后访问：
- Swagger UI（可在线测试）：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/health>

测试 `/inspect`：

```bash
# 最简单：仅传图
curl -F "image=@data/test_images/bus.jpg" http://127.0.0.1:8000/inspect

# 自定义参数
curl -F "image=@data/test_images/bus.jpg" \
     -F "conf=0.3" \
     -F "save=true" \
     http://127.0.0.1:8000/inspect | python -m json.tool
```

返回 JSON：`detection`（检测结果）+ `vlm_report`（视觉报告）+ `risk_level`（风险等级）+ `standards`（引用的标准列表 + 相关度）+ `agent_decisions`（Agent 决策日志）+ `saved_files`（落地文件路径）。

> ⚠️ macOS 注意：用 `127.0.0.1` 而非 `localhost`，避免 uvicorn 绑 IPv4 而 curl 解析到 IPv6。

---

## 7. 方式 D：Docker 部署

适用：环境隔离、CI/CD、给团队共享。

### 7.1 启动 Colima（仅 macOS，没装 Docker Desktop 的情况）

```bash
colima start
docker context use colima
```

### 7.2 一键起服务

```bash
# 确认 .env 文件存在（含 DASHSCOPE_API_KEY）
ls .env   # 必须存在

# 构建镜像 + 后台启动
docker compose up --build -d

# 查看日志（首次构建会拉 python:3.12-slim，约 1–2 分钟）
docker compose logs -f api
# 看到 "Application startup complete." 即可 Ctrl+C 退出日志

# 健康检查
curl http://127.0.0.1:8000/health
# {"status":"ok"}

# 完整跑一次
curl -F "image=@data/test_images/bus.jpg" http://127.0.0.1:8000/inspect
```

### 7.3 容器与数据

- `reports/`、`uploads/` 通过 bind mount 映射到宿主机，**报告/上传图直接落本地目录**
- ChromaDB 向量索引存到 named volume `chroma_index`，**重启/重建不会丢**
- 镜像大小约 **1.78 GB**（已剔除 torch/ultralytics 等构建期依赖）

### 7.4 停止与清理

```bash
docker compose down              # 停服务，保留 volume
docker compose down -v           # 同时删 volume（会丢 ChromaDB 索引）
```

### 7.5 电脑重启后

```bash
colima start                     # Colima 不像 Docker Desktop 会自启
docker compose up -d             # 容器再起
```

---

## 8. API 端点参考

| Method | Path | 说明 |
|--------|------|------|
| `GET` | `/health` | 健康检查，返回 `{"status":"ok"}` |
| `GET` | `/standards` | 列出 RAG 知识库中已索引的检验标准 |
| `POST` | `/inspect` | 上传图，跑 Agent，返回完整报告 JSON |
| `GET` | `/docs` | Swagger UI（FastAPI 自带） |

`POST /inspect` 参数（form-data）：

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `image` | file | 必填 | jpg/png/webp/bmp |
| `conf` | float | 0.25 | 检测置信度 |
| `save` | bool | true | 是否落盘报告文件 |

支持的文件后缀：`.jpg .jpeg .png .webp .bmp`。其它后缀会返回 415。

---

## 9. 报告输出说明

每次运行产出三份文件（带时间戳，覆盖同图旧文件）：

```
reports/
├── bus_20260821_114832.md      # 人读
├── bus_20260821_114832.json    # 机读
└── bus_20260821_114832_det.jpg # 标注图
```

**`*.md` 结构**：

```markdown
# Visual Inspection Report

- Image: data/test_images/bus.jpg
- Generated: 2026-08-21 11:48:32
- Detector: yolov8m (ONNX, 87ms, 5 objects)

## VLM Analysis
> Scene description ...
> Inventory ...
> Detector Gaps ...
> Risk Assessment: high risk ...

## Applicable Standards
1. ISO 45001 — ...
2. BS EN 1992 — ...
3. EN 13134 — ...

## Agent Decisions
- initial detection: 5 objects at conf=0.25
- risk_level parsed as: high
- retrieving 5 standards
```

**`*.json`**：结构化字段，键名与 API 返回一致（`detection`/`vlm_report`/`risk_level`/`standards`/`agent_decisions`）。

**`*_det.jpg`**：原图 + 红色检测框 + 类别 + 置信度。

---

## 10. CLI 参数表

### `run_pipeline.py`

| 参数 | 默认 | 说明 |
|------|------|------|
| `image`（位置参数）| `data/test_images/bus.jpg` | 输入图路径 |
| `--onnx` | `yolov8m.onnx` | ONNX 模型路径 |
| `--conf` | `0.25` | 检测置信度阈值 |
| `--k` | `3` | 检索标准条数 |
| `--save-dir` | `reports` | 报告输出目录 |
| `--no-save` | 关 | 只打印到终端，不写文件 |

### `run_agent.py`

| 参数 | 默认 | 说明 |
|------|------|------|
| `image` | `data/test_images/bus.jpg` | 输入图路径 |
| `--conf` | `0.25` | 检测置信度 |
| `--save-dir` | `reports` | 报告输出目录 |
| `--no-save` | 关 | 只打印到终端 |

### `uvicorn app.api.server:app`

| 参数 | 默认 | 说明 |
|------|------|------|
| `--host` | `127.0.0.1` | 监听地址（容器内用 `0.0.0.0`） |
| `--port` | `8000` | 端口 |
| `--reload` | 关 | 开发模式热重载 |

---

## 11. 常见问题排查

### 11.1 报 `ProxyError` / `TunnelError`，但明明没设代理

shell 环境变量里有遗留代理（公司 VPN / Clash 退出后没清干净），httpx/requests 会拿去用。

```bash
env | grep -i proxy       # 看看
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
# 或者一次性绕过
curl --noproxy '*' http://127.0.0.1:8000/health
```

**根本修法**：去 `~/.zshrc` / `~/.bash_profile` 把这些 export 删掉或注释。

### 11.2 `curl localhost:8000` 被拒，但服务确实在跑

两个原因二选一：

- uvicorn 默认绑 `127.0.0.1`（IPv4），但 `localhost` 解析到 `::1`（IPv6）→ **改用 `127.0.0.1`**
- 上面的代理变量劫持 → 加 `--noproxy '*'`

### 11.3 ChromaDB 显示 `relevance: 0.01` 或负数

向量库创建时没指定距离度量，默认是 L2。**改度量必须删库重建**：

```bash
rm -rf .chroma_db
# 重跑任意一条命令，retriever 会按 cosine 重建
```

代码里建集合时已写 `metadata={"hnsw:space": "cosine"}`，正常情况不会出问题；只在你**手动改过 retriever 或换过模型维度**时需要重建。

### 11.4 VLM 同一张图两次跑结论不一样

早期版本 `temperature=0.2` 会引起漂移（"no hazardous"↔"moderate"）。项目里已经固定 `temperature=0`，**任何复现性问题不要再去调这个**，先检查：
- 是否改了 prompt（prompt 模板在 `app/vlm/client.py`）
- 是否换了模型（默认 `qwen-vl-max`，更高温度）

### 11.5 大图上传导致 VLM 报错

`VLMClient` 已经做了图片自动压缩（>1024px 等比缩小再 base64）。**如果你手动关掉了这个逻辑**，先恢复：

```python
# app/vlm/client.py
img = resize_if_needed(img, max_side=1024)
```

### 11.6 Docker 拉镜像报 `docker-credential-desktop not found`

Docker Desktop 残留配置没清干净：

```bash
# 编辑 ~/.docker/config.json
# 把 "credsStore" 和 "credHelpers" 这两个 key 整段删掉
# 只保留 "experimental": "enabled"（如果有）
```

### 11.7 Colima 启动后 `docker ps` 仍报找不到 daemon

```bash
colima start
docker context use colima
docker ps
```

### 11.8 `pip install torch` 装到一半失败

Intel Mac 不要用最新 torch。**严格按 `requirements.txt` 锁版本**：

```text
torch==2.2.2
numpy==1.26.4
opencv-python<5
```

这三个版本错位就会触发 numpy 2.x ABI 冲突。

### 11.9 报告里 risk_level 永远是 "n/a"

要么 VLM 输出没有标准的 "Risk Assessment" 段，要么解析器在关键词兜底分支挂了。看下报告 .md 内容，正常应当形如：

```
> Risk Assessment: (high|moderate|low) risk ...
```

如果 VLM 输出格式变了，去 `app/agent/graph.py` 的 `parse_risk_level` 函数调正则。

---

## 12. 目录结构参考

```
visual-inspection-report-generator/
├── app/
│   ├── detection/detector.py     # 手写 YOLODetector（ONNX, letterbox, NMS）
│   ├── vlm/client.py             # Qwen-VL-Max 调用 + 图片自动压缩
│   ├── rag/retriever.py          # ChromaDB 集合（cosine）
│   ├── agent/graph.py            # LangGraph StateGraph
│   ├── reporting/exporter.py     # Markdown + JSON + 标注图导出
│   └── api/server.py             # FastAPI
├── data/
│   ├── standards/                # 6 个检验标准 md
│   └── test_images/              # 示例图
├── docs/                         # 文档
│   ├── USER_GUIDE.md             # 英文操作手册
│   └── USER_GUIDE_zh.md          # ← 本文件
├── reports/                      # 报告输出（gitignored）
├── uploads/                      # API 上传暂存
├── .chroma_db/                   # 向量索引（gitignored）
├── run_pipeline.py               # 方式 A
├── run_agent.py                  # 方式 B
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements.docker.txt       # 镜像精简依赖
└── README.md
```

---

## 附：完整流程 30 秒体验

```bash
git clone https://github.com/mino19790622-dot/visual-inspection-report-generator.git
cd visual-inspection-report-generator
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "DASHSCOPE_API_KEY=sk-你的key" > .env
python run_pipeline.py data/test_images/bus.jpg
open reports/   # macOS 直接看刚生成的报告
```

如果想跑 Agent：

```bash
python run_agent.py data/test_images/bus.jpg
```

如果想跑成服务：

```bash
uvicorn app.api.server:app --port 8000 &
open http://127.0.0.1:8000/docs
```

或者直接 Docker：

```bash
docker compose up --build -d
sleep 30 && curl http://127.0.0.1:8000/health
```
