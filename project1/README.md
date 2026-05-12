# Enterprise-RAG — 企业知识库 RAG 问答系统

基于 RAG（检索增强生成）架构的企业级知识库问答系统，支持多格式文档理解、混合检索、私有化模型部署。

## 架构图

```
┌─────────────────────────────────────────────────────────┐
│                    用户入口                              │
│         Streamlit UI (:8501)  │  REST API (:8080)       │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                  FastAPI Gateway                         │
│          /api/query  /api/query/stream                   │
│          /api/documents  /api/conversations              │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                  RAG Pipeline                            │
│  ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Loader  │→ │Splitter │→ │ Embedder │→ │Retriever │  │
│  │多格式解析│  │文本分割  │  │BGE-M3向量│  │混合检索   │  │
│  └─────────┘  └─────────┘  └──────────┘  └──────────┘  │
│                                                     │   │
│  ┌──────────┐                                       │   │
│  │Generator │← Cot Prompt + 防幻觉 + 历史上下文 ─────┘   │
│  │Qwen2.5-7B│                                           │
│  └──────────┘                                           │
└─────────────────────────────────────────────────────────┘
         │                    │
┌────────▼──────┐  ┌─────────▼────────┐
│   Milvus      │  │  vLLM Inference  │
│ Vector Store  │  │  Qwen2.5-7B      │
│   :19530      │  │     :8000        │
└───────────────┘  └──────────────────┘
```

## 快速开始

### 环境要求

- Python 3.10+
- GPU（推荐）：NVIDIA A10/A100/RTX 4090（24G+ VRAM）
- 或 CPU-only 模式（使用 llama.cpp 量化模型）

### 1. 安装依赖

```bash
cd E:\pythonProject\ai_agent\api_use\project1
pip install -r requirements.txt
```

### 2. 配置

编辑 `config.yaml` 设置模型和数据库连接：

```yaml
llm:
  provider: "local"          # local / openai
  api_base: "http://localhost:8000/v1"
vector_db:
  backend: "milvus"          # milvus / chroma / qdrant
```

### 3. 启动模型服务

```bash
# 方式A: vLLM（推荐，需GPU）
bash scripts/deploy_qwen.sh

# 方式B: llama.cpp（CPU/轻量GPU）
bash scripts/deploy_qwen_llamacpp.sh

# 方式C: OpenAI API 降级
# 修改 config.yaml: llm.provider = "openai"
```

### 4. 启动服务

```bash
# FastAPI 后端（:8080）
python -m uvicorn api.main:app --host 0.0.0.0 --port 8080

# Streamlit 前端（:8501）
streamlit run app.py

# Docker 一键部署
docker-compose up -d
```

### 5. 访问

- 前端界面：http://localhost:8501
- API 文档：http://localhost:8080/docs
- 健康检查：http://localhost:8080/health

## 项目结构

```
project1/
├── app.py                  # Streamlit 前端
├── evaluate.py             # Ragas 评测脚本
├── config.yaml             # 全局配置
├── requirements.txt        # Python 依赖
├── Dockerfile              # 应用镜像
├── docker-compose.yml      # 服务编排
├── README.md               # 项目文档
├── docs/
│   └── design.md           # 技术设计文档
├── src/
│   ├── config.py           # 配置加载器
│   ├── loader.py           # 文档加载（PDF/DOCX/XLSX/图片...）
│   ├── splitter.py         # 文本分割（递归+语义分块）
│   ├── embedder.py         # BGE-M3 向量化+BM25 索引
│   ├── retriever.py        # 混合检索+Reranker
│   ├── generator.py        # LLM 生成（CoT+防幻觉）
│   └── pipeline.py         # 端到端 RAG 管道
├── api/
│   └── main.py             # FastAPI 后端
├── scripts/
│   ├── deploy_qwen.sh      # vLLM 部署脚本
│   └── deploy_qwen_llamacpp.sh  # llama.cpp 部署脚本
├── data/
│   ├── uploads/            # 上传文件目录
│   └── demo/               # 演示数据集（5篇文档）
└── tests/
```

## API 文档

### `POST /api/query`

执行 RAG 问答。

```json
{
    "question": "公司的年假政策是什么？",
    "top_k": 5,
    "use_cot": true,
    "conversation_id": null
}
```

响应：
```json
{
    "answer": "根据《员工手册》第二章...",
    "reasoning": "用户询问年假政策...",
    "contexts": [
        {
            "text": "员工每年享有15天带薪年假...",
            "metadata": {"source": "员工手册_2024.md", "page": 2},
            "score": 0.92
        }
    ],
    "question": "公司的年假政策是什么？"
}
```

### `POST /api/query/stream`

流式问答（SSE）。

### `POST /api/documents/upload`

上传文档（multipart/form-data）。

### `GET /api/documents`

列出已入库文档。

### `DELETE /api/documents/{name}`

删除文档。

## 向量数据库切换

在 `config.yaml` 中修改 `vector_db.backend`：

```yaml
# Milvus (默认)
vector_db:
  backend: "milvus"
  milvus:
    host: "localhost"
    port: 19530

# 切换至 Chroma（零配置本地部署）
vector_db:
  backend: "chroma"
  chroma:
    persist_directory: "./data/chroma_db"

# 切换至 Qdrant
vector_db:
  backend: "qdrant"
  qdrant:
    url: "http://localhost:6333"
```

## 评测

```bash
# 运行完整评测
python evaluate.py

# 仅生成测试问题
python evaluate.py --questions

# 使用缓存结果
python evaluate.py --skip-gen
```

评测指标：
| 指标 | 目标值 | 说明 |
|------|--------|------|
| Faithfulness | ≥ 0.90 | 答案对上下文的忠实度 |
| Answer Relevancy | ≥ 0.90 | 答案与问题的相关性 |
| Context Precision | - | 检索上下文精度 |
| Context Recall | - | 检索上下文召回率 |

评测报告输出至 `eval_report.html`。

## 踩坑记录

### 1. PDF 解析
- **问题**：扫描件 PDF 使用 PyMuPDF `get_text()` 返回空字符串
- **解决**：自动检测文本为空时启用 pytesseract OCR，设置合适 DPI（200）平衡速度与精度

### 2. 中文分块
- **问题**：LangChain 默认分隔符对中文支持差，按句号分割导致语义不连贯
- **解决**：自定义分隔符优先级 `\n\n → \n → 。→ ！→ ？→ 空格`，添加语义分块开关

### 3. 混合检索权重
- **问题**：固定权重（0.5/0.5）在不同场景表现差异大
- **解决**：改为可配置权重，默认 Dense=0.7 / Sparse=0.3，技术文档可调高 BM25 比例

### 4. LLM 幻觉
- **问题**：Qwen2.5-7B 在无答案时容易编造信息
- **解决**：CoT 分步推理 + 防幻觉 Prompt + 置信度阈值

### 5. 大文件处理
- **问题**：100MB+ PDF 解析耗时 2-5 分钟
- **解决**：异步处理 + 进度反馈，单页解析超时 30 秒自动跳过

## 性能瓶颈分析

| 操作 | 小文件 (<1MB) | 中等文件 (10MB) | 大文件 (100MB) |
|------|--------------|----------------|---------------|
| PDF 解析 | <1s | 3-5s | 30-120s |
| 文本分块 | <1s | 1-2s | 5-10s |
| 向量化 (100 chunks) | 2-5s | 5-10s | 15-30s |
| Milvus 索引 | <1s | 2-3s | 10-20s |
| 检索 (Top-20) | <1s | <1s | 1-2s |
| LLM 生成 | 2-5s | 3-8s | 5-15s |

**优化建议**：
- GPU 加速 Embedding（batch_size=32）
- Milvus IVF_FLAT + nlist=128 平衡精度与速度
- vLLM Continuous Batching 提升并发吞吐

## 许可证

MIT License
