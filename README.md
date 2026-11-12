# 📄 Research Paper Assistant（本地科研论文 RAG 助手）

> 完全基于本地模型运行的 arXiv 论文检索、双语问答、结构化速读与 PDF 原文溯源工作台。  
> A fully local, privacy-first RAG research workspace built with Streamlit, FAISS, BM25, PyMuPDF, and Ollama — supporting bilingual (Chinese/English) Q&A, structured paper reading templates, PDF evidence highlighting, and persistent multi-session management.

详细中文使用与分层架构说明见 [使用说明](docs/使用说明.md) 与 [架构说明](docs/架构说明.md)。

---

## ✨ 核心特性 (Key Features)

1. **三栏式研究工作台（Three-Pane Workspace）**
   - **左侧会话栏**：支持新建、切换、内联重命名、单条删除与重置会话；按“论文问答（全库）”与“单篇精读”区分图标，状态持久化至本地 `data/conversations.sqlite3`（带版本号乐观锁防覆盖）。
   - **中间问答与精读区**：支持全库多论文问答、单篇限定精读，以及 ChatPaper 风格的**结构化速读**一键提炼（综合摘要 / 核心方法 / 实验对比 / 局限与未来）。
   - **右侧资料与来源面板**：按页预览本地 PDF，并在窄屏（≤1150px）自动折行防挤压。
2. **中英跨语言混合检索（Cross-Lingual Hybrid Retrieval）**
   - 中文提问自动翻译为标准英文学术检索词（过滤畸形拼写与冗余停用词），并行检索 **FAISS**（稠密语义）与 **BM25**（精确词法），通过 **RRF**（Reciprocal Rank Fusion）融合。
   - 内置会议 Checklist/参考文献噪声过滤、BM25 余弦一致性门槛、弱尾部剪枝，以及单篇/中文场景回退阈值机制。
3. **可核对的引用溯源与 PDF 高亮（Evidence Grounding & Highlighting）**
   - 基于字符级 `page_spans` 记录跨页片段（如 `第 3-4 页`），上下文内嵌 `[Page N]` 标记。
   - 每条引用附带检索置信度徽章（`高相关` / `中相关` / `参考`），点击“查看原文”或“查看原文页面”可跳转至对应页并在内存中绘制半透明黄色高亮框（不修改磁盘原 PDF）。
4. **原子级双索引持久化（Atomic Index Governance）**
   - 论文解析与入库通过临时目录暂存、SHA-256 校验、`.commit_tx.json` 事务日志与回滚保护，避免中断导致的 FAISS / BM25 不同步或重复写入。

---

## 🏗️ 项目架构 (Architecture)

```text
research_assistant/
├── __main__.py                 # 统一 CLI 入口 (python -m research_assistant)
├── config.py                   # 路径、模型、分块与检索超参数单一配置源
├── bootstrap.py                # 组合根：装配用例与基础设施适配器
├── domain/                     # 纯领域逻辑：数据契约、RRF 与置信度计算、跨页引文映射
├── application/                # 应用用例：论文导入、目录查询、问答检索、结构化速读策略
├── infrastructure/             # 外部适配器：arXiv、PyMuPDF 解析与渲染、Embedding、FAISS/BM25、Ollama、SQLite
└── presentation/               # Streamlit 三栏 UI、会话状态桥接、来源面板与样式
docs/                           # 使用说明与架构文档
evaluation/                     # 检索与生成评测脚本、数据集与评测报告
tests/                          # 单元测试与隔离/联调回归脚本
requirements.txt                # 核心运行依赖
启动助手.cmd                     # Windows 一键启动脚本（可选）
```

---

## 🚀 快速开始 (Setup & Installation)

### 环境要求

- Python 3.10+
- 本地已安装并可运行 [Ollama](https://ollama.com/)

### 1. 克隆仓库并创建虚拟环境

```bash
git clone https://github.com/opheliafarrant/Research-Paper-Assistant.git
cd Research-Paper-Assistant

python -m venv .venv

# Windows (PowerShell / CMD)
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 拉取本地模型

```bash
ollama pull qwen2.5:3b
```

> 默认使用轻量模型 `qwen2.5:3b` 以兼顾普通 CPU/小显存设备；如硬件资源充足，可在 `research_assistant/config.py` 中将 `OLLAMA_MODEL` 切换为 `qwen2.5:7b` 等更大模型以获得更严谨的引用归因表现。  
> 首次运行嵌入检索时，`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` 会自动下载并缓存到本地；若项目根目录存在 `.hf-cache/` 与 `models/`，`config.py` 会自动优先启用本地离线缓存。

### 4. 启动应用

**通用跨平台启动：**

```bash
streamlit run research_assistant/presentation/streamlit_app.py
```

启动后在浏览器访问 `http://localhost:8501`。首次使用可在左侧边栏点击 **“获取新论文”**，系统将按 `research_assistant/config.py` 中的 `ARXIV_TOPICS` 拉取最新论文并构建检索索引。

*(Windows 用户也可按需使用根目录下的 `启动助手.cmd` 启动本地环境。)*

---

## 💻 命令行用法 (CLI)

除 Web 界面外，所有核心管线均可通过包命令直接调用：

```bash
python -m research_assistant list                                  # 查看已收录论文
python -m research_assistant fetch                                 # 预览 arXiv 最新待抓取论文（不写入索引）
python -m research_assistant ingest                                # 拉取新论文并原子更新 FAISS/BM25 索引
python -m research_assistant ask "这篇论文提出了什么核心方法？"        # 全库检索问答
python -m research_assistant ask "What method does it propose?" --paper-id 2611.09892
python -m research_assistant summary 2611.09892                    # 生成单篇结构化摘要
```

---

## ⚙️ 主要配置 (`research_assistant/config.py`)

常用配置项可直接修改 `research_assistant/config.py`，运行时环境变量示例见 [`.env.example`](.env.example)：

| 配置项 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `ARXIV_TOPICS` | `["cs.AI", "cs.LG", "cs.CL"]` | 抓取的 arXiv 学科分类 |
| `MAX_PAPERS_PER_TOPIC` | `5` | 每次每个分类最多抓取的新论文数 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `600` / `80` | 文本分块目标字符数与重叠长度 |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 多语言本地向量模型 |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Ollama 本地生成模型 |
| `TOP_K` / `SCORE_THRESHOLD` / `RRF_K` | `6` / `0.40` / `60` | 检索返回片段数、默认余弦阈值与 RRF 常数 |

---

## 🧪 测试与评测说明 (Testing & Evaluation)

运行单元测试套件：

```bash
python -B -m unittest discover -s tests -p "test_*.py"
```

检索与生成评测脚本、标注集与详细分析见 [RAG检索与生成评测报告](evaluation/RAG检索与生成评测报告.md)：
- **开发集单篇（`n=16`）**：英文 RRF 标注引句 `Hit@6 = 12/16 (0.7500)`、`MRR@6 = 0.5625`；引入中文检索翻译与噪声过滤后，中文 RRF `Hit@6` 从 `5/16 (0.3125)` 提升至 `11/16 (0.6875)`、`MRR@6` 从 `0.1187` 提升至 `0.3323`。
- **独立集（Held-Out，`n=8` 可答 + `4` 自然不可答探针）**：人工审阅检索 `Hit@6` 中英文均达 `8/8`，自然不可答拒答率为英文 `3/4`、中文 `4/4`。
- **已知模型边界**：在 `qwen2.5:3b` 默认配置下，模型在多片段综合时仍可能出现页码/片段编号错配，以及极少数相似数值张冠李戴问题，生成结果建议结合右侧 PDF 原文预览核对。

---

## 🔒 隐私与开源许可 (Privacy & License)

- **本地数据**：下载的论文 PDF 保存在 `data/pdfs/`，向量与词法索引保存在 `data/cache/`，对话记录保存在 `data/conversations.sqlite3`，生成与向量推理全部在本地完成；仅在点击抓取论文时访问 arXiv API。
- **上游致谢**：本项目早期雏形基于 MIT 协议的开源项目 [Hagar-Ghazi/Research-Paper-Assistant](https://github.com/Hagar-Ghazi/Research-Paper-Assistant)（commit `9e8f0f2`），在此基础上完成了分层架构重构、中文跨语言检索、跨页引文与 PDF 高亮、结构化速读、原子索引事务与 SQLite 会话持久化。
- **License**：基于 [MIT License](LICENSE) 开源。
