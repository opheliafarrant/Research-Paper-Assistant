# 论文助手检索与生成评测

评测日期：2026-11-11。目标是把检索与生成分开检查，提供可重复执行的评测程序、证据标注、原始输出和逐题复核记录。本次没有调用付费 API，也没有修改论文 PDF、FAISS/BM25 索引或会话数据库。

## 样本和边界

- 16 个有答案的问题，来自 4 篇现有论文，每篇 4 个问题，覆盖术语、方法、实验与局限；另有 2 个虚构设备标识的无答案探针。英文与中文问题分别运行。
- `rag_eval_gold_v1.json` 记录问题、期望答案、PDF 物理页码与证据引句；引句已检查存在于指定 PDF 页，且至少能在当前索引的一个完整块中匹配。中文问题对应文本见 `rag_eval_queries_zh_v1.json`。
- 这批样本由单个 AI 助手选择与核对，不是独立专家标注或盲测；问题与论文覆盖有限，相关证据并未穷举。无答案探针包含“虚构”提示，属于容易判断的合成样本。
- 自动匹配按论文 ID 和归一化后的引句文本检查，未验证每次命中的块必须来自标注页，也不把未标注的其他正确证据自动视为错误。引句跨块无法完整匹配时，程序会明确提示标注无法由当前单块表示。

## 一 检索质量

配置：本地多语言 MiniLM、FAISS、BM25 和 RRF；FAISS 默认阈值 0.40、RRF k=60，每路最多取 12 个候选，最终评估前 6 个。主要指标为：

- **标注证据 Hit@6**：每题前 6 块中至少有一块包含某条指定原文引句的题目占比。
- **标注证据覆盖率@6**：每题命中的标注引句数 ÷ 该题标注引句总数，再对题目取平均。
- **标注证据 MRR@6**：每题首个包含指定引句的结果名次取倒数，再取平均；无命中记 0。它不是完整相关性标注下的通用 MRR。

两个无答案探针不纳入这三个指标。以下均为每组 16 个有答案问题，单篇模式限定目标论文：

| 问题语言与模式 | 检索方式 | 标注证据 Hit@6 | 标注证据覆盖率@6 | 标注证据 MRR@6 |
| --- | --- | ---: | ---: | ---: |
| 英文 · 单篇 | Dense | 9/16 | 53.1% | 0.317 |
| 英文 · 单篇 | BM25 | 11/16 | 65.6% | 0.438 |
| 英文 · 单篇 | RRF | 12/16 | 71.9% | 0.565 |
| 中文 · 单篇 | Dense | 8/16 | 46.9% | 0.159 |
| 中文 · 单篇 | BM25 | 2/16 | 12.5% | 0.033 |
| 中文 · 单篇 | RRF | 5/16 | 28.1% | 0.119 |

同一批问题另作全库检索，问题前加入目标论文标题，两个无答案探针不参与全库模式：

| 问题语言与模式 | 检索方式 | 标注证据 Hit@6 | 标注证据覆盖率@6 | 标注证据 MRR@6 |
| --- | --- | ---: | ---: | ---: |
| 英文 · 全库 | Dense | 7/16 | 40.6% | 0.105 |
| 英文 · 全库 | BM25 | 8/16 | 46.9% | 0.323 |
| 英文 · 全库 | RRF | 10/16 | 59.4% | 0.183 |
| 中文 · 全库 | Dense | 6/16 | 34.4% | 0.077 |
| 中文 · 全库 | BM25 | 4/16 | 21.9% | 0.203 |
| 中文 · 全库 | RRF | 3/16 | 15.6% | 0.083 |

全库问题加入了论文标题，和单篇问题文本不同；不应将两表差异完全归因于检索范围。英文、中文各自问题措辞也不同。当前中文样本中 RRF 标注命中低于 Dense，提示需要检查中文词项匹配、融合排序和阈值，但不能据此断言某一算法在其他数据集上始终更好。

### 对实际返回片段的单次逐条审阅

英文与中文单篇 RRF 对 16 个有答案问题各返回 6 块。同一个 AI 助手依据“是否直接回答问题的一部分”分别逐条检查两组各 96 块：英文 36 块标为相关，**本次审阅 P@6 = 36/96 = 37.5%，至少一块相关的题目为 15/16**；中文 27 块标为相关，**本次审阅 P@6 = 27/96 = 28.1%，至少一块相关的题目为 12/16**。判断与理由分别保存在 `rag_quality_review_v1.json` 和 `rag_quality_review_zh_v1.json`；评分可用 `score_rag_quality_review.py` 复算。

英文的 15/16 与上面的标注引句 Hit@6 为 12/16 不矛盾：部分返回块提供了未列入引句标注的替代证据。中文同理，单次相关性审阅得到的 12/16 与指定引句命中的 5/16 是不同口径。相关性审阅并非独立人工专家标注，判断带有主观性，且与标注引句指标不同，不应把两者混为一个“准确率”。

## 二 生成质量

对英文和中文单篇模式分别执行 16 个有答案问题与 2 个无答案探针，使用本地 `qwen2.5:3b`。生成调用走项目现有 `answer()`；完整回答和当次上下文分别保存在英文 `rag_quality_results_v2.json` 和中文 `rag_quality_results_zh_v1.json`。

以下是一个 AI 助手逐题对照期望答案及实际上下文的审阅，**不是独立人工评分**。两个维度分别统计：

- 覆盖：相对于题目列出的预期事实，分为完整、部分、未覆盖；不代表穷举了该论文的所有正确答案。
- 证据支持：回答中的具体主张是否受实际提供的上下文支持；有答案问题若表示上下文不足，单独计为“未作实质回答”。覆盖完整和包含额外无依据主张可以同时发生。

| 模式 | 完整覆盖 | 部分覆盖 | 未覆盖 | 有实质陈述且受上下文支持 | 含上下文不支持的主张 | 需复核 | 有答案题未作实质回答 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 英文单篇，16 题 | 10 | 4 | 2 | 12 | 3 | 1 | 0 |
| 中文单篇，16 题 | 4 | 6 | 6 | 6 | 6 | 1 | 3 |

英文和中文各有 2 个虚构标识无答案探针；本次四条回答均表示给定材料无法确认。由于题目明示“虚构”或 `fictitious`，这只能说明本次简单探针的输出，不能代表一般拒答可靠性。

代表案例：

- 英文 `flow-method`：正确证据进入上下文，但回答没有说出“条件流匹配”，而给出不完整公式；说明检索命中不等于回答覆盖。
- 中文 `grasp-critic`：回答正确提到 critic 蒸馏为效用网络，却把蒸馏说成发生在推理阶段；上下文区分训练后蒸馏与推理时抓取选择。
- 中文 `quant-fluctuation`：回答提到正确的约 20% 波动，但写 PDF 第 3 页；相关块起始页为 3，事实原文位于 PDF 第 4 页，体现跨页起始页码的引用定位问题。
- 中文 `quant-limits`：包含所需内容的块 87 已进入上下文，但回答说论文没有提出具体问题。

英文在当前环境重跑一次后，18 条回答与第一次逐字一致；这只证明这两次具体运行相同，不保证其他环境或后续运行确定。英文与中文各只有一次逐题审阅；模型参数、提示词、问题语言和判分主观性都影响结果。

## 三 使用方式与产物

在项目根目录运行，输出路径必须是**不存在的新文件**，脚本以独占创建模式写入，避免覆盖历史结果：

```powershell
.\.venv\Scripts\python.exe -B -m evaluation.rag_quality_eval --mode retrieval --scope single --output evaluation/新的检索结果.json
.\.venv\Scripts\python.exe -B -m evaluation.rag_quality_eval --mode retrieval --scope library --language zh --output evaluation/新的全库中文结果.json
.\.venv\Scripts\python.exe -B -m evaluation.rag_quality_eval --mode all --language zh --output evaluation/新的中文生成结果.json
.\.venv\Scripts\python.exe -B -m evaluation.score_rag_quality_review
.\.venv\Scripts\python.exe -B -m evaluation.score_rag_quality_review --review evaluation/rag_quality_review_zh_v1.json
```

`--mode all` 会调用本地 Ollama；可用 `--ollama-url http://127.0.0.1:端口` 指定隔离的本地服务。生成阶段禁止非本地 HTTP 端点。服务不可用时逐条记录失败并返回非零状态，不能把异常当作错误回答或拒答。

`rag_quality_results_v1.json` 是首次运行时本地服务未启动的失败记录；`rag_quality_results_v2.json` 是成功的英文单篇评测与英文审阅对应文件；`rag_quality_results_v3.json` 是英文重复运行；`rag_quality_results_zh_v1.json` 是中文单篇评测。全库使用 `rag_retrieval_library_v3.json` 和 `rag_retrieval_library_zh_v1.json`。早期全库 v1/v2 是保留的运行记录；以 v3 为英文全库当前报告。结果记录输入文件哈希，较新的运行还记录运行时版本和代码文件哈希；早期 v2 没有完整代码哈希，不能据此重建精确运行时快照。

时间字段为单次测量：检索细分不包括模型加载和问题向量化；生成调用耗时包括该次问答检索与模型生成。它们不是 P95，也不是产品延迟保证。

## 四 下一步（历史待办）

1. 请具有论文背景的人工审阅者复核证据、96 个英文片段标签和全部答案判断；增加不同论文、跨页表格、语义改写及自然的无答案问题，并保留未参与调参的独立测试集。
2. 排查中文问题的词项匹配与 RRF 排序，以及正确证据已召回但回答遗漏或否认的情况。对任何改动用同一批独立样本比较，不预设必然提高。
3. 对块起始页与实际证据页不一致的情况建立字符或段落级页面映射；避免把页码正确率与检索命中混为一谈。
4. 历史索引重复仍存在；当前检索阶段去重不改变 BM25 的底层语料统计。索引治理应先在隔离副本中验证，不直接覆盖当前资料库。

---

## 五 2026-11-11 六项质量改进后对照与独立（Held-Out）评测

针对第四节提出的四点问题及前序基线暴露的六类缺陷，本轮已在正式代码与生产索引中落地修复，并在本地隔离环境复测：

1. **中文跨语言检索打通**：在 [`research.py`](../research_assistant/application/research.py) 实现 `_translate_query_for_search`、`_clean_translation`、`_acceptable_translation` 与统一 `retrieve_chunks`，将中文检索词翻译为英文检索语句（拒绝 CamelCase、长连字符串联等畸形输出），再执行 FAISS + BM25 + RRF。
2. **模板噪声过滤与弱尾部剪枝**：在 [`fusion.py`](../research_assistant/domain/fusion.py) 实现 `is_boilerplate_chunk`（过滤 NeurIPS Checklist / 伦理与复现规范条目）、`filter_bm25_by_cosine`（`min_cosine=0.30`）与 `prune_weak_tail`（保留有效双路交集或高余弦 `single_path_min_cosine=0.48` 片段）。
3. **生成引用约束与确定性**：在 [`research.py`](../research_assistant/application/research.py) 的 `answer` 提示词中移除硬编码示例论文/页码，强调按 Excerpt 原文 `(Paper ID, Page)` 对应作答；在 [`ollama.py`](../research_assistant/infrastructure/ollama.py) 为 `temperature == 0.0` 注入 `seed=42`；在 [`rag_quality_eval.py`](../evaluation/rag_quality_eval.py) 单次 `run()` 内缓存检索翻译结果，消除 GPU 浮点非确定性导致的检索-生成上下文不一致（`mismatch=[]`）。
4. **跨页字符级页码映射**：在 [`pdf.py`](../research_assistant/infrastructure/pdf.py) 写入 `page_spans` 与 `page_end`，配合 [`citations.py`](../research_assistant/domain/citations.py) 在上下文内插入 `[Page N]` 标记并在 [`source_panel.py`](../research_assistant/presentation/source_panel.py) 提供续页跳转。以 `2611.09868` 块 `24`（`quant-fluctuation`）为例，字符 `0..81` 在第 3 页、核心结论字符 `81..655` 在第 4 页，`page=3, page_end=4`，改进后中文回答可准确标出 `(Paper ID: 2611.09868, p.3-4)`。
5. **索引原子去重重建**：通过 [`indexes.py`](../research_assistant/infrastructure/indexes.py) 的 `rebuild_deduplicated_indexes` 与 `commit_indexes`（带 `manifest.json` 状态与校验和恢复），将生产 `data/cache/` 从历史重复的 `3,596` 条去重并原子提交为 `2,336` 条（`16` 篇论文，`manifest.json` 版本 `c2625def7488`，`status=COMMITTED`）。
6. **独立测试集与自然不可答探针**：新增 [`rag_eval_holdout_v1.json`](../evaluation/rag_eval_holdout_v1.json) 与 [`rag_eval_holdout_queries_zh_v1.json`](../evaluation/rag_eval_holdout_queries_zh_v1.json)，涵盖原开发集未使用的 4 篇馆藏论文（`2611.09847`、`2611.09878`、`2611.09882`、`2611.09803`）共 8 道第 1 页锚定可答问题，以及 4 道领域合理的自然不可答探针（Kubernetes/EC2、ImageNet ResNet-50、AMASS TensorRT、LoRA/QLoRA，不含“虚构/fictitious”提示词）。调参未接触该独立集。

### 5.1 开发集前后对比：单篇（n=16 可答 + 2 探针）与全库检索

> 口径提示：在 `rag_quality_eval.py` 中文模式下，`dense` 与 `bm25` 独立对照轨直接使用原始中文问题（未翻译），而 `rrf` 调用生产 `research.retrieve_chunks`（内部执行中文到英文检索词翻译后再融合）。

| 评测范围 | 版本与语言 | Dense Hit@6 | BM25 Hit@6 | RRF 指定引句 Hit@6 | RRF Recall@6 | RRF MRR@6 | 结果快照（SHA-256 前 12 位） |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 单篇开发集（n=16） | 英文改前 | 9/16 (0.5625) | 11/16 (0.6875) | 12/16 (0.7500) | 0.7188 | 0.5646 | `rag_quality_results_v3.json` (`31871a06eb52`) |
| 单篇开发集（n=16） | 英文改后 | 9/16 (0.5625) | 10/16 (0.6250) | 12/16 (0.7500) | 0.7188 | 0.5625 | `rag_quality_results_en_after_v2.json` (`c87b6df7d2d2`) |
| 单篇开发集（n=16） | 中文改前 | 8/16 (0.5000) | 2/16 (0.1250) | 5/16 (0.3125) | 0.2812 | 0.1187 | `rag_quality_results_zh_v1.json` (`11623d4a90a9`) |
| 单篇开发集（n=16） | 中文改后 | 8/16 (0.5000) | 2/16 (0.1250) | **11/16 (0.6875)** | **0.6562** | **0.3323** | `rag_quality_results_zh_after_v2.json` (`a98fee203581`) |
| 全库开发集（n=16） | 英文改前 | 7/16 (0.4375) | 8/16 (0.5000) | 10/16 (0.6250) | 0.5938 | 0.1833 | `rag_retrieval_library_v3.json` |
| 全库开发集（n=16） | 英文改后 | 7/16 (0.4375) | 8/16 (0.5000) | 9/16 (0.5625) | 0.5312 | 0.1729 | `rag_retrieval_library_en_after_v1.json` |
| 全库开发集（n=16） | 中文改前 | 6/16 (0.3750) | 4/16 (0.2500) | 3/16 (0.1875) | 0.1562 | 0.0833 | `rag_retrieval_library_zh_v1.json` |
| 全库开发集（n=16） | 中文改后 | 6/16 (0.3750) | 4/16 (0.2500) | **8/16 (0.5000)** | **0.4688** | **0.1896** | `rag_retrieval_library_zh_after_v1.json` |

- **全库英文改后波动说明**：`rag_retrieval_library_en_after_v1.json` 仅在 `amp-trials` 一题从第 6 位（chunk `35`）移出，使指定引句 Hit@6 由 10/16 变为 9/16；这是生产索引从 3,596 条去重至 2,336 条后全库 BM25 IDF 重算与尾部截断共同作用的结果。
- **Checklist 干扰改善案例**：中文局限题 `quant-limits` 改前前 6 块中包含 5 个 Checklist/指南块、核心块 `87` 仅排第 5（`[145, 167, 168, 140, 87, 155]`）；改后块 `87` 升至第 1（`[87, 165, 1, 138]`）；英文同题则由 `[87, 12, 135, 131, 134, 137]` 收敛为 3 个纯净正文块 `[87, 12, 58]`。中文局限类整体 RRF 指定引句 Hit@6 从 1/4（0.25）升至 **4/4（1.00）**。

### 5.2 开发集逐块与生成人工审阅前后对比（单 AI 审阅）

| 指标 | 英文改前 | 英文改后 | 中文改前 | 中文改后 |
| :--- | :--- | :--- | :--- | :--- |
| 审阅相关片段 / 实际返回总数 | 36/96 (37.5%) | 36/83 (43.4%) | 27/96 (28.1%) | **41/84 (48.8%)** |
| `score_rag_quality_review` P@6（固定分母 6） | 0.3750 | 0.3750 | 0.2812 | **0.4271** |
| 审阅 Hit@6（至少含一条相关块） | 15/16 (93.75%) | 15/16 (93.75%) | 12/16 (75.0%) | **15/16 (93.75%)** |
| 事实完整覆盖（full / partial / none） | 10 / 4 / 2 | **12 / 3 / 1** | 4 / 6 / 6 | **10 / 5 / 1** |
| 可答却错误拒答（abstained on answerable） | 0/16 | 1/16（`grasp-future` 缺失块 `78`） | 3/16 | **0/16** |
| 严格支持率（supported / unsupported / uncertain / abstained） | 12 / 3 / 1 / 0 | 9 / 6 / 0 / 1 | 6 / 6 / 1 / 3 | 4 / 12 / 0 / 0 |
| 简单虚构探针正确拒答 | 2/2 | 2/2 | 2/2 | 2/2 |

> **重要发现（覆盖率上升与严格引用约束下的错误暴露）**：改后英文全覆盖从 10 升至 12，中文全覆盖从 4 升至 10 且消除全部可答拒答；但由于新版提示词强制要求每条结论对应 Excerpt 与 `(Paper ID, Page)`，本地 3B 模型（`qwen2.5:3b`）在多片段综合或中英数字对齐时频繁出现页码错配、Excerpt 编号混用或误读文献/表格（如中文 `flow-datasets` 把表内数字 `797` 当作 `p. 797` 并把参考文献 `[4]` PeakVI 误当成 BMMC；`grasp-reward` 把阈值 `ε_p` 误写成“小于零”；`amp-limits` 多处 Excerpt 与页码交叉错位），导致严格口径下的 `supported` 比例下降（英文 12→9，中文 6→4）。这说明 3B 小模型在严格引用任务上仍存在显著瓶颈，需通过更强模型或引用后校验解决，而非隐藏问题。

### 5.3 独立集（Held-Out）与自然不可答探针实测

独立集包含 4 篇未参与开发调参的论文（`n=8` 可答 + `n=4` 自然不可答探针），单篇检索+生成结果如下：

| 独立集语言与快照 | Dense Hit@6 | BM25 Hit@6 | RRF 指定引句 Hit@6 | 审阅相关/返回 | 审阅 Hit@6 | 覆盖（full / partial / none） | 严格支持（supported / unsupported） | 自然不可答探针 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 英文（`rag_quality_holdout_en_v1.json`, `e3f7df466eae`） | 4/8 (0.5000) | 7/8 (0.8750) | 4/8 (0.5000) | 25/43 (58.1%, 固定P@6=0.5208) | **8/8 (100%)** | 7 / 1 / 0 | 5 / 3 | **3/4** |
| 中文（`rag_quality_holdout_zh_v1.json`, `7bbb6d093386`） | 5/8 (0.6250) | 1/8 (0.1250) | 3/8 (0.3750) | 22/40 (55.0%, 固定P@6=0.4583) | **8/8 (100%)** | 6 / 2 / 0 | 4 / 4 | **4/4** |

- **独立集自动化引句命中低于单路基线的原因定位**：
  1. 英文独立集所有 8 条黄金引句锚定在第 1 页摘要/引言；`prune_weak_tail`（`single_path_min_cosine=0.48`）截掉了第 6 位余弦 `0.355` 的摘要块（如 `graded-blur` 块 1），使自动化引句未中，但 RRF 返回的正文/实验表块（138、165）仍完整回答了问题（人工审阅 Hit@6 为 8/8）。本轮严格遵守不在测试集反调阈值的原则，未在评测后修改 `0.48` 参数。
  2. 中文独立集中，部分问题由同义正文块而非第 1 页精确引句回答：`cua-interface` 命中第 1 页块 `4`（含 `screenshots, mouse clicks, and keyboard inputs` 并完整答对），`flowmop-spline` 命中方法 §3.1 与附录 B 块 `22/25/98`，但单串子串匹配脚本记为 `0`；此外 `graded-blur` 将“未知模糊”译为 `"unknown fuzzy condition"`，造成翻译查询语义偏移（`faiss_count=0`，仅靠 BM25 命中块 `138` 表 4）。
- **独立集生成与自然拒答残留缺陷（诚实披露）**：
  - **英文自然探针 `noanswer-nat-graded` 失败**：尽管问题询问的是论文未实验的 `ImageNet-1K` 与 `ResNet-50`，模型仍将块 `130`（p. 18）中的 `MNIST-Scale` 准确率 `86.96%` 虚构并归属为 `ImageNet-1K / ResNet-50` 验证准确率。中文版该题正确拒答（`4/4`）。
  - **中文 `flowmop-outputs` 退化重复循环**：在 `temperature=0.0, seed=42` 下输出长达 `63,659` 字符的重复句子，并错引第 3 页而非第 1–2 页。
  - **页面错引与阶段混淆**：`laya-order`（EN）、`laya-shuffle`（EN/ZH）、`flowmop-outputs`（EN/ZH）、`graded-accuracy`（ZH）出现 Excerpt 页码错指或将架构方案误称“基线”的情况。
  - **剩余不可过滤清单文本**：`quant-limits` 中文仍召回不含明确清单标题关键字的指南续块 `138`（p. 19）与 `165`（p. 22），不过主证据块 `87` 已稳居首位。

### 5.4 当前盲区与后续方向

1. 本次全部人工片段与答案判断仍由单一 AI 审阅完成（`rag_quality_review_en_after_v2.json`、`rag_quality_review_zh_after_v2.json`、`rag_quality_holdout_review_en_v1.json`、`rag_quality_holdout_review_zh_v1.json`），尚缺独立人类领域专家双盲复核。
2. 独立集虽使用 4 篇未调参论文，但仍源自同一 16 篇本地论文库且全部可答引句位于第 1 页，尚不足以代表正文深处跨页表格、数学公式和全新领域论文的泛化上限。
3. 后续如继续优化，重点应放在：（a）为翻译层增加领域术语表或双语对照校验以减少“模糊->fuzzy”偏移；（b）为 3B 生成模型补充引用片段-页码一致性后校验或升级更大参数量本地模型，并检测重复生成提前截断；（c）构建真正独立的新论文全文档测试集。
4. **评测后真实界面回归补丁说明**：在收尾运行 [`verify_live_ui.py`](../tests/verify_live_ui.py) 时发现，单篇精读模式下的通用英文元问题（`"What method does the paper propose?"` 针对 `2611.09892`）因缺少专有名词，稠密分数最高为 `0.3701`（低于 `0.40`）且 BM25 命中片段余弦均 `<= 0.2558`（被 `min_cosine=0.30` 过滤），原仅对中文问题触发 `min_score=0.30` FAISS 回退导致单篇英文返回空片段。已在 [`research.py`](../research_assistant/application/research.py) 放宽为 `if not fused and (_HAS_CJK_RE.search(query) or paper_id is not None):`（全库英文检索仍维持 `0.40` 默认阈值，历史评测脚本 [`rag_quality_eval.py`](../evaluation/rag_quality_eval.py) 未改），并在 [`test_retrieval_pipeline.py`](../tests/test_retrieval_pipeline.py) 增补单测；核查全部 6 份已保存评测结果文件（共 `92` 次检索，含 16+16 英文/中文开发集、8+8 英文/中文独立集、16+16 全库检索及 12 道不可答探针）确认无任何一题原先出现 `rrf == []`，因此上表 5.1–5.3 已记录评测结果不受该回退分支影响（仅 [`research.py`](../research_assistant/application/research.py) 文件 SHA-256 更新）。

### 5.5 本轮复现命令

```powershell
.\.venv\Scripts\python.exe -B -m evaluation.rag_quality_eval --mode all --language en --ollama-url http://127.0.0.1:11435 --output evaluation/rag_quality_results_en_after_v2.json
.\.venv\Scripts\python.exe -B -m evaluation.rag_quality_eval --mode all --language zh --ollama-url http://127.0.0.1:11435 --output evaluation/rag_quality_results_zh_after_v2.json
.\.venv\Scripts\python.exe -B -m evaluation.rag_quality_eval --dataset evaluation/rag_eval_holdout_v1.json --mode all --language en --ollama-url http://127.0.0.1:11435 --output evaluation/rag_quality_holdout_en_v1.json
.\.venv\Scripts\python.exe -B -m evaluation.rag_quality_eval --dataset evaluation/rag_eval_holdout_v1.json --zh-queries evaluation/rag_eval_holdout_queries_zh_v1.json --mode all --language zh --ollama-url http://127.0.0.1:11435 --output evaluation/rag_quality_holdout_zh_v1.json
.\.venv\Scripts\python.exe -B -m evaluation.rag_quality_eval --mode retrieval --scope library --language en --output evaluation/rag_retrieval_library_en_after_v1.json
.\.venv\Scripts\python.exe -B -m evaluation.rag_quality_eval --mode retrieval --scope library --language zh --ollama-url http://127.0.0.1:11435 --output evaluation/rag_retrieval_library_zh_after_v1.json
.\.venv\Scripts\python.exe -B -m evaluation.score_rag_quality_review --review evaluation/rag_quality_review_en_after_v2.json
.\.venv\Scripts\python.exe -B -m evaluation.score_rag_quality_review --review evaluation/rag_quality_review_zh_after_v2.json
.\.venv\Scripts\python.exe -B -m evaluation.score_rag_quality_review --review evaluation/rag_quality_holdout_review_en_v1.json
.\.venv\Scripts\python.exe -B -m evaluation.score_rag_quality_review --review evaluation/rag_quality_holdout_review_zh_v1.json
```

