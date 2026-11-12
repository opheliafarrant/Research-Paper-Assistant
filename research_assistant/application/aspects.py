"""
Aspect strategies and context retrieval for paper summarisation and deep-dive extractions.
Provides structured strategies for methods, experiments, limitations, and general summary.
"""

from dataclasses import dataclass, field
import logging
from typing import Optional

from research_assistant import config
from research_assistant.domain.contracts import ResearchDependencies
from research_assistant.domain.fusion import rrf_fuse as _rrf_fuse

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AspectStrategy:
    aspect_id: str
    display_title: str
    dense_query: str
    sparse_query: str
    system_prompt: str
    prompt_header: str
    prompt_requirements: list[str] = field(default_factory=list)

    def build_prompt(self, title: str, paper_id: str, context: str) -> str:
        parts = [
            f"Paper title: {title}",
            f"Paper ID:    {paper_id}\n",
            f"Excerpts:\n{context}\n",
            self.prompt_header,
        ]
        if self.prompt_requirements:
            req_lines = "\n".join(
                f"{i}. {req}" for i, req in enumerate(self.prompt_requirements, start=1)
            )
            parts.append(req_lines)
            parts.append("注意：必须严格忠实于所给片段事实，切勿臆测或套用通用泛化模板。")
        return "\n".join(parts)


_ASPECT_REGISTRY: dict[str, AspectStrategy] = {
    "methods": AspectStrategy(
        aspect_id="methods",
        display_title="核心方法与算法架构",
        dense_query="methodology architecture model algorithm framework approach pipeline design 核心方法 算法 架构",
        sparse_query="method methodology architecture model algorithm framework approach pipeline formulation",
        system_prompt=(
            "You are a research assistant. Extract and explain the core methodology and technical framework of the paper.\n"
            "Focus on:\n"
            "  • Core Concept – what is the primary technical idea or philosophy?\n"
            "  • Architecture / Algorithm – how does the method work step-by-step?\n"
            "  • Key Innovations – what makes this approach unique compared to previous work?\n"
            "Write the response in Chinese with clear bullet points and bold keywords. Be precise and technical."
        ),
        prompt_header="请依据上述论文片段，深度提炼并以中文阐述该论文的【核心方法与算法架构】。\n请包含：",
        prompt_requirements=[
            "核心思路与建模逻辑；",
            "关键算法/模块实现细节；",
            "主要技术创新点。使用清晰的要点列出。",
        ],
    ),
    "experiments": AspectStrategy(
        aspect_id="experiments",
        display_title="实验评估与对比分析",
        dense_query="experiments evaluation benchmark baseline dataset metrics results comparison 实验 评估 基准",
        sparse_query="experiments evaluation benchmark baseline dataset metrics results comparison ablation",
        system_prompt=(
            "You are a research assistant. Extract and summarise the experimental setup, evaluation benchmarks, and empirical results.\n"
            "Focus on:\n"
            "  • Datasets & Tasks – which benchmarks or datasets were evaluated?\n"
            "  • Baseline Comparisons – which models or baselines were compared against?\n"
            "  • Key Findings & Metrics – what are the quantitative performance gains or conclusions?\n"
            "Write the response in Chinese with clear bullet points. Be factual."
        ),
        prompt_header="请依据上述论文片段，提炼并以中文阐述该论文的【实验评估与对比分析】。\n请包含：",
        prompt_requirements=[
            "评估使用的数据集/任务场景；",
            "对比的基准模型（Baselines）；",
            "关键量化指标与核心实验结论。使用清晰的要点列出。",
        ],
    ),
    "limitations": AspectStrategy(
        aspect_id="limitations",
        display_title="研究局限性与未来方向",
        dense_query="limitations, open questions, future work, discussion, outlook, weaknesses, drawbacks 局限性 讨论 不足 未来方向",
        sparse_query="limitations limitation outlook future work discussion open questions weaknesses",
        system_prompt=(
            "You are an academic research assistant.\n"
            "Your task is to objectively extract all limitations, failure boundaries, and future directions explicitly stated in the provided paper excerpts.\n"
            "Guidelines:\n"
            "- Ground every bullet point strictly on the provided text; do not hallucinate.\n"
            "- Enumerate each distinct limitation and future work item explicitly mentioned by the authors.\n"
            "- Write in Chinese with clear section headers and bullet points.\n"
        ),
        prompt_header="请依据上述论文片段中作者在讨论与展望（Outlook / Discussion / Limitations）中的具体陈述，以中文完整提炼出【研究局限性与未来方向】。\n请严格分两部分阐述：",
        prompt_requirements=[
            "【核心局限与失效边界】逐项列出作者在原文中明确指出的方法局限、假设前提、延迟瓶颈或适用边界；",
            "【未来研究方向】逐项列出作者在展望中明确提出的具体技术改进设想与后续探索路径；",
            "要求：必须完整提取原文事实并保留作者提及的具体技术概念，严禁遗漏要点，严禁臆测或套用通用泛化模板。",
        ],
    ),
    "summary": AspectStrategy(
        aspect_id="summary",
        display_title="结构化摘要",
        dense_query="abstract introduction overview contribution conclusion 论文摘要 核心贡献",
        sparse_query="abstract introduction overview contribution conclusion summary",
        system_prompt=(
            "You are a research assistant. Summarise the provided paper excerpts "
            "into a structured summary with these four sections:\n"
            "  • Objective – what problem does the paper address?\n"
            "  • Methods   – what approach or techniques are used?\n"
            "  • Results   – what are the key findings?\n"
            "  • Limitations – what are the acknowledged weaknesses or open questions?\n"
            "Write the summary in Chinese, including the section headings. "
            "Be concise. Use bullet points within each section."
        ),
        prompt_header="请依据上述论文片段，以中文撰写结构化摘要，并使用中文小标题：研究目标、研究方法、主要结果、局限性。",
        prompt_requirements=[],
    ),
}


def get_aspect_strategy(aspect: Optional[str]) -> AspectStrategy:
    """Return the corresponding aspect strategy, defaulting to general summary."""
    if not aspect:
        return _ASPECT_REGISTRY["summary"]
    return _ASPECT_REGISTRY.get(aspect, _ASPECT_REGISTRY["summary"])


def _expand_contiguous_chunks(
    fused: list[dict],
    all_meta: Optional[list[dict]],
    paper_id: str,
    top_anchor_k: int = 3,
    max_backward_hops: int = 2,
    max_forward_hops: int = 4,
    max_total: int = 15,
) -> list[dict]:
    """
    Preserve ranked hits before spending the remaining budget on neighboring chunks.
    Among the top anchors, prefer keyword-supported hits for expansion. This is a
    heuristic; page and budget limits cannot guarantee complete sections.
    """
    if not all_meta or not fused:
        return fused

    paper_chunk_map = {
        c["chunk_idx"]: c
        for c in all_meta
        if c.get("paper_id") == paper_id and "chunk_idx" in c
    }
    if not paper_chunk_map:
        return fused

    collected_indices: set[int] = set()
    result: list[dict] = []

    def _add_chunk(idx: int) -> bool:
        if idx in paper_chunk_map and idx not in collected_indices and len(result) < max_total:
            collected_indices.add(idx)
            result.append(paper_chunk_map[idx])
            return True
        return False

    # Keep ranked evidence before expanding, so neighbors cannot evict retrieved hits.
    for chunk in fused:
        c_idx = chunk.get("chunk_idx")
        if c_idx is not None:
            _add_chunk(c_idx)

    # Lexical matches can identify section headings that semantic retrieval misses.
    anchors = sorted(fused[:top_anchor_k], key=lambda c: c.get("_bm25_score", 0), reverse=True)
    for anchor in anchors:
        a_idx = anchor.get("chunk_idx")
        if a_idx is None:
            continue
        _add_chunk(a_idx)
        a_page = anchor.get("page", 0)

        # Following chunks often continue an incomplete sentence or paragraph.
        for f_hop in range(1, max_forward_hops + 1):
            cand_idx = a_idx + f_hop
            if cand_idx in paper_chunk_map:
                cand = paper_chunk_map[cand_idx]
                if abs(cand.get("page", 0) - a_page) <= 1:
                    _add_chunk(cand_idx)
                else:
                    break

        for b_hop in range(1, max_backward_hops + 1):
            cand_idx = a_idx - b_hop
            if cand_idx in paper_chunk_map:
                cand = paper_chunk_map[cand_idx]
                if abs(cand.get("page", 0) - a_page) <= 1:
                    _add_chunk(cand_idx)
                else:
                    break

    return sorted(result, key=lambda c: (c.get("page", 0), c.get("chunk_idx", 0)))


def retrieve_aspect_chunks(
    paper_id: str,
    title: str,
    strategy: AspectStrategy,
    *,
    dependencies: ResearchDependencies,
    top_k: int = 12,
) -> list[dict]:
    """
    Retrieve paper chunks for a specific aspect using hybrid dense + sparse retrieval
    fused with RRF, with contiguous boundary expansion.
    """
    faiss_idx, bm25_idx = dependencies.load_indexes()
    if faiss_idx is None:
        return []

    # 1. Dense retrieval (semantic matches)
    dense_text = f"{title} {strategy.dense_query}" if strategy.aspect_id == "summary" else strategy.dense_query
    q_vec = dependencies.embed([dense_text])
    faiss_chunks = faiss_idx.search(q_vec, k=top_k, paper_id=paper_id)

    # 2. Sparse retrieval (exact keyword and section header matches)
    sparse_text = f"{title} {strategy.sparse_query}" if strategy.aspect_id == "summary" else strategy.sparse_query
    bm25_chunks = bm25_idx.search(sparse_text, k=top_k, paper_id=paper_id) if bm25_idx else []

    # 3. Hybrid fusion via RRF (suppress checklist/bibliography boilerplate when prose matches exist)
    fused = _rrf_fuse(faiss_chunks, bm25_chunks, k=config.RRF_K, top_n=top_k, filter_boilerplate=True)
    if not fused:
        fused = faiss_chunks or bm25_chunks

    if not fused:
        return []

    # 4. Context contiguity expansion (bidirectional anchor window)
    all_meta = getattr(faiss_idx, "meta", None)
    expanded = _expand_contiguous_chunks(fused, all_meta, paper_id=paper_id, max_total=15)

    # 5. Sort by (page, chunk_idx) for coherent reading flow
    return sorted(expanded, key=lambda c: (c.get("page", 0), c.get("chunk_idx", 0)))
