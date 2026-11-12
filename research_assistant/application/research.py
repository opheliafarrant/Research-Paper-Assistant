"""
The core RAG pipeline: query → retrieve → fuse → generate
answer(query, paper_id = None)  ->  dict
Returns:
        {
            "answer":   str,           # LLM-generated answer
            "sources":  list[dict],    # top fused chunks used as context
            "query":    str,
        }

summarise(paper_id, title)  ->  str
    Generate a concise summary of a single paper from its indexed chunks
"""

import logging
import re
from typing import Optional
from research_assistant import config
from research_assistant.application.aspects import get_aspect_strategy, retrieve_aspect_chunks
from research_assistant.domain.citations import format_chunk_text_with_pages, format_page_range
from research_assistant.domain.contracts import Answer, Summary, ResearchDependencies
from research_assistant.domain.fusion import (
    filter_bm25_by_cosine,
    mentions_boilerplate,
    prune_weak_tail,
    rrf_fuse as _rrf_fuse,
)
logger = logging.getLogger(__name__)

_HAS_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

_TRANSLATED_BM25_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "has", "have", "how", "in", "into", "is", "it", "its",
    "of", "on", "or", "that", "the", "their", "this", "to", "was", "were",
    "what", "when", "where", "which", "who", "why", "will", "with",
})

_TRANSLATE_SYSTEM = (
    "You translate Chinese academic queries into clear, standard English search questions.\n"
    "Rules:\n"
    "1. Keep ALL Latin terms, model names, dataset names, and technical identifiers verbatim.\n"
    "2. Translate EVERY Chinese word into accurate ML English with normal spaces between words.\n"
    "3. Distinguish quantization vs. quantitative and reasoning vs. inference accurately.\n"
    "4. Output ONLY one plain English question on a single line — NO Chinese characters, underscores, CamelCase word-joining, multi-hyphen word chains, quotes, or explanations."
)

_GREETING_PATTERNS = re.compile(
    r"^\s*(hi|hello|hey|greetings|good\s+(morning|afternoon|evening)|howdy|sup|what'?s\s+up|hiya)\W*$",
    re.IGNORECASE,
)

_GREETING_PREFIX = re.compile(
    r"^\s*(hi|hello|hey|greetings|good\s+(morning|afternoon|evening)|howdy|sup|what'?s\s+up|hiya)[,!\s]+",
    re.IGNORECASE,
)

def _classify_query(query: str) -> tuple[str, str]:
    stripped = query.strip()

    if _GREETING_PATTERNS.match(stripped) or stripped in {"你好", "您好", "嗨", "你好！", "您好！"}:
        return "greeting", stripped

    greeting_match = _GREETING_PREFIX.match(stripped)
    if greeting_match:
        remainder = stripped[greeting_match.end():].strip()
        if remainder:
            return "greeting_with_question", remainder

    return "research", stripped


def _build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks with explicit arXiv IDs and inline multi-page markers."""
    sections = []
    for i, c in enumerate(chunks, start=1):
        pages = format_page_range(c)
        body = format_chunk_text_with_pages(c)
        sections.append(
            f"Excerpt {i} (Paper ID: {c['paper_id']}, Page: {pages}, Title: {c['title']}):\n"
            f"{body}"
        )
    return "\n\n".join(sections)


def _clean_translation(original: str, raw: str) -> str:
    line = next((ln.strip() for ln in (raw or "").splitlines() if ln.strip()), "")
    cleaned = line.strip("`\"'")
    if "_" in cleaned and "_" not in original:
        cleaned = cleaned.replace("_", " ")
    return " ".join(cleaned.split())


def _acceptable_translation(original: str, translated: str) -> bool:
    if not translated or len(translated) > 400 or "\n" in translated or _HAS_CJK_RE.search(translated):
        return False
    if " " not in translated and len(translated) > 16:
        return False
    if re.search(r"[A-Za-z]+(?:-[A-Za-z]+){3,}", translated):
        return False
    if not re.search(r"[A-Za-z]", translated):
        return False
    latin_tokens = re.findall(r"[A-Za-z][A-Za-z0-9]*(?:[-_.][A-Za-z0-9]+)*", original)
    lower = translated.casefold()
    return all(token.casefold() in lower for token in latin_tokens)


def _translate_query_for_search(query: str, generate_fn) -> tuple[str, bool]:
    """Translate Chinese research questions into clean English search queries when possible."""
    if not _HAS_CJK_RE.search(query) or generate_fn is None:
        return query, False
    try:
        raw = generate_fn(f"Translate to English:\n{query}", system=_TRANSLATE_SYSTEM)
        candidate = _clean_translation(query, raw if isinstance(raw, str) else "")
        if _acceptable_translation(query, candidate):
            return candidate, True
        retry_prompt = (
            "Rewrite as one grammatical English question using normal spaces between words "
            "(no Chinese characters, no underscores, no CamelCase, no multi-word hyphens), "
            "preserving all technical IDs verbatim:\n"
            f"Original: {query}\nDraft: {candidate}"
        )
        raw_retry = generate_fn(retry_prompt, system=_TRANSLATE_SYSTEM)
        candidate_retry = _clean_translation(query, raw_retry if isinstance(raw_retry, str) else "")
        if _acceptable_translation(query, candidate_retry):
            return candidate_retry, True
    except Exception as exc:
        logger.warning("Query translation fell back to original query: %s", exc)
    return query, False


def retrieve_chunks(
    query: str,
    paper_id: Optional[str] = None,
    *,
    faiss_idx,
    bm25_idx,
    embed_fn,
    generate_fn=None,
) -> tuple[list[dict], dict]:
    """Unified cross-lingual hybrid retrieval, boilerplate filtering, and weak-tail pruning."""
    search_query, translated_ok = _translate_query_for_search(query, generate_fn)
    q_vec = embed_fn([search_query])
    faiss_results = faiss_idx.search(q_vec, k=config.TOP_K * 2, paper_id=paper_id)

    bm25_results: list[dict] = []
    if bm25_idx is not None and not _HAS_CJK_RE.search(search_query):
        bm25_query = search_query
        if translated_ok:
            filtered_terms = [
                tok for tok in re.findall(r"[A-Za-z0-9]+", search_query.lower())
                if tok not in _TRANSLATED_BM25_STOPWORDS
            ]
            if filtered_terms:
                bm25_query = " ".join(filtered_terms)
        bm25_results = bm25_idx.search(bm25_query, k=config.TOP_K * 2, paper_id=paper_id)

    logger.info(
        "Retrieved %d FAISS + %d BM25 chunks (translated=%s) for query: %r",
        len(faiss_results), len(bm25_results), translated_ok, query[:60],
    )

    cosine_map: dict[tuple, float] = {}
    if hasattr(faiss_idx, "cosine_scores"):
        raw_cosines = faiss_idx.cosine_scores(q_vec, bm25_results + faiss_results)
        if isinstance(raw_cosines, dict):
            cosine_map = raw_cosines

    bm25_filtered = filter_bm25_by_cosine(bm25_results, cosine_map, min_cosine=0.30)
    query_asks_bp = mentions_boilerplate(query) or mentions_boilerplate(search_query)

    fused = _rrf_fuse(
        faiss_results,
        bm25_filtered,
        k=config.RRF_K,
        top_n=config.TOP_K,
        filter_boilerplate=True,
        query_mentions_boilerplate=query_asks_bp,
    )
    if not fused and (_HAS_CJK_RE.search(query) or paper_id is not None):
        # Cross-language or generic single-paper queries can fall just below the default threshold.
        fallback_vec = embed_fn([query]) if search_query != query else q_vec
        fallback = faiss_idx.search(fallback_vec, k=config.TOP_K * 2, paper_id=paper_id, min_score=0.30)
        fused = _rrf_fuse(
            fallback,
            [],
            k=config.RRF_K,
            top_n=config.TOP_K,
            filter_boilerplate=True,
            query_mentions_boilerplate=query_asks_bp,
        )
        if hasattr(faiss_idx, "cosine_scores") and fused:
            extra_cos = faiss_idx.cosine_scores(fallback_vec, fused)
            if isinstance(extra_cos, dict):
                cosine_map.update(extra_cos)

    fused = prune_weak_tail(fused, cosine_map, min_keep=3, single_path_min_cosine=0.48)
    meta = {
        "search_query": search_query,
        "translated": translated_ok,
        "faiss_count": len(faiss_results),
        "bm25_count": len(bm25_filtered),
    }
    return fused, meta





# public API 
def answer(query: str, paper_id: Optional[str] = None, *, dependencies: ResearchDependencies) -> Answer:
    """
    Full RAG pipeline
    Parameters:
        query    : str   
        paper_id : str | None
        If given retrieval is restricted to chunks from that paper only
        (used in single-paper deep-dive mode in the UI)

    Returns:
    {
        "answer":  str,
        "sources": list[dict],   # fused top-k chunks
        "query":   str,
    }
    """

    intent, clean_query = _classify_query(query)

    if intent == "greeting":
        return {
            "answer":  ("你好！可以用中文或英文询问已收录论文的内容。" if re.search(r"[\u4e00-\u9fff]", clean_query) else
                        "Hello! Ask me about the indexed papers in English or Chinese."),
            "sources": [],
            "query":   query,
        }

    if intent == "greeting_with_question":
        query = clean_query

    faiss_idx, bm25_idx = dependencies.load_indexes()
    if faiss_idx is None:
        return {
            "answer":  "还没有建立论文索引，请先点击侧栏的“获取新论文”。",
            "sources": [],
            "query":   query,
        }
    

    # retrieve & fuse
    fused, _ = retrieve_chunks(
        query,
        paper_id,
        faiss_idx=faiss_idx,
        bm25_idx=bm25_idx,
        embed_fn=dependencies.embed,
        generate_fn=dependencies.generate,
    )
    if not fused:
        return {
            "answer":  "没有找到相关论文片段，请换一种方式提问。",
            "sources": [],
            "query":   query,
        }

    # generate
    context = _build_context(fused)
    system = (
        "You are a precise academic research assistant that answers questions about papers.\n"
        "Strict grounding rules:\n"
        "1. Answer ONLY using facts explicitly stated in the provided Excerpts. "
        "Never invent paper sections (e.g., abstract/introduction), datasets, methods, metrics, or Paper IDs not present in the Excerpts.\n"
        "2. Inspect ALL Excerpts carefully so key framework names, algorithms, training vs. inference stages, baselines, metrics, and limitations are stated accurately without omissions or stage confusion.\n"
        "3. When citing a claim, copy the exact `Paper ID` and `Page` from THAT Excerpt's header or inline `[Page N]` marker (format: `(<Paper ID>, p.<Page>)`), NEVER the Excerpt number or any other paper's ID.\n"
        "4. Always write a complete, direct prose answer first and embed citations inline; NEVER reply with only a Paper ID or page number.\n"
        "5. If the Excerpts do not contain enough information to answer the question (or mention unknown devices/benchmarks/settings), clearly state what is not reported in the Excerpts rather than guessing.\n"
        "6. Respond strictly in the same language as the user's question."
    )

    answer_instruction = (
        "请严格依据上述论文片段用中文写出完整、简洁的回答（须包含关键方法名称、阶段、数据集、基线或局限等具体内容，勿只输出编号，并在对应结论处标注该片段头部给出的真实 arXiv 编号与页码）："
        if _HAS_CJK_RE.search(query)
        else "Answer the Question concisely and completely in English based strictly on ALL Excerpts above (state the exact method/framework names, stages, benchmarks, baselines, or limitations directly, and embed exact Paper ID and page citations from the supporting Excerpt headers; never reply with only a citation):"
    )
    prompt = (
        f"Excerpts:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"{answer_instruction}"
    )

    logger.info("Calling Ollama (%s)", config.OLLAMA_MODEL)
    generated = dependencies.generate(prompt, system = system)

    return {
        "answer":  generated,
        "sources": fused,
        "query":   query,
    }



def summarise(
    paper_id: str,
    title: str,
    aspect: Optional[str] = None,
    *,
    dependencies: ResearchDependencies,
) -> str:
    """Return the summary text, retaining the existing public interface."""
    return summarise_with_sources(paper_id, title, aspect, dependencies=dependencies)["text"]


def summarise_with_sources(
    paper_id: str,
    title: str,
    aspect: Optional[str] = None,
    *,
    dependencies: ResearchDependencies,
) -> Summary:
    """Return a summary and the exact retrieved chunks supplied to the model."""
    faiss_idx, _ = dependencies.load_indexes()
    if faiss_idx is None:
        return {"text": "还没有建立论文索引，请先点击侧栏的“获取新论文”。", "sources": []}

    strategy = get_aspect_strategy(aspect)
    chunks = retrieve_aspect_chunks(paper_id, title, strategy, dependencies=dependencies)

    if not chunks:
        return {"text": f"没有找到论文 {paper_id} 的索引片段。", "sources": []}

    context = _build_context(chunks)
    prompt = strategy.build_prompt(title=title, paper_id=paper_id, context=context)

    logger.info("Summarising paper %s (aspect=%s)", paper_id, aspect)
    return {"text": dependencies.generate(prompt, system=strategy.system_prompt), "sources": chunks}


# Test 
if __name__ == "__main__":
    from research_assistant.bootstrap import build_application
    app = build_application()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    print("=" * 60)
    print("RAG ENGINE SMOKE TEST")
    print("=" * 60)



    # Q&A test
    # test_query = "What methods are used for efficient visual state space modeling?"
    test_query = "Hello How are You"
    print(f"\nQuery: {test_query}\n")
    result = app.answer(test_query)

    print("ANSWER:")
    print(result["answer"])
    print(f"\nSOURCES ({len(result['sources'])} chunks):")
    for s in result["sources"]:
        print(
            f"  [{s['paper_id']} p{s['page']}]"
            f"  rrf={s['_rrf_score']:.4f}"
            f"  {s['text'][:70].strip()!r}"
        )

    # summarise test
    faiss_idx, _ = app.load_indexes()
    if faiss_idx and faiss_idx.meta:
        first = faiss_idx.meta[0]
        pid, title = first["paper_id"], first["title"]
        print(f"\n{'='*60}")
        print(f"SUMMARY TEST  —  {pid}")
        print(f"Title: {title}")
        print("=" * 60)
        print(app.summarise(pid, title))
