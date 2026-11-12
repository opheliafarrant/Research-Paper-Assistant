"""Pure ranking and confidence scoring logic for combining retrieved paper chunks."""

import re


_BOILERPLATE_CHECKLIST_RE = re.compile(
    r"(?:Guidelines:\s*•|The answer \[(?:Yes|No|N/?A)\] means|\bNeurIPS Paper Checklist\b|"
    r"\bAnswer:\s*\[(?:Yes|No|N/?A)\]\s*Justification:|"
    r"\bchecklist guidelines carefully\b|\bDo not modify the questions\b|"
    r"\bnegative societal impacts\b|\bpoint to the section\(s\) where\b|"
    r"\ba \[No\] answer to this question\b|\bNeurIPS Code of Ethics\b|"
    r"\bguidelines for their institution\b|\bdo not include any information that would break anonymity\b|"
    r"\bdiscuss possible limitations of their approach\b|\bNeurIPS code and data submission guidelines\b)",
    re.IGNORECASE,
)

_QUERY_BOILERPLATE_RE = re.compile(
    r"(?:checklist|safeguard|reproducib|ethics|guideline|bibliography|reference|acknowledg|"
    r"清单|伦理|可复现|参考文献|致谢)",
    re.IGNORECASE,
)


def is_boilerplate_chunk(text: str) -> bool:
    """Detect conference checklist boilerplate or dense reference-list chunks without dropping mixed prose."""
    if not text:
        return False
    if _BOILERPLATE_CHECKLIST_RE.search(text):
        return True
    year_refs = len(re.findall(r"\(\d{4}[a-z]?\)", text))
    bracket_refs = len(re.findall(r"\[\d+\]", text))
    return year_refs >= 4 and bracket_refs >= 3


def mentions_boilerplate(query: str) -> bool:
    """Return True when the user explicitly asks about checklists, ethics, or references."""
    return bool(_QUERY_BOILERPLATE_RE.search(query or ""))


def filter_bm25_by_cosine(
    bm25_results: list[dict],
    cosine_by_key: dict[tuple, float],
    min_cosine: float = 0.30,
) -> list[dict]:
    """Drop BM25-only matches with near-zero semantic cosine when cosine measurements are available."""
    if not cosine_by_key:
        return bm25_results
    kept: list[dict] = []
    for chunk in bm25_results:
        key = (chunk.get("paper_id"), chunk.get("chunk_idx"))
        cos = cosine_by_key.get(key)
        if cos is None or cos >= min_cosine:
            kept.append(chunk)
    return kept


def prune_weak_tail(
    fused: list[dict],
    cosine_by_key: dict[tuple, float],
    *,
    min_keep: int = 3,
    single_path_min_cosine: float = 0.48,
) -> list[dict]:
    """
    Preserve top `min_keep` hits unconditionally, and retain later slots only if
    supported by hybrid consensus or strong dense cosine similarity.
    """
    if len(fused) <= min_keep or not cosine_by_key:
        return fused
    kept = list(fused[:min_keep])
    for chunk in fused[min_keep:]:
        has_hybrid = chunk.get("_faiss_score") is not None and chunk.get("_bm25_score") is not None
        key = (chunk.get("paper_id"), chunk.get("chunk_idx"))
        cos = cosine_by_key.get(key)
        if has_hybrid or cos is None or cos >= single_path_min_cosine:
            kept.append(chunk)
    return kept


def calculate_confidence(chunk: dict, max_rrf: float = 2.0 / 61.0) -> tuple[float, str, str]:
    """
    Calculate normalized confidence score and badge label for a retrieved chunk.
    Returns (score, label, badge) where badge is e.g. '高相关 0.85'.
    """
    faiss = chunk.get("_faiss_score")
    bm25 = chunk.get("_bm25_score")
    rrf = chunk.get("_rrf_score", 0.0)
    norm_rrf = min(1.0, rrf / max_rrf) if max_rrf > 0 else 0.5

    if faiss is not None and bm25 is not None:
        # Both dense semantic and sparse keyword matched (hybrid consensus)
        score = 0.5 * faiss + 0.4 * norm_rrf + 0.1
        score = min(0.98, max(0.75, score))
    elif faiss is not None:
        score = 0.7 * faiss + 0.3 * norm_rrf
        score = min(0.92, max(0.40, score))
    elif bm25 is not None:
        score = 0.45 + 0.45 * norm_rrf
        score = min(0.85, max(0.40, score))
    else:
        score = max(0.30, norm_rrf)

    score = round(score, 2)
    if score >= 0.70:
        label = "高相关"
    elif score >= 0.50:
        label = "中相关"
    else:
        label = "参考"

    badge = f"{label} {score:.2f}"
    return score, label, badge


def rrf_fuse(
    faiss_results: list[dict],
    bm25_results: list[dict],
    k: int,
    top_n: int,
    *,
    filter_boilerplate: bool = False,
    query_mentions_boilerplate: bool = False,
) -> list[dict]:
    scores: dict[str, float] = {}
    chunks: dict[str, dict] = {}
    for results in (faiss_results, bm25_results):
        seen: set[str] = set()
        for chunk in results:
            key = f"{chunk['paper_id']}::{chunk['chunk_idx']}"
            if key in seen:
                continue
            seen.add(key)
            rank = len(seen)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in chunks:
                chunks[key] = chunk.copy()
            else:
                if "_faiss_score" in chunk:
                    chunks[key]["_faiss_score"] = chunk["_faiss_score"]
                if "_bm25_score" in chunk:
                    chunks[key]["_bm25_score"] = chunk["_bm25_score"]

    if filter_boilerplate and not query_mentions_boilerplate and scores:
        non_bp = {
            key: sc
            for key, sc in scores.items()
            if not is_boilerplate_chunk(chunks[key].get("text", ""))
        }
        if non_bp:
            scores = non_bp

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    max_rrf = 2.0 / (k + 1.0)
    result = []
    for key, score in ranked[:top_n]:
        chunk = chunks[key].copy()
        chunk['_rrf_score'] = round(score, 6)
        conf_score, conf_label, conf_badge = calculate_confidence(chunk, max_rrf=max_rrf)
        chunk['_confidence'] = conf_score
        chunk['_confidence_label'] = conf_label
        chunk['_confidence_badge'] = conf_badge
        result.append(chunk)
    return result

