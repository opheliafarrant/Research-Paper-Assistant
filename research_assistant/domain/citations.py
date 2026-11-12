"""Pure page-span and citation formatting helpers for retrieved chunks."""

import re


def chunk_pages(chunk: dict) -> tuple[int, int]:
    """Return inclusive (start_page, end_page) for a chunk, tolerating legacy single-page dicts."""
    start = int(chunk.get("page", 1))
    spans = chunk.get("page_spans") or []
    if spans:
        pages = [int(s["page"]) for s in spans if "page" in s]
        if pages:
            return min(pages), max(pages)
    end = int(chunk.get("page_end", start))
    return start, max(start, end)


def format_page_range(chunk: dict) -> str:
    """Return a compact page label like '3' or '3-4'."""
    start, end = chunk_pages(chunk)
    return str(start) if start == end else f"{start}-{end}"


def format_chunk_text_with_pages(chunk: dict) -> str:
    """Insert inline [Page N] markers when a chunk spans multiple PDF pages."""
    text = chunk.get("text", "")
    spans = chunk.get("page_spans") or []
    if len(spans) <= 1:
        return text
    parts: list[str] = []
    for span in spans:
        segment = text[span["start"] : span["end"]].strip()
        if segment:
            parts.append(f"[Page {span['page']}] {segment}")
    return " ".join(parts) if parts else text


def resolve_chunk_page(chunk: dict, focus_text: str = "") -> int:
    """Resolve the most relevant physical page inside a chunk, defaulting to its start page."""
    start, end = chunk_pages(chunk)
    if start == end or not focus_text:
        return start
    spans = chunk.get("page_spans") or []
    text = chunk.get("text", "")
    if not spans or not text:
        return start

    # Score each page span by shared non-trivial normalized token overlap with focus_text.
    focus_tokens = set(re.findall(r"[a-z0-9]{2,}|[\u4e00-\u9fff]+", focus_text.lower()))
    if not focus_tokens:
        return start

    best_page = start
    best_overlap = -1
    for span in spans:
        seg = text[span["start"] : span["end"]].lower()
        seg_tokens = set(re.findall(r"[a-z0-9]{2,}|[\u4e00-\u9fff]+", seg))
        overlap = len(focus_tokens & seg_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_page = int(span["page"])
    return best_page if best_overlap > 0 else start
