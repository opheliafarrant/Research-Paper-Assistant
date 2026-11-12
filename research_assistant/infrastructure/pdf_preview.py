"""Read-only access to local paper PDFs for source preview."""

import re
from pathlib import Path

from research_assistant import config

try:
    import pymupdf as fitz
except ImportError:
    import fitz


def resolve_paper_pdf(paper_id: str) -> Path | None:
    directory = config.PDF_DIR.resolve()
    path = (directory / f"{paper_id}.pdf").resolve()
    if path.parent != directory:
        raise ValueError("Invalid paper identifier")
    return path if path.is_file() else None


def page_count(path: Path) -> int:
    with fitz.open(str(path)) as document:
        return document.page_count


def _find_highlight_rects(page, text: str) -> list:
    """Locate bounding rectangles for snippet text on a PDF page."""
    if not text:
        return []
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []

    # If the text is short, try exact matching directly
    if len(clean) <= 100:
        rects = page.search_for(clean)
        if rects:
            return rects

    # Otherwise, search by sentences or punctuation-delimited phrases
    sentences = [s.strip() for s in re.split(r"[\n\.\?\!]+", clean) if len(s.strip()) >= 15]
    all_rects = []
    for s in sentences[:8]:
        rects = page.search_for(s[:60])
        if not rects and len(s) > 35:
            rects = page.search_for(s[:30])
        all_rects.extend(rects)

    # Fallback to contiguous phrase slices if sentence matching yielded no rects
    if not all_rects and len(clean) >= 20:
        for start in range(0, min(len(clean), 150), 30):
            sub = clean[start:start + 30].strip()
            if len(sub) >= 15:
                rects = page.search_for(sub)
                if rects:
                    all_rects.extend(rects)
                    break

    # Deduplicate rects by rounded coordinates
    unique = []
    seen = set()
    for r in all_rects:
        key = (round(r.x0, 1), round(r.y0, 1), round(r.x1, 1), round(r.y1, 1))
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def render_page(path: Path, page_number: int, highlight_text: str = "") -> bytes:
    with fitz.open(str(path)) as document:
        if not 1 <= page_number <= document.page_count:
            raise ValueError("PDF page number is out of range")
        page = document.load_page(page_number - 1)
        if highlight_text:
            rects = _find_highlight_rects(page, highlight_text)
            if rects:
                shape = page.new_shape()
                for r in rects:
                    shape.draw_rect(r)
                shape.finish(
                    color=(0.92, 0.60, 0.0),
                    fill=(1.0, 0.94, 0.4),
                    fill_opacity=0.35,
                    width=1.0,
                )
                shape.commit()
        return page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2), alpha=False).tobytes("png")
