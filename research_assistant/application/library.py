"""Read-only paper catalog use case."""

from typing import Callable


def list_papers(*, load_indexes: Callable[[], tuple]) -> list[dict]:
    faiss_index, _ = load_indexes()
    if faiss_index is None or not faiss_index.meta:
        return []
    seen, papers = set(), []
    for chunk in faiss_index.meta:
        paper_id = chunk["paper_id"]
        if paper_id not in seen:
            seen.add(paper_id)
            papers.append({"paper_id": paper_id, "title": chunk["title"]})
    return papers
