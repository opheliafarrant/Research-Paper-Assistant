"""Read-only exploratory comparison of current aspect retrieval alternatives.

Run from project root: python -B -m evaluation.compare_limitations_retrieval
The three current-index cases are hand-selected and intentionally not a benchmark.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

from research_assistant import config
from research_assistant.application.aspects import _expand_contiguous_chunks, get_aspect_strategy
from research_assistant.domain.fusion import rrf_fuse
from research_assistant.infrastructure import embedding, indexes


def evaluate(facts, distractors, chunks):
    ids = {chunk["chunk_idx"] for chunk in chunks}
    hits = [fact["id"] for fact in facts if set(fact["evidence_chunks"]) & ids]
    distractor_set = set(distractors)
    distractor_hits = sorted(list(ids & distractor_set))
    return {
        "hits": hits,
        "hit_count": len(hits),
        "total_facts": len(facts),
        "distractor_count": len(distractor_hits),
        "distractor_chunks": distractor_hits,
        "returned": len(chunks),
        "unique_chunks": len(ids),
        "chunks": [chunk["chunk_idx"] for chunk in chunks],
    }


def main():
    path = Path(__file__).with_name("limitations_cases.json")
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
    faiss_index, bm25_index = indexes.load_indexes()
    if faiss_index is None or bm25_index is None:
        raise RuntimeError("Both local indexes are required")
    strategy = get_aspect_strategy("limitations")
    vector = embedding.embed([strategy.dense_query])
    report = {"label_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "faiss_rows": len(faiss_index.meta), "bm25_rows": len(bm25_index.chunks), "cases": []}
    for case in cases:
        pid = case["paper_id"]
        available = {chunk["chunk_idx"] for chunk in faiss_index.meta if chunk["paper_id"] == pid}
        for fact in case["facts"]:
            if not set(fact["evidence_chunks"]) <= available:
                raise ValueError(f"Missing labeled evidence for {pid}: {fact['id']}")
            if not any(fact["evidence_text"].casefold() in chunk["text"].casefold()
                       for chunk in faiss_index.meta
                       if chunk["paper_id"] == pid and chunk["chunk_idx"] in fact["evidence_chunks"]):
                raise ValueError(f"Labeled evidence text changed for {pid}: {fact['id']}")
        distractors = case.get("distractor_chunks", [])
        if not set(distractors) <= available:
            raise ValueError(f"Missing distractor chunks for {pid}")
        dense = faiss_index.search(vector, k=15, paper_id=pid)
        sparse = bm25_index.search(strategy.sparse_query, k=15, paper_id=pid)
        fused12 = rrf_fuse(dense[:12], sparse[:12], k=config.RRF_K, top_n=12)
        fused15 = rrf_fuse(dense, sparse, k=config.RRF_K, top_n=15)
        expanded = _expand_contiguous_chunks(fused12, faiss_index.meta, paper_id=pid, max_total=15)
        approaches = {"dense_12": dense[:12], "bm25_12": sparse[:12], "rrf_12": fused12,
                      "dense_15": dense, "bm25_15": sparse, "rrf_15": fused15,
                      "rrf_12_expanded_15": expanded}
        counts = Counter(chunk["chunk_idx"] for chunk in faiss_index.meta if chunk["paper_id"] == pid)
        report["cases"].append({"paper_id": pid, "source_pages": case["source_pages"],
                                "index_rows": sum(counts.values()), "unique_index_chunks": len(counts),
                                "methods": {name: evaluate(case["facts"], distractors, chunks) for name, chunks in approaches.items()}})
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
