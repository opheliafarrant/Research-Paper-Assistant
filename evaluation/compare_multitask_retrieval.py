"""
Multi-Task RAG Retrieval Evaluation Suite (v2 - Hardened)
Evaluates Term, Method, Experiment, Unanswerable, and Limitation questions across 4 papers.
Metrics are computed against ground-truth verbatim text snippets, completely decoupled from chunk_idx.

Methodological hardening fixes applied per audit:
1. Aligns field name to `_confidence` (set by domain/fusion.py).
2. Uses production `config.SCORE_THRESHOLD` (0.40) for dense retrieval.
3. Decouples MRR: MRR is only computed on relevance-ranked candidate lists (Dense, Sparse, RRF).
   `rrf_expanded_15` is sorted by (page, chunk_idx) for LLM reading order; reporting MRR on page order is mathematically invalid.
4. Separates retrieval-layer score distribution from end-to-end LLM refusal.
"""

import sys
import os
import json
import hashlib
from pathlib import Path
from collections import defaultdict

# Ensure project root is in sys.path
sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8")

from research_assistant import config
from research_assistant.infrastructure.indexes import load_indexes
from research_assistant.infrastructure.embedding import embed
from research_assistant.domain.fusion import rrf_fuse
from research_assistant.application.aspects import _expand_contiguous_chunks

BENCHMARK_PATH = Path("evaluation/multitask_eval_cases.json")
RESULTS_PATH = Path("evaluation/multitask_eval_results_v2.json")


def _check_evidence_match(chunks: list[dict], snippets: list[str]) -> tuple[bool, int, float, int]:
    """
    Check if verbatim snippets appear in retrieved chunks.
    Returns:
        hit (bool): At least 1 snippet matched
        matched_count (int): How many distinct snippets matched
        mrr (float): Reciprocal rank of first matching chunk (1-based)
        first_hit_rank (int): Rank of first matching chunk (0 if none)
    """
    if not snippets or not chunks:
        return False, 0, 0.0, 0

    matched_snippets = set()
    first_hit_rank = 0

    for rank, chunk in enumerate(chunks, start=1):
        chunk_text_lower = chunk.get("text", "").lower()
        chunk_matched_any = False
        for s in snippets:
            if s.lower() in chunk_text_lower:
                matched_snippets.add(s)
                chunk_matched_any = True
        if chunk_matched_any and first_hit_rank == 0:
            first_hit_rank = rank

    hit = len(matched_snippets) > 0
    mrr = (1.0 / first_hit_rank) if first_hit_rank > 0 else 0.0
    return hit, len(matched_snippets), mrr, first_hit_rank


def run_benchmark():
    if not BENCHMARK_PATH.exists():
        print(f"Benchmark file not found: {BENCHMARK_PATH}")
        sys.exit(1)

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    benchmark_bytes = BENCHMARK_PATH.read_bytes()
    benchmark_sha256 = hashlib.sha256(benchmark_bytes).hexdigest()
    print(f"Loaded Benchmark: {BENCHMARK_PATH.name} (SHA-256: {benchmark_sha256})")
    print(f"Total Cases: {len(benchmark_data['cases'])}")

    faiss_idx, bm25_idx = load_indexes()
    if not faiss_idx or not bm25_idx:
        print("Failed to load search indexes from data/cache/")
        sys.exit(1)

    methods = ["dense_10", "sparse_10", "rrf_10", "rrf_expanded_15"]
    case_results = []

    # Category aggregators: category -> method -> list of (hit, mrr, coverage)
    category_stats = defaultdict(lambda: defaultdict(list))
    unanswerable_stats = defaultdict(lambda: defaultdict(list))

    for case in benchmark_data["cases"]:
        c_id = case["id"]
        cat = case["category"]
        pid = case["paper_id"]
        query = case["query"]
        snippets = case.get("evidence_snippets", [])
        is_unans = case.get("is_unanswerable", False)

        q_vec = embed(query)

        # 1. Dense (FAISS) - aligned to production SCORE_THRESHOLD (0.40)
        dense_results = faiss_idx.search(q_vec, k=10, paper_id=pid, min_score=config.SCORE_THRESHOLD)

        # 2. Sparse (BM25)
        sparse_results = bm25_idx.search(query, k=10, paper_id=pid)

        # 3. RRF Fusion
        rrf_results = rrf_fuse(dense_results, sparse_results, k=60, top_n=10)

        # 4. RRF + Contiguous Expansion (Note: output is sorted by (page, chunk_idx))
        expanded_results = _expand_contiguous_chunks(
            rrf_results,
            faiss_idx.meta,
            paper_id=pid,
            top_anchor_k=3,
            max_backward_hops=2,
            max_forward_hops=4,
            max_total=15,
        )

        pipeline_runs = {
            "dense_10": dense_results,
            "sparse_10": sparse_results,
            "rrf_10": rrf_results,
            "rrf_expanded_15": expanded_results,
        }

        case_summary = {
            "id": c_id,
            "category": cat,
            "paper_id": pid,
            "query": query,
            "is_unanswerable": is_unans,
            "total_snippets": len(snippets),
            "methods": {},
        }

        for m_name in methods:
            chunks = pipeline_runs[m_name]
            if is_unans:
                # Aligned to actual chunk['_confidence'] from fusion.py and '_faiss_score'
                high_conf_count = sum(1 for c in chunks if c.get("_confidence", 0.0) >= 0.70 or c.get("_faiss_score", 0.0) >= 0.65)
                max_score = max([c.get("_confidence", c.get("_faiss_score", 0.0)) for c in chunks], default=0.0)
                unanswerable_stats[cat][m_name].append({
                    "high_conf_count": high_conf_count,
                    "max_score": round(max_score, 3),
                    "returned": len(chunks),
                })
                case_summary["methods"][m_name] = {
                    "returned": len(chunks),
                    "max_score": round(max_score, 3),
                    "high_conf_count": high_conf_count,
                }
            else:
                hit, matched_count, mrr, first_rank = _check_evidence_match(chunks, snippets)
                coverage = (matched_count / len(snippets)) if snippets else 0.0
                
                # MRR is invalid for rrf_expanded_15 because it is ordered by (page, chunk_idx)
                valid_mrr = mrr if m_name != "rrf_expanded_15" else None

                category_stats[cat][m_name].append({
                    "hit": hit,
                    "mrr": valid_mrr,
                    "coverage": coverage,
                    "matched_count": matched_count,
                    "total_snippets": len(snippets),
                })
                case_summary["methods"][m_name] = {
                    "returned": len(chunks),
                    "hit": hit,
                    "matched_count": matched_count,
                    "coverage": round(coverage, 3),
                    "mrr": round(valid_mrr, 3) if valid_mrr is not None else None,
                    "first_rank": first_rank if m_name != "rrf_expanded_15" else None,
                }

        case_results.append(case_summary)

    # Save detailed evaluation run to a versioned path
    full_output = {
        "benchmark_sha256": benchmark_sha256,
        "dense_threshold": config.SCORE_THRESHOLD,
        "total_cases": len(case_results),
        "cases": case_results,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(full_output, f, ensure_ascii=False, indent=2)

    print(f"\nSaved detailed results to: {RESULTS_PATH}")
    print("\n" + "=" * 90)
    print("MULTI-TASK RETRIEVAL EVALUATION SUMMARY (Hardened & Decoupled)")
    print("=" * 90)

    header = f"{'Category':<15} | {'Metric':<10} | {'Dense (Top10)':<13} | {'BM25 (Top10)':<13} | {'RRF (Top10)':<13} | {'RRF+Exp (Top15)':<15}"
    print(header)
    print("-" * len(header))

    ordered_cats = ["term", "method", "experiment", "limitation"]
    for cat in ordered_cats:
        stats = category_stats[cat]
        if not stats:
            continue
        count = len(stats["dense_10"])

        # Hit Rate
        hit_row = f"{cat} (N={count})".ljust(15) + " | " + "Hit Rate".ljust(10) + " | "
        for m in methods:
            hits = sum(1 for item in stats[m] if item["hit"])
            rate = hits / count if count > 0 else 0.0
            hit_row += f"{rate * 100:>5.1f}% ({hits}/{count})".ljust(13) + " | "
        print(hit_row.rstrip(" |"))

        # MRR (Excludes rrf_expanded_15)
        mrr_row = "".ljust(15) + " | " + "MRR".ljust(10) + " | "
        for m in methods:
            if m == "rrf_expanded_15":
                mrr_row += "N/A (page-ord) ".ljust(13) + " | "
            else:
                avg_mrr = sum(item["mrr"] for item in stats[m] if item["mrr"] is not None) / count if count > 0 else 0.0
                mrr_row += f"{avg_mrr:>5.3f}".ljust(13) + " | "
        print(mrr_row.rstrip(" |"))

        # Evidence Coverage
        cov_row = "".ljust(15) + " | " + "Coverage".ljust(10) + " | "
        for m in methods:
            avg_cov = sum(item["coverage"] for item in stats[m]) / count if count > 0 else 0.0
            cov_row += f"{avg_cov * 100:>5.1f}%".ljust(13) + " | "
        print(cov_row.rstrip(" |"))
        print("-" * len(header))

    # Overall (Answerable questions)
    all_ans_count = sum(len(category_stats[c]["dense_10"]) for c in ordered_cats)
    if all_ans_count > 0:
        overall_hit = f"OVERALL (N={all_ans_count})".ljust(15) + " | " + "Hit Rate".ljust(10) + " | "
        for m in methods:
            hits = sum(sum(1 for item in category_stats[c][m] if item["hit"]) for c in ordered_cats)
            rate = hits / all_ans_count
            overall_hit += f"{rate * 100:>5.1f}% ({hits}/{all_ans_count})".ljust(13) + " | "
        print(overall_hit.rstrip(" |"))

        overall_mrr = "".ljust(15) + " | " + "MRR".ljust(10) + " | "
        for m in methods:
            if m == "rrf_expanded_15":
                overall_mrr += "N/A (page-ord) ".ljust(13) + " | "
            else:
                total_mrr = sum(sum(item["mrr"] for item in category_stats[c][m] if item["mrr"] is not None) for c in ordered_cats)
                avg_mrr = total_mrr / all_ans_count
                overall_mrr += f"{avg_mrr:>5.3f}".ljust(13) + " | "
        print(overall_mrr.rstrip(" |"))
        print("=" * len(header))

    # Unanswerable Summary
    unans_count = len(unanswerable_stats["unanswerable"]["dense_10"])
    if unans_count > 0:
        print(f"\nUnanswerable Retrieval Filter Behavior (N={unans_count}):")
        print("  [Notice]: Only evaluates whether retrieval assigns high scores; end-to-end refusal requires LLM prompt verification.")
        for m in methods:
            avg_max_score = sum(item["max_score"] for item in unanswerable_stats["unanswerable"][m]) / unans_count
            high_conf = sum(item["high_conf_count"] for item in unanswerable_stats["unanswerable"][m])
            print(f"  • {m:<16}: Average Top Score = {avg_max_score:.3f} | High-confidence False Alarms (>=0.70) = {high_conf}")


if __name__ == "__main__":
    run_benchmark()
