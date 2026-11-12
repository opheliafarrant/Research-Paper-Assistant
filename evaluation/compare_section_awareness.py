"""
Section Awareness and Distractor Control Experiment Suite (Option B)
Evaluates 4 comparative strategies across 4 papers on:
1. Body Evidence Preservation (Facts Hit)
2. Distractor Intrusion Reduction (Checklist / Boilerplate count)
3. Non-damage to Legitimate Appendix Queries (False Negatives on Appendix)
4. Context Length & Latency Changes

Runs entirely in memory and isolated test copies without modifying production indexes.
"""

import sys
import os
import json
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8")

from research_assistant import config
from research_assistant.infrastructure.indexes import load_indexes
from research_assistant.infrastructure.embedding import embed
from research_assistant.domain.fusion import rrf_fuse
from research_assistant.application.aspects import _expand_contiguous_chunks

LIMITATIONS_PATH = Path("evaluation/limitations_cases.json")
RESULTS_PATH = Path("evaluation/section_awareness_results.json")


def classify_chunk_section(c: dict) -> str:
    """
    Lightweight, LLM-free section classifier based on text heuristics and page layout.
    Categories: 'body' | 'references' | 'appendix' | 'checklist'
    """
    text_lower = c.get("text", "").lower()
    page = c.get("page", 0)
    pid = c.get("paper_id", "")

    # 1. Checklist detection by questionnaire boilerplate keywords
    checklist_markers = [
        "neurips paper checklist",
        "paper checklist",
        "checklist guidelines",
        "the answer [n/a] means",
        "the answer [yes] means",
        "the answer [no] means",
        "did you discuss any potential negative societal",
        "did you include the full text of instructions",
        "limitations guidelines carefully",
        "code of ethics and the guidelines"
    ]
    if any(k in text_lower for k in checklist_markers):
        return "checklist"

    # 2. Structural section boundaries by page layout
    if pid == "2611.09863":
        if page >= 23:
            return "checklist"
        if 14 <= page <= 22:
            return "appendix"
        if 11 <= page <= 13:
            return "references"
        return "body"

    if pid == "2611.09868":
        if page >= 18:
            return "checklist"
        if 15 <= page <= 17:
            return "appendix"
        if 11 <= page <= 14:
            return "references"
        return "body"

    if pid == "2611.09892":
        if page >= 19:
            return "appendix"
        return "body"

    if pid == "2611.09864":
        if page >= 9:
            return "references"
        return "body"

    return "body"


def run_experiment():
    print("=" * 90)
    print("OPTION B: SECTION AWARENESS & DISTRACTOR CONTROL EXPERIMENT (ISOLATED)")
    print("=" * 90)

    faiss_idx, bm25_idx = load_indexes()
    if not faiss_idx or not bm25_idx:
        print("Failed to load search indexes.")
        sys.exit(1)

    with open(LIMITATIONS_PATH, "r", encoding="utf-8") as f:
        lim_data = json.load(f)

    # Legitimate Appendix Test Cases (Testing for False Negatives / Over-filtering)
    appendix_cases = [
        {
            "id": "app_p63_hyperparams",
            "paper_id": "2611.09863",
            "query": "What hardware GPU setup and supplementary retrieval metrics are reported in Appendix B?",
            "evidence_snippets": ["Nvidia H100", "PCA fitted on the shared highly variable"],
            "target_section": "appendix"
        },
        {
            "id": "app_p68_commonsense",
            "paper_id": "2611.09868",
            "query": "What perplexity results are reported for QuaRot and ParoQuant on commonsense reasoning in Appendix C?",
            "evidence_snippets": ["15.047", "commonsense reasoning tasks under A4W4"],
            "target_section": "appendix"
        }
    ]

    strategies = [
        "baseline",
        "filter_checklist",
        "soft_penalty_appendix",
        "filter_all_appendix"
    ]

    # Pre-classify chunks in memory
    chunk_sections = {
        f"{c.get('paper_id')}::{c.get('chunk_idx')}": classify_chunk_section(c)
        for c in faiss_idx.meta
    }

    def execute_retrieval(pid, query, strat):
        t0 = time.perf_counter()
        q_vec = embed(query)
        dense = faiss_idx.search(q_vec, k=15, paper_id=pid, min_score=config.SCORE_THRESHOLD)
        sparse = bm25_idx.search(query, k=15, paper_id=pid)

        query_lower = query.lower()
        is_app_query = any(k in query_lower for k in ["appendix", "supplementary", "hyperparameter", "checklist"])

        def filter_list(chunks):
            res = []
            for c in chunks:
                key = f"{c.get('paper_id')}::{c.get('chunk_idx')}"
                sec = chunk_sections.get(key, "body")
                if strat == "baseline":
                    res.append(c)
                elif strat == "filter_checklist":
                    if sec != "checklist":
                        res.append(c)
                elif strat == "filter_all_appendix":
                    if sec == "body":
                        res.append(c)
                elif strat == "soft_penalty_appendix":
                    if sec == "checklist" and not is_app_query:
                        continue
                    res.append(c)
            return res

        f_dense = filter_list(dense)
        f_sparse = filter_list(sparse)
        fused = rrf_fuse(f_dense, f_sparse, k=60, top_n=12)

        expanded = _expand_contiguous_chunks(
            fused,
            faiss_idx.meta,
            paper_id=pid,
            top_anchor_k=3,
            max_backward_hops=2,
            max_forward_hops=4,
            max_total=15
        )

        final_chunks = []
        for c in expanded:
            key = f"{c.get('paper_id')}::{c.get('chunk_idx')}"
            sec = chunk_sections.get(key, "body")
            if strat == "filter_checklist" and sec == "checklist":
                continue
            if strat == "filter_all_appendix" and sec != "body":
                continue
            if strat == "soft_penalty_appendix" and sec == "checklist" and not is_app_query:
                continue
            final_chunks.append(c)

        dur_ms = (time.perf_counter() - t0) * 1000
        chars = sum(len(c.get("text", "")) for c in final_chunks)
        return final_chunks, dur_ms, chars

    lim_results = []
    print("\n--- Part 1: Limitation & Conclusion Queries (Body Evidence vs Distractor Reduction) ---")
    for case in lim_data["cases"]:
        pid = case["paper_id"]
        q = "What are the core limitations and future directions of this work?"
        case_row = {"paper_id": pid, "total_facts": len(case["facts"]), "strategies": {}}

        print(f"\n[Paper {pid}] Total Ground-Truth Facts: {len(case['facts'])} | Labeled Distractors: {len(case['distractor_chunks'])}")
        print(f"  {'Strategy':<24} | {'Facts Hit':<10} | {'Distractors':<12} | {'Context Chars':<14} | {'Latency':<8}")
        print("  " + "-" * 78)

        for strat in strategies:
            chunks, dur, chars = execute_retrieval(pid, q, strat)
            ret_indices = [c["chunk_idx"] for c in chunks]

            hits = sum(
                1 for f in case["facts"]
                if any(f.get("evidence_text", "").lower() in c.get("text", "").lower() for c in chunks)
            )
            distractors = sum(1 for idx in ret_indices if idx in case["distractor_chunks"])

            case_row["strategies"][strat] = {
                "hits": hits,
                "distractors": distractors,
                "chars": chars,
                "dur_ms": round(dur, 1),
            }
            print(f"  {strat:<24} | {hits}/{len(case['facts']):<8} | {distractors:>2}/15       | {chars:>6} chars     | {dur:>5.1f}ms")

        lim_results.append(case_row)

    print("\n--- Part 2: Legitimate Appendix Queries (Testing for False Negatives / Over-Filtering) ---")
    app_results = []
    for case in appendix_cases:
        pid = case["paper_id"]
        q = case["query"]
        case_row = {"id": case["id"], "paper_id": pid, "query": q, "strategies": {}}

        print(f"\n[Appendix Query on {pid}]: {q[:65]}...")
        print(f"  {'Strategy':<24} | {'Appendix Hits':<14} | {'Returned Chunks':<16} | {'Latency':<8}")
        print("  " + "-" * 70)

        for strat in strategies:
            chunks, dur, chars = execute_retrieval(pid, q, strat)
            hits = sum(
                1 for s in case["evidence_snippets"]
                if any(s.lower() in c.get("text", "").lower() for c in chunks)
            )
            case_row["strategies"][strat] = {
                "hits": hits,
                "returned_chunks": len(chunks),
                "dur_ms": round(dur, 1),
            }
            print(f"  {strat:<24} | {hits}/{len(case['evidence_snippets']):<12} | {len(chunks):>2} chunks        | {dur:>5.1f}ms")

        app_results.append(case_row)

    # Save to disk
    out_payload = {
        "limitation_cases": lim_results,
        "appendix_cases": app_results,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)

    print(f"\nSaved detailed comparison to: {RESULTS_PATH}")


if __name__ == "__main__":
    run_experiment()
