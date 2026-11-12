"""Evaluate retrieval separately from generation against page-grounded paper evidence.

Run from the project root.  Output is written only to a new path explicitly supplied
by the caller; existing results, indexes, PDFs, and conversations are never overwritten.
Generation invokes the project's local Ollama service and requires no paid API.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import statistics
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit

import pymupdf

from research_assistant import config
from research_assistant.application import research
from research_assistant.bootstrap import build_application
from research_assistant.domain.fusion import rrf_fuse
from research_assistant.infrastructure import embedding, indexes, ollama

DATASET = Path(__file__).with_name("rag_eval_gold_v1.json")
ZH_QUERIES = Path(__file__).with_name("rag_eval_queries_zh_v1.json")
KS = (1, 3, 5, 6)


def normalized(text):
    """Normalize whitespace and common dash forms without dropping words."""
    text = unicodedata.normalize("NFKC", text).casefold().replace("–", "-").replace("—", "-")
    return "".join(text.split())


def evidence_matches(evidence, chunk, paper_id):
    return chunk.get("paper_id") == paper_id and normalized(evidence["quote"]) in normalized(chunk.get("text", ""))


def retrieval_metrics(evidence, chunks, paper_id, k):
    """Annotated evidence hit and recall; unlabelled chunks are not negatives."""
    selected = chunks[:k]
    matches = [any(evidence_matches(item, chunk, paper_id) for chunk in selected) for item in evidence]
    first = next((i for i, chunk in enumerate(selected, 1)
                  if any(evidence_matches(item, chunk, paper_id) for item in evidence)), None)
    return {"hit": bool(first), "evidence_recall": sum(matches) / len(evidence),
            "first_rank": first, "reciprocal_rank": 1 / first if first else 0.0,
            "evidence_hits": matches, "returned": len(selected)}


def fuse_with_fallback(question, vector, dense, sparse, faiss_index, paper_id):
    """Mirror the production answer path's fallback only for empty Chinese results."""
    fused = rrf_fuse(dense, sparse, k=config.RRF_K, top_n=config.TOP_K)
    if not fused and re.search(r"[\u4e00-\u9fff]", question):
        fallback = faiss_index.search(vector, k=config.TOP_K * 2, paper_id=paper_id, min_score=0.30)
        return rrf_fuse(fallback, [], k=config.RRF_K, top_n=config.TOP_K), True
    return fused, False


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_validate(dataset_path, faiss_index):
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    seen_ids, documents, checked = set(), {}, []
    try:
        for case in dataset["cases"]:
            cid, pid = case["id"], case["paper_id"]
            if cid in seen_ids:
                raise ValueError(f"Duplicate case ID: {cid}")
            seen_ids.add(cid)
            if pid not in documents:
                documents[pid] = pymupdf.open(str(config.PDF_DIR / f"{pid}.pdf"))
            pdf = documents[pid]
            chunks = [chunk for chunk in faiss_index.meta if chunk.get("paper_id") == pid]
            if not chunks:
                raise ValueError(f"No indexed chunks: {pid}")
            if case["type"] == "unanswerable_probe":
                marker = normalized(case["absence_marker"])
                if not marker or any(marker in normalized(page.get_text()) for page in pdf):
                    raise ValueError(f"Absence marker found in PDF: {cid}")
                if any(marker in normalized(chunk["text"]) for chunk in chunks):
                    raise ValueError(f"Absence marker found in index: {cid}")
                checked.append({"id": cid, "marker_absent": True})
                continue
            if not case["evidence"]:
                raise ValueError(f"Missing evidence: {cid}")
            for item in case["evidence"]:
                page = item["page"]
                if not item["quote"].strip() or not 1 <= page <= len(pdf):
                    raise ValueError(f"Invalid evidence in {cid}")
                if normalized(item["quote"]) not in normalized(pdf[page - 1].get_text()):
                    raise ValueError(f"Evidence missing from PDF page {page}: {cid}: {item['quote']}")
                if not any(evidence_matches(item, chunk, pid) for chunk in chunks):
                    raise ValueError(f"Evidence not represented in one indexed chunk: {cid}: {item['quote']}")
            checked.append({"id": cid, "evidence_units": len(case["evidence"])})
    finally:
        for pdf in documents.values():
            pdf.close()
    return dataset, checked


def summarize_retrieval(cases):
    groups = defaultdict(list)
    for case in cases:
        if case["type"] != "unanswerable_probe":
            groups["overall"].append(case)
            groups[case["type"]].append(case)
    result = {}
    for group, rows in groups.items():
        result[group] = {}
        for method in ("dense", "bm25", "rrf"):
            result[group][method] = {}
            for k in KS:
                values = [row["methods"][method][str(k)] for row in rows]
                result[group][method][str(k)] = {
                    "queries": len(values), "hit_rate": sum(v["hit"] for v in values) / len(values),
                    "mean_annotated_evidence_recall": statistics.mean(v["evidence_recall"] for v in values),
                    "mrr": statistics.mean(v["reciprocal_rank"] for v in values),
                }
    return result


def run(dataset_path, mode, ollama_url=None, scope="single", language="en", zh_queries_path=None):
    if ollama_url:
        config.OLLAMA_BASE_URL = ollama_url.rstrip("/")
    if mode == "all" or ollama_url:
        parsed = urlsplit(config.OLLAMA_BASE_URL)
        if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Only a local HTTP Ollama endpoint is allowed for generation")
    faiss_index, bm25_index = indexes.load_indexes()
    if faiss_index is None or bm25_index is None:
        raise RuntimeError("Both local indexes are required")
    data, validation = load_and_validate(dataset_path, faiss_index)
    cases = data["cases"]
    zh_path = zh_queries_path or ZH_QUERIES
    if language == "zh":
        translations = json.loads(zh_path.read_text(encoding="utf-8"))
        if set(translations) != {case["id"] for case in cases}:
            raise ValueError("Chinese query set does not match dataset case IDs")
        for case in cases:
            case["question"] = translations[case["id"]]
    if scope == "library":
        # These synthetic probes are formulated for a selected paper, not the entire library.
        cases = [case for case in cases if case["type"] != "unanswerable_probe"]
        for case in cases:
            title = next(chunk["title"] for chunk in faiss_index.meta if chunk["paper_id"] == case["paper_id"])
            case["question"] = (f"针对论文《{title}》，{case['question']}" if language == "zh"
                                else f"For the paper {title}, {case['question']}")
    print(f"VALIDATED {len(cases)} cases; local PDF anchors and index representation checked", flush=True)
    # One encoding batch; subsequent measured retrieval excludes embedding/model startup.
    vectors = embedding.embed([case["question"] for case in cases])
    app = build_application() if mode == "all" else None
    base_gen = ollama.generate if (mode == "all" or ollama_url) else None
    translation_cache: dict[tuple[str, str], str] = {}

    def gen_fn(prompt: str, system: str = "") -> str:
        # Stabilize per-run query translation so eval retrieval and app.answer see identical RRF inputs.
        if system == research._TRANSLATE_SYSTEM:
            key = (prompt, system)
            if key not in translation_cache:
                translation_cache[key] = base_gen(prompt, system=system)
            return translation_cache[key]
        return base_gen(prompt, system=system)

    if base_gen is None:
        gen_fn = None
    elif app is not None:
        answer_kw = getattr(app.answer, "keywords", None)
        if isinstance(answer_kw, dict) and "dependencies" in answer_kw:
            deps = answer_kw["dependencies"]
            answer_kw["dependencies"] = type(deps)(
                load_indexes=deps.load_indexes,
                embed=deps.embed,
                generate=gen_fn,
            )
    rows = []
    for case, vector in zip(cases, vectors):
        pid, question = case["paper_id"], case["question"]
        start = time.perf_counter()
        paper_filter = pid if scope == "single" else None
        dense = faiss_index.search(vector, k=config.TOP_K * 2, paper_id=paper_filter)
        dense_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        sparse = bm25_index.search(question, k=config.TOP_K * 2, paper_id=paper_filter)
        bm25_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        fused, ret_meta = research.retrieve_chunks(
            question,
            paper_filter,
            faiss_idx=faiss_index,
            bm25_idx=bm25_index,
            embed_fn=embedding.embed,
            generate_fn=gen_fn,
        )
        fusion_ms = (time.perf_counter() - start) * 1000
        fallback_used = bool(not fused and re.search(r"[\u4e00-\u9fff]", question))
        paths = {"dense": dense[:config.TOP_K], "bm25": sparse[:config.TOP_K], "rrf": fused}
        row = {"id": case["id"], "type": case["type"], "paper_id": pid, "question": question,
               "expected": case["expected"], "evidence": case["evidence"],
               "retrieval_meta": ret_meta,
               "timing_ms": {"dense": dense_ms, "bm25": bm25_ms, "rrf_pipeline": fusion_ms},
               "chinese_fallback_used": fallback_used,
               "returned_ids": {name: [c["chunk_idx"] for c in chunks] for name, chunks in paths.items()},
               "returned_documents": {name: [{"paper_id": c["paper_id"], "chunk_idx": c["chunk_idx"],
                                               "page": c["page"], "page_end": c.get("page_end", c["page"])}
                                              for c in chunks]
                                      for name, chunks in paths.items()}}
        if case["type"] != "unanswerable_probe":
            row["methods"] = {name: {str(k): retrieval_metrics(case["evidence"], chunks, pid, k) for k in KS}
                              for name, chunks in paths.items()}
        else:
            row["note"] = "Synthetic missing-marker probe: retrieval results are not an abstention or correctness score."
        if app is not None:
            start = time.perf_counter()
            try:
                result = app.answer(question, paper_id=paper_filter)
                row["generation"] = {"status": "ok", "answer": result["answer"], "sources": result["sources"],
                                     "elapsed_seconds": time.perf_counter() - start}
            except Exception as exc:
                row["generation"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}",
                                     "elapsed_seconds": time.perf_counter() - start}
        rows.append(row)
        print(f"CASE {case['id']} retrieval complete" + (f"; generation {row['generation']['status']}" if app else ""), flush=True)
    manifest_files = [dataset_path, config.CACHE_DIR / "faiss.index", config.CACHE_DIR / "faiss_meta.pkl", config.CACHE_DIR / "bm25.pkl"]
    if language == "zh":
        manifest_files.append(zh_path)
    manifest_files += [config.PDF_DIR / f"{pid}.pdf" for pid in sorted({case['paper_id'] for case in cases})]
    code_files = [Path(__file__), config.BASE_DIR / "research_assistant" / "application" / "research.py",
                  config.BASE_DIR / "research_assistant" / "domain" / "fusion.py",
                  config.BASE_DIR / "research_assistant" / "infrastructure" / "indexes.py",
                  config.BASE_DIR / "research_assistant" / "infrastructure" / "embedding.py",
                  config.BASE_DIR / "research_assistant" / "config.py"]
    model_manifest = config.MODELS_DIR / "manifests" / "registry.ollama.ai" / "library" / "qwen2.5" / "3b"
    return {"schema_version": 1, "dataset_version": data["version"], "mode": mode,
            "configuration": {"embedding_model": config.EMBEDDING_MODEL, "generation_model": config.OLLAMA_MODEL,
                              "ollama_url": config.OLLAMA_BASE_URL,
                              "dense_threshold": config.SCORE_THRESHOLD, "rrf_k": config.RRF_K,
                              "candidate_k": config.TOP_K * 2, "final_k": config.TOP_K,
                              "query_language": language,
                              "scope": "single-paper" if scope == "single" else "entire-library; title prepended",
                              "unanswerable_probes": "marker absence, not semantic proof" if scope == "single" else "excluded"},
            "caveats": ["Exploratory labels are not exhaustive or independent held-out data.",
                        "Evidence hit requires a complete normalized quote in one returned chunk; alternative valid evidence is unjudged.",
                        "Evidence-level recall is relative to the listed evidence units, not all relevant passages.",
                        "Precision and accuracy cannot be computed without complete relevance judgments.",
                        "Generation quality requires separate human review; no keyword score is called faithfulness.",
                        "Timing is single-run warm retrieval; no p95 or end-to-end SLA is implied."],
            "input_sha256": {str(p): sha256(p) for p in manifest_files},
            "code_sha256": {str(p): sha256(p) for p in code_files},
            "local_model_manifest_sha256": sha256(model_manifest) if model_manifest.exists() else None,
            "runtime": {"python": platform.python_version(),
                        "faiss_cpu": importlib.metadata.version("faiss-cpu"),
                        "sentence_transformers": importlib.metadata.version("sentence-transformers"),
                        "rank_bm25": importlib.metadata.version("rank-bm25")},
            "validation": validation,
            "retrieval_summary": summarize_retrieval(rows), "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--mode", choices=("retrieval", "all"), default="retrieval")
    parser.add_argument("--scope", choices=("single", "library"), default="single", help="Library queries prepend the target paper title")
    parser.add_argument("--language", choices=("en", "zh"), default="en")
    parser.add_argument("--zh-queries", type=Path, help="Optional Chinese query mapping JSON when --language zh")
    parser.add_argument("--ollama-url", help="Optional local HTTP endpoint for an isolated Ollama instance")
    parser.add_argument("--output", type=Path, required=True, help="New output path; never overwrite an existing file")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    report = run(args.dataset, args.mode, args.ollama_url, args.scope, args.language, args.zh_queries)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"SAVED {args.output}")
    for name, metrics in report["retrieval_summary"]["overall"].items():
        m = metrics["6"]
        print(f"{name} Hit@6={m['hit_rate']:.3f} annotated_evidence_recall@6={m['mean_annotated_evidence_recall']:.3f} MRR@6={m['mrr']:.3f} n={m['queries']}")
    failures = sum(row.get("generation", {}).get("status") == "error" for row in report["cases"])
    if failures:
        raise SystemExit(f"{failures} generation calls failed; report retained for diagnosis")


if __name__ == "__main__":
    main()
