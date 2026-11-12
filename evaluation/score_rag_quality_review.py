"""Score the explicitly reviewed RRF candidates and generated answers.

The review is a single AI-assistant judgment, not independent human ground truth.
Run: python -B -m evaluation.score_rag_quality_review
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
REVIEW = ROOT / "rag_quality_review_v1.json"


def score(results, review):
    result_cases = {case["id"]: case for case in results["cases"]}
    review_cases = {case["id"]: case for case in review["cases"]}
    if len(result_cases) != len(results["cases"]) or len(review_cases) != len(review["cases"]):
        raise ValueError("Duplicate case IDs")
    if set(result_cases) != set(review_cases):
        raise ValueError("Review must cover exactly the generated cases")
    answerable, probes = [], []
    for cid, item in review_cases.items():
        result = result_cases[cid]
        generation = result.get("generation", {})
        if generation.get("status") != "ok":
            raise ValueError(f"Generation did not complete for {cid}")
        source_ids = [chunk["chunk_idx"] for chunk in generation["sources"]]
        relevant = item.get("relevant_chunks")
        if relevant is not None and (len(relevant) != len(set(relevant)) or not set(relevant) <= set(source_ids)):
            raise ValueError(f"Invalid reviewed chunks for {cid}")
        if not item.get("reason"):
            raise ValueError(f"Missing review explanation for {cid}")
        if result["type"] == "unanswerable_probe":
            if item["coverage"] != "not_applicable" or item["grounding"] != "abstained" or not isinstance(item.get("abstention_correct"), bool):
                raise ValueError(f"Invalid no-answer judgment for {cid}")
            probes.append(item)
            continue
        if item["coverage"] not in {"full", "partial", "none"} or item["grounding"] not in {"supported", "unsupported", "uncertain", "abstained"}:
            raise ValueError(f"Invalid generation judgment for {cid}")
        if source_ids != result["returned_ids"]["rrf"]:
            raise ValueError(f"Generation context differs from recorded RRF results for {cid}")
        hits = [i in set(relevant) for i in source_ids] if relevant is not None else None
        answerable.append({"id": cid, "hits": hits, "coverage": item["coverage"], "grounding": item["grounding"]})
    if not answerable:
        raise ValueError("No reviewed answerable cases")
    precision, hit_rates = {}, {}
    judged = [row for row in answerable if row["hits"] is not None]
    for k in (1, 3, 5, 6):
        precision[str(k)] = sum(sum(row["hits"][:k]) / k for row in judged) / len(judged) if judged else None
        hit_rates[str(k)] = sum(any(row["hits"][:k]) for row in judged) / len(judged) if judged else None
    return {"answerable_count": len(answerable), "rrf_reviewed_source_queries": len(judged),
            "rrf_precision_at_k": precision, "rrf_hit_at_k": hit_rates,
            "relevant_chunks": sum(sum(row["hits"]) for row in judged),
            "returned_chunks": sum(len(row["hits"]) for row in judged),
            "generation_coverage": dict(Counter(row["coverage"] for row in answerable)),
            "generation_grounding": dict(Counter(row["grounding"] for row in answerable)),
            "unanswerable_probes": len(probes),
            "correct_abstentions": sum(item["abstention_correct"] for item in probes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=REVIEW)
    args = parser.parse_args()
    review = json.loads(args.review.read_text(encoding="utf-8"))
    result_path = ROOT / review["results_file"]
    actual_sha = hashlib.sha256(result_path.read_bytes()).hexdigest()
    if actual_sha != review["results_sha256"]:
        raise ValueError("Result file differs from the reviewed snapshot")
    results = json.loads(result_path.read_text(encoding="utf-8"))
    print(json.dumps(score(results, review), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
