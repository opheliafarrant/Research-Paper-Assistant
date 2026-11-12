"""Tests for evidence-aware retrieval metrics; no model calls or library writes."""
import unittest

from evaluation.rag_quality_eval import normalized, retrieval_metrics, summarize_retrieval, run, fuse_with_fallback
from evaluation.score_rag_quality_review import score


class RagQualityEvalTests(unittest.TestCase):
    def test_evidence_hit_recall_and_first_rank_are_distinct(self):
        evidence = [{"quote": "alpha evidence"}, {"quote": "beta evidence"}]
        chunks = [
            {"paper_id": "p", "text": "irrelevant"},
            {"paper_id": "other", "text": "alpha evidence"},
            {"paper_id": "p", "text": "ALPHA   evidence"},
            {"paper_id": "p", "text": "beta evidence"},
        ]
        at3 = retrieval_metrics(evidence, chunks, "p", 3)
        self.assertTrue(at3["hit"])
        self.assertEqual(at3["first_rank"], 3)
        self.assertAlmostEqual(at3["reciprocal_rank"], 1 / 3)
        self.assertEqual(at3["evidence_recall"], .5)
        self.assertEqual(retrieval_metrics(evidence, chunks, "p", 4)["evidence_recall"], 1.0)

    def test_missing_evidence_has_no_rank(self):
        result = retrieval_metrics([{"quote": "expected"}], [{"paper_id": "p", "text": "other"}], "p", 6)
        self.assertFalse(result["hit"])
        self.assertIsNone(result["first_rank"])
        self.assertEqual(result["reciprocal_rank"], 0)
        self.assertEqual(result["evidence_recall"], 0)

    def test_normalization_handles_whitespace_and_dash(self):
        self.assertEqual(normalized(" A–B \nC "), normalized("a-b c"))

    def test_unanswerable_probe_is_excluded_from_retrieval_relevance_metrics(self):
        scored = {str(k): {"hit": True, "evidence_recall": 1.0, "reciprocal_rank": 1.0} for k in (1, 3, 5, 6)}
        cases = [{"type": "term", "methods": {m: scored for m in ("dense", "bm25", "rrf")}},
                 {"type": "unanswerable_probe"}]
        result = summarize_retrieval(cases)
        self.assertEqual(result["overall"]["rrf"]["6"]["queries"], 1)
        self.assertEqual(result["overall"]["rrf"]["6"]["hit_rate"], 1.0)

    def test_review_precision_and_generation_are_scored_separately(self):
        results = {"cases": [{"id": "q1", "type": "term", "returned_ids": {"rrf": [1, 2, 3]},
                              "generation": {"status": "ok", "sources": [{"chunk_idx": 1}, {"chunk_idx": 2}, {"chunk_idx": 3}]}}]}
        review = {"cases": [{"id": "q1", "relevant_chunks": [2], "coverage": "none",
                             "grounding": "unsupported", "reason": "A relevant source did not yield a correct answer"}]}
        result = score(results, review)
        self.assertEqual(result["rrf_precision_at_k"]["3"], 1 / 3)
        self.assertEqual(result["rrf_hit_at_k"]["1"], 0)
        self.assertEqual(result["generation_coverage"], {"none": 1})

    def test_review_rejects_unknown_or_missing_labels(self):
        results = {"cases": [{"id": "q1", "type": "term", "returned_ids": {"rrf": [1]},
                              "generation": {"status": "ok", "sources": [{"chunk_idx": 1}]}}]}
        review = {"cases": [{"id": "q1", "relevant_chunks": [99], "coverage": "full", "grounding": "supported", "reason": "invalid"}]}
        with self.assertRaises(ValueError):
            score(results, review)

    def test_unreviewed_sources_are_not_reported_as_zero_precision(self):
        results = {"cases": [{"id": "q1", "type": "term", "returned_ids": {"rrf": [1]},
                              "generation": {"status": "ok", "sources": [{"chunk_idx": 1}]}}]}
        review = {"cases": [{"id": "q1", "coverage": "none", "grounding": "abstained", "reason": "no evidence"}]}
        result = score(results, review)
        self.assertEqual(result["rrf_reviewed_source_queries"], 0)
        self.assertIsNone(result["rrf_precision_at_k"]["6"])
        self.assertEqual(result["generation_grounding"], {"abstained": 1})

    def test_generation_refuses_nonlocal_endpoint(self):
        from pathlib import Path
        from unittest.mock import patch
        from research_assistant import config

        with patch.object(config, "OLLAMA_BASE_URL", "http://localhost:11434"):
            with self.assertRaises(ValueError):
                run(Path("unused"), "all", "https://untrusted.example")

    def test_chinese_fallback_only_when_fusion_is_empty(self):
        from unittest.mock import MagicMock

        chunk = {"paper_id": "p", "chunk_idx": 1, "text": "source", "_faiss_score": .35}
        index = MagicMock()
        index.search.return_value = [chunk]
        result, used = fuse_with_fallback("研究的局限？", [1, 0], [], [], index, "p")
        self.assertTrue(used)
        self.assertEqual(result[0]["chunk_idx"], 1)
        self.assertEqual(index.search.call_args.kwargs["min_score"], .30)
        index.reset_mock()
        result, used = fuse_with_fallback("研究的局限？", [1, 0], [chunk], [], index, "p")
        self.assertFalse(used)
        index.search.assert_not_called()
        result, used = fuse_with_fallback("What are the limitations?", [1, 0], [], [], index, "p")
        self.assertFalse(used)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
