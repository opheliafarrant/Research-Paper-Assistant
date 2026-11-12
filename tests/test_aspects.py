"""Unit tests for aspect strategies and targeted context retrieval."""

import unittest
from unittest.mock import MagicMock
import numpy as np

from research_assistant.application.aspects import (
    get_aspect_strategy,
    _expand_contiguous_chunks,
    retrieve_aspect_chunks,
    AspectStrategy,
)
from research_assistant.domain.contracts import ResearchDependencies


class AspectTests(unittest.TestCase):
    def test_get_aspect_strategy_fallbacks_to_summary(self):
        self.assertEqual(get_aspect_strategy("methods").aspect_id, "methods")
        self.assertEqual(get_aspect_strategy("experiments").aspect_id, "experiments")
        self.assertEqual(get_aspect_strategy("limitations").aspect_id, "limitations")
        self.assertEqual(get_aspect_strategy("summary").aspect_id, "summary")
        self.assertEqual(get_aspect_strategy(None).aspect_id, "summary")
        self.assertEqual(get_aspect_strategy("unknown").aspect_id, "summary")

    def test_build_prompt_structure(self):
        strategy = get_aspect_strategy("limitations")
        prompt = strategy.build_prompt("Test Paper", "2026.0001", "Context sample")
        self.assertIn("Paper title: Test Paper", prompt)
        self.assertIn("Paper ID:    2026.0001", prompt)
        self.assertIn("Context sample", prompt)
        self.assertIn("研究局限性与未来方向", prompt)
        self.assertIn("核心局限与失效边界", prompt)

    def test_expand_contiguous_chunks_bridges_adjacent_sections(self):
        fused = [
            {"paper_id": "p1", "page": 12, "chunk_idx": 127, "text": "Limitation part 1"},
        ]
        meta = [
            {"paper_id": "p1", "page": 12, "chunk_idx": 126, "text": "Prior section ending"},
            {"paper_id": "p1", "page": 12, "chunk_idx": 127, "text": "Limitation part 1"},
            {"paper_id": "p1", "page": 12, "chunk_idx": 128, "text": "Limitation part 2"},
            {"paper_id": "p1", "page": 15, "chunk_idx": 150, "text": "Unrelated"},
        ]
        expanded = _expand_contiguous_chunks(fused, meta, paper_id="p1", max_total=5)
        indices = [c["chunk_idx"] for c in expanded]
        self.assertEqual(indices, [126, 127, 128])

    def test_expand_contiguous_chunks_respects_max_total(self):
        fused = [{"paper_id": "p1", "page": 1, "chunk_idx": 1}]
        meta = [{"paper_id": "p1", "page": 1, "chunk_idx": i} for i in range(1, 10)]
        expanded = _expand_contiguous_chunks(fused, meta, paper_id="p1", max_total=2)
        self.assertEqual(len(expanded), 2)

    def test_expansion_preserves_ranked_hits_before_neighbors(self):
        meta = [{"paper_id": "p1", "page": 1, "chunk_idx": i, "text": str(i)} for i in range(20)]
        fused = [dict(meta[i]) for i in (1, 5, 9, 13)]
        expanded = _expand_contiguous_chunks(fused, meta, paper_id="p1", max_total=6)
        indices = {c["chunk_idx"] for c in expanded}
        self.assertEqual(len(expanded), 6)
        self.assertTrue({1, 5, 9, 13}.issubset(indices))

    def test_keyword_supported_anchor_receives_continuation_budget(self):
        meta = [{"paper_id": "p1", "page": 1, "chunk_idx": i, "text": str(i)} for i in range(20)]
        fused = [dict(meta[1]), dict(meta[8], _bm25_score=5.0), dict(meta[15])]
        expanded = _expand_contiguous_chunks(fused, meta, paper_id="p1", max_total=6)
        self.assertTrue({8, 9, 10, 11}.issubset({c["chunk_idx"] for c in expanded}))

    def test_retrieve_aspect_chunks_calls_hybrid_indexes(self):
        faiss_chunk = {"paper_id": "p1", "page": 1, "chunk_idx": 0, "text": "dense match"}
        bm25_chunk = {"paper_id": "p1", "page": 1, "chunk_idx": 1, "text": "sparse match"}
        
        mock_faiss = MagicMock()
        mock_faiss.search.return_value = [faiss_chunk]
        mock_faiss.meta = [faiss_chunk, bm25_chunk]

        mock_bm25 = MagicMock()
        mock_bm25.search.return_value = [bm25_chunk]

        deps = ResearchDependencies(
            load_indexes=MagicMock(return_value=(mock_faiss, mock_bm25)),
            embed=MagicMock(return_value=np.zeros((1, 384), dtype="float32")),
            generate=MagicMock(return_value="OK"),
        )
        strategy = get_aspect_strategy("limitations")
        results = retrieve_aspect_chunks("p1", "Title", strategy, dependencies=deps, top_k=2)
        self.assertGreaterEqual(len(results), 2)
        self.assertTrue(mock_faiss.search.called)
        self.assertTrue(mock_bm25.search.called)


if __name__ == "__main__":
    unittest.main()
