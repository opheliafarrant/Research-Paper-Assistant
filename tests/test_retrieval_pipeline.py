"""Unit tests for cross-lingual retrieval, noise filtering, and context citation formatting."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import faiss
import numpy as np

from research_assistant.application import research
from research_assistant.domain.contracts import ResearchDependencies
from research_assistant.domain.fusion import (
    filter_bm25_by_cosine,
    is_boilerplate_chunk,
    mentions_boilerplate,
    prune_weak_tail,
    rrf_fuse,
)
from research_assistant.infrastructure.indexes import BM25Index, FAISSIndex


class RetrievalPipelineTests(unittest.TestCase):
    def test_boilerplate_filter_removes_checklist_noise_without_dropping_mixed_prose(self):
        body = {
            "paper_id": "p1",
            "chunk_idx": 78,
            "page": 8,
            "text": "Future work extends evaluation to household objects and SPOT. ACKNOWLEDGEMENT We thank... REFERENCES",
        }
        checklist = {
            "paper_id": "p1",
            "chunk_idx": 145,
            "page": 20,
            "text": "Guidelines: • The answer [N/A] means the paper does not include experiments. Question: Reproducibility Answer: [N/A] Justification: None.",
        }
        ethics_bp = {
            "paper_id": "p1",
            "chunk_idx": 179,
            "page": 24,
            "text": "Authors must adhere to the NeurIPS Code of Ethics and the guidelines for their institution. Do not include any information that would break anonymity.",
        }
        bib = {
            "paper_id": "p1",
            "chunk_idx": 160,
            "page": 18,
            "text": "[1] Alice (2022) Foo. [2] Bob (2023) Bar. [3] Carol (2024) Baz. [4] Dan (2025) Qux.",
        }
        self.assertFalse(is_boilerplate_chunk(body["text"]))
        self.assertTrue(is_boilerplate_chunk(checklist["text"]))
        self.assertTrue(is_boilerplate_chunk(ethics_bp["text"]))
        self.assertTrue(is_boilerplate_chunk(bib["text"]))

        fused = rrf_fuse([checklist, ethics_bp, body, bib], [checklist], k=60, top_n=6, filter_boilerplate=True)
        self.assertEqual([c["chunk_idx"] for c in fused], [78])

        # Explicitly asking about checklist retains checklist results.
        self.assertTrue(mentions_boilerplate("What does the NeurIPS checklist say?"))
        retained = rrf_fuse(
            [checklist, body],
            [],
            k=60,
            top_n=6,
            filter_boilerplate=True,
            query_mentions_boilerplate=True,
        )
        self.assertEqual([c["chunk_idx"] for c in retained], [145, 78])

    def test_bm25_cosine_filter_and_weak_tail_pruning(self):
        chunks = [
            {"paper_id": "p1", "chunk_idx": i, "text": f"chunk {i}", "page": i + 1, "title": "Paper"}
            for i in range(6)
        ]
        vecs = np.array(
            [
                [1.0, 0.0],   # ch 0: cos 1.0
                [0.8, 0.6],   # ch 1: cos 0.8
                [0.6, 0.8],   # ch 2: cos 0.6
                [0.5, 0.866], # ch 3: cos 0.5 (strong dense)
                [0.42, 0.907],# ch 4: cos 0.42 (weak single-path)
                [0.15, 0.988],# ch 5: cos 0.15 (noise)
            ],
            dtype="float32",
        )
        fi = FAISSIndex(index=faiss.IndexFlatIP(2))
        fi.add(vecs, chunks)
        qv = np.array([[1.0, 0.0]], dtype="float32")
        cos_map = fi.cosine_scores(qv, chunks)

        filtered_bm25 = filter_bm25_by_cosine([chunks[5], chunks[4], chunks[0]], cos_map, min_cosine=0.30)
        self.assertEqual([c["chunk_idx"] for c in filtered_bm25], [4, 0])

        fa_res = [dict(chunks[i], _faiss_score=float(cos_map[("p1", i)])) for i in (0, 1, 2, 3, 4)]
        bm_res = [dict(chunks[0], _bm25_score=10.0), dict(chunks[4], _bm25_score=None)]
        fused = rrf_fuse(fa_res, bm_res[:1], k=60, top_n=6)
        pruned = prune_weak_tail(fused, cos_map, min_keep=3, single_path_min_cosine=0.48)
        self.assertEqual([c["chunk_idx"] for c in pruned], [0, 1, 2, 3])

    def test_chinese_query_translates_when_valid_and_falls_back_gracefully_when_invalid(self):
        chunks = [
            {
                "paper_id": "2611.09868",
                "chunk_idx": 24,
                "page": 3,
                "page_end": 4,
                "page_spans": [{"page": 3, "start": 0, "end": 18}, {"page": 4, "start": 18, "end": 59}],
                "title": "CodeQuant2",
                "text": "Early decoding.   First 30 steps std is 20% higher overall.",
            }
        ]
        # Build tiny context check
        ctx = research._build_context(chunks)
        self.assertIn("Excerpt 1 (Paper ID: 2611.09868, Page: 3-4, Title: CodeQuant2):", ctx)
        self.assertIn("[Page 3]", ctx)
        self.assertIn("[Page 4]", ctx)

        mock_faiss = MagicMock()
        mock_faiss.search.return_value = [dict(chunks[0], _faiss_score=0.72)]
        mock_faiss.cosine_scores.return_value = {("2611.09868", 24): 0.72}
        mock_bm25 = MagicMock()
        mock_bm25.search.return_value = [dict(chunks[0], _bm25_score=8.5)]

        calls = []

        def fake_generate(prompt, system=""):
            calls.append((prompt, system))
            if "Translate to English" in prompt:
                return "What activation fluctuation occurs in CodeQuant2 early decoding?"
            return "前30步标准差高约20%（2611.09868, p.4）。"

        deps = ResearchDependencies(
            load_indexes=MagicMock(return_value=(mock_faiss, mock_bm25)),
            embed=MagicMock(return_value=np.ones((1, 2), dtype="float32")),
            generate=fake_generate,
        )
        ans = research.answer("CodeQuant2 早期解码阶段观察到什么激活波动？", paper_id="2611.09868", dependencies=deps)
        self.assertEqual(len(calls), 2)
        self.assertIn("codequant2", mock_bm25.search.call_args.args[0].lower())
        self.assertEqual(ans["sources"][0]["page_end"], 4)
        # Ensure answer prompts do not leak a concrete unrelated Paper ID example.
        self.assertNotIn("2611.09868, p.", calls[1][0])
        self.assertNotIn("2611.09868, p.", calls[1][1])

        self.assertFalse(research._acceptable_translation("高层奖励目标", "TaskCompletionGoalInRewards"))
        self.assertFalse(research._acceptable_translation("未来研究问题", "What-future-research-questions-do-papers-address"))

        # Invalid/broken translation retaining Chinese falls back without crashing
        bad_deps = ResearchDependencies(
            load_indexes=MagicMock(return_value=(mock_faiss, mock_bm25)),
            embed=MagicMock(return_value=np.ones((1, 2), dtype="float32")),
            generate=MagicMock(return_value="依然包含中文的无效输出"),
        )
        ans_bad = research.answer("早期解码有什么问题？", dependencies=bad_deps)
        self.assertEqual(ans_bad["sources"][0]["chunk_idx"], 24)

    def test_english_single_paper_generic_query_falls_back_to_lower_threshold(self):
        fallback_chunk = {
            "paper_id": "2611.09892",
            "chunk_idx": 11,
            "page": 2,
            "title": "On Agentic Manipulation Policies",
            "text": "We introduce AMP1 for generalizable robotic manipulation.",
            "_faiss_score": 0.34,
        }
        weak_bm25_chunk = {
            "paper_id": "2611.09892",
            "chunk_idx": 99,
            "page": 8,
            "title": "On Agentic Manipulation Policies",
            "text": "Additional method keyword match with low semantic similarity.",
            "_bm25_score": 6.2,
        }

        mock_faiss = MagicMock()

        def fake_faiss_search(vec, k=12, paper_id=None, min_score=None):
            if min_score == 0.30:
                return [fallback_chunk]
            return []

        mock_faiss.search.side_effect = fake_faiss_search
        mock_faiss.cosine_scores.side_effect = lambda vec, chunks: {
            ("2611.09892", 99): 0.20,
            ("2611.09892", 11): 0.34,
        }
        mock_bm25 = MagicMock()
        mock_bm25.search.return_value = [weak_bm25_chunk]

        embed_fn = MagicMock(return_value=np.ones((1, 2), dtype="float32"))

        # Single-paper mode recovers via min_score=0.30 fallback when dense/BM25 both yield empty fusion.
        chunks, meta = research.retrieve_chunks(
            "What method does the paper propose?",
            paper_id="2611.09892",
            faiss_idx=mock_faiss,
            bm25_idx=mock_bm25,
            embed_fn=embed_fn,
            generate_fn=None,
        )
        self.assertEqual([c["chunk_idx"] for c in chunks], [11])
        self.assertEqual(meta["faiss_count"], 0)
        self.assertEqual(meta["bm25_count"], 0)

        # Library-wide English search keeps the stricter default threshold when fusion is empty.
        mock_faiss.search.reset_mock()
        chunks_global, _ = research.retrieve_chunks(
            "What method does the paper propose?",
            paper_id=None,
            faiss_idx=mock_faiss,
            bm25_idx=mock_bm25,
            embed_fn=embed_fn,
            generate_fn=None,
        )
        self.assertEqual(chunks_global, [])
        self.assertEqual(mock_faiss.search.call_count, 1)


if __name__ == "__main__":
    unittest.main()
