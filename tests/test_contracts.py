"""Fast regression checks for the stable retrieval and language contracts."""

import unittest
from unittest.mock import patch
from pathlib import Path

import faiss
import numpy as np

from research_assistant.application.research import _classify_query
from research_assistant.domain.fusion import rrf_fuse
from research_assistant.infrastructure.indexes import FAISSIndex, BM25Index


class ContractTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from research_assistant.infrastructure import conversation_store

        directory = tempfile.TemporaryDirectory(prefix="rpa-contract-test-")
        self.addCleanup(directory.cleanup)
        patcher = patch.object(conversation_store, "DB_PATH", Path(directory.name) / "conversations.sqlite3")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_fusion_combines_results_without_mutating_inputs(self):
        first = {"paper_id": "a", "chunk_idx": 1, "text": "first"}
        second = {"paper_id": "b", "chunk_idx": 2, "text": "second"}
        result = rrf_fuse([first, second], [second], k=60, top_n=2)
        self.assertEqual([item["paper_id"] for item in result], ["b", "a"])
        self.assertGreater(result[0]["_rrf_score"], result[1]["_rrf_score"])
        self.assertNotIn("_rrf_score", second)
        self.assertIn("_confidence", result[0])
        self.assertIn("_confidence_badge", result[0])

    def test_duplicate_results_do_not_amplify_rrf_or_consume_rank(self):
        first = {"paper_id": "a", "chunk_idx": 1, "text": "first"}
        second = {"paper_id": "a", "chunk_idx": 2, "text": "second"}
        result = rrf_fuse([first, first, second], [], k=60, top_n=2)
        self.assertEqual(result[0]["_rrf_score"], round(1 / 61, 6))
        self.assertEqual(result[1]["_rrf_score"], round(1 / 62, 6))

    def test_search_deduplicates_without_losing_other_papers_or_chunks(self):
        from unittest.mock import MagicMock

        rows = [{"paper_id": "a", "chunk_idx": 0}, {"paper_id": "a", "chunk_idx": 0},
                {"paper_id": "b", "chunk_idx": 0}, {"paper_id": "a", "chunk_idx": 1}]
        vectors = np.array([[1, 0], [1, 0], [.9, .1], [.8, .2]], dtype="float32")
        index = FAISSIndex(index=faiss.IndexFlatIP(2))
        index.add(vectors, rows)
        self.assertEqual([(c["paper_id"], c["chunk_idx"]) for c in index.search(np.array([[1, 0]], dtype="float32"), k=3)],
                         [("a", 0), ("b", 0), ("a", 1)])
        model = MagicMock()
        model.get_scores.return_value = [4, 3, 2, 1]
        bm25 = BM25Index(bm25=model, tokens=[], chunks=rows)
        self.assertEqual([(c["paper_id"], c["chunk_idx"]) for c in bm25.search("query", k=3)],
                         [("a", 0), ("b", 0), ("a", 1)])

    def test_confidence_calculation_and_badge(self):
        from research_assistant.domain.fusion import calculate_confidence

        # Hybrid match (both dense and sparse hit)
        score_high, label_high, badge_high = calculate_confidence({
            "_faiss_score": 0.80, "_bm25_score": 15.0, "_rrf_score": 0.032,
        })
        self.assertGreaterEqual(score_high, 0.70)
        self.assertEqual(label_high, "高相关")
        self.assertTrue(badge_high.startswith("高相关"))

        # Moderate match
        score_mid, label_mid, badge_mid = calculate_confidence({
            "_faiss_score": 0.55, "_rrf_score": 0.016,
        })
        self.assertGreaterEqual(score_mid, 0.50)
        self.assertEqual(label_mid, "中相关")
        self.assertTrue(badge_mid.startswith("中相关"))

        # Low reference match
        score_low, label_low, badge_low = calculate_confidence({"_rrf_score": 0.005})
        self.assertLess(score_low, 0.50)
        self.assertEqual(label_low, "参考")

    def test_paper_filter_and_chinese_fallback_keep_default_threshold(self):
        vectors = np.array([[0.35, (1 - 0.35 ** 2) ** 0.5],
                            [0.9, (1 - 0.9 ** 2) ** 0.5]], dtype="float32")
        index = FAISSIndex(index=faiss.IndexFlatIP(2))
        index.add(vectors, [{"paper_id": "target"}, {"paper_id": "other"}])
        query = np.array([[1, 0]], dtype="float32")
        self.assertEqual(index.search(query, k=1, paper_id="target"), [])
        self.assertEqual(index.search(query, k=1, paper_id="target", min_score=0.30)[0]["paper_id"], "target")
        self.assertEqual(index.search(query, k=1)[0]["paper_id"], "other")

    def test_chinese_and_english_questions_are_research(self):
        for question in ("这篇论文的主要方法是什么？", "What does it propose?"):
            with self.subTest(question=question):
                self.assertEqual(_classify_query(question)[0], "research")

    def test_page_configuration_is_applied_for_each_session(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest

        with patch.object(st, "set_page_config", wraps=st.set_page_config) as configure:
            for _ in range(2):
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "research_assistant" / "presentation" / "streamlit_app.py")).run(timeout=60)
                self.assertFalse(app.exception)
                self.assertTrue(any(button.label == "获取新论文" for button in app.sidebar.button))
            self.assertEqual(configure.call_count, 2)

    def test_deep_dive_controls_are_available(self):
        from dataclasses import replace
        from streamlit.testing.v1 import AppTest
        from research_assistant.bootstrap import build_application

        services = replace(
            build_application(),
            list_papers=lambda: [{"paper_id": "p1", "title": "Sample Paper"}],
        )
        with patch("research_assistant.bootstrap.build_application", return_value=services):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "research_assistant" / "presentation" / "streamlit_app.py")).run(timeout=60)
            app.sidebar.radio[0].set_value("单篇精读").run(timeout=60)
            self.assertFalse(app.exception)
            self.assertTrue(app.selectbox)
            self.assertTrue({"生成摘要", "清除摘要", "核心方法", "实验对比", "局限与未来"}.issubset({button.label for button in app.button}))

    def test_summarise_aspects_customize_prompt_and_system(self):
        from unittest.mock import MagicMock
        from research_assistant.application import research
        from research_assistant.domain.contracts import ResearchDependencies

        mock_deps = ResearchDependencies(
            load_indexes=MagicMock(return_value=(
                MagicMock(search=MagicMock(return_value=[{"paper_id": "p1", "page": 1, "chunk_idx": 0, "title": "Paper", "text": "Content"}])),
                MagicMock(search=MagicMock(return_value=[]))
            )),
            embed=MagicMock(return_value=np.zeros((1, 384), dtype="float32")),
            generate=MagicMock(return_value="Generated extraction"),
        )
        for aspect, keyword in [
            ("methods", "核心方法与算法架构"),
            ("experiments", "实验评估与对比分析"),
            ("limitations", "研究局限性与未来方向"),
            ("summary", "结构化摘要"),
        ]:
            with self.subTest(aspect=aspect):
                res = research.summarise("p1", "Paper", aspect=aspect, dependencies=mock_deps)
                self.assertEqual(res, "Generated extraction")
                last_call = mock_deps.generate.call_args
                prompt_text = last_call.kwargs.get("prompt", "") or (last_call.args[0] if last_call.args else "")
                self.assertIn(keyword, prompt_text)

    def test_summary_sources_are_exactly_the_chunks_sent_to_generator(self):
        from unittest.mock import MagicMock
        from research_assistant.application import research
        from research_assistant.domain.contracts import ResearchDependencies

        chunks = [{"paper_id": "p1", "page": 2, "chunk_idx": 4, "title": "Paper", "text": "actual evidence"}]
        deps = ResearchDependencies(load_indexes=MagicMock(return_value=(object(), object())),
                                    embed=MagicMock(), generate=MagicMock(return_value="summary"))
        with patch.object(research, "retrieve_aspect_chunks", return_value=chunks) as retrieve:
            result = research.summarise_with_sources("p1", "Paper", "methods", dependencies=deps)
        self.assertEqual(result, {"text": "summary", "sources": chunks})
        self.assertIn("actual evidence", deps.generate.call_args.args[0])
        retrieve.assert_called_once()


if __name__ == "__main__":
    unittest.main()
