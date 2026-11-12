"""Regression checks for release-hardening fixes (error boundaries, atomic download, Ollama options)."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import requests

from research_assistant import config
from research_assistant.application import ingestion
from research_assistant.domain.contracts import IngestionDependencies
from research_assistant.infrastructure import arxiv, ollama, pdf


class OllamaHardeningTests(unittest.TestCase):
    def test_generate_sets_repetition_and_length_bounds(self):
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [b'{"response": "hello", "done": true}']
        with patch.object(ollama.requests, "post", return_value=mock_resp) as post:
            out = ollama.generate("test prompt", system="sys")
        self.assertEqual(out, "hello")
        options = post.call_args.kwargs["json"]["options"]
        self.assertEqual(options["temperature"], 0.0)
        self.assertEqual(options["seed"], 42)
        self.assertEqual(options["num_predict"], 1024)
        self.assertEqual(options["repeat_penalty"], 1.1)

    def test_generate_wraps_timeout_and_http_errors(self):
        with patch.object(ollama.requests, "post", side_effect=requests.exceptions.Timeout("slow")):
            with self.assertRaises(RuntimeError) as ctx:
                ollama.generate("hi")
            self.assertIn("timed out", str(ctx.exception))

        bad_resp = MagicMock()
        bad_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
        with patch.object(ollama.requests, "post", return_value=bad_resp):
            with self.assertRaises(RuntimeError) as ctx:
                ollama.generate("hi")
            self.assertIn("failed", str(ctx.exception))


class IngestionAndDownloadHardeningTests(unittest.TestCase):
    def test_ingestion_run_returns_structured_counts(self):
        empty_deps = IngestionDependencies(
            prepare=lambda: None,
            fetch=lambda: [],
            parse=lambda _p: [],
            add_chunks=lambda _c: None,
            record=lambda _ids: None,
        )
        self.assertEqual(ingestion.run(dependencies=empty_deps), {"fetched": 0, "ingested": [], "failed": []})

        papers = [{"paper_id": "ok-1", "title": "Ok"}, {"paper_id": "bad-2", "title": "Bad"}]

        def parse(paper):
            if paper["paper_id"] == "bad-2":
                raise ValueError("corrupt pdf")
            return [{"paper_id": "ok-1", "chunk_idx": 0, "text": "x", "title": "Ok", "page": 1}]

        mixed_deps = IngestionDependencies(
            prepare=lambda: None,
            fetch=lambda: papers,
            parse=parse,
            add_chunks=lambda _c: None,
            record=lambda _ids: None,
        )
        summary = ingestion.run(dependencies=mixed_deps)
        self.assertEqual(summary, {"fetched": 2, "ingested": ["ok-1"], "failed": ["bad-2"]})

    def test_download_pdf_does_not_leave_partial_file_on_failure(self):
        class FailingStream:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                raise OSError("network drop")

        with tempfile.TemporaryDirectory(prefix="rpa-dl-test-") as temp:
            dest = Path(temp) / "broken.pdf"
            with patch("urllib.request.urlopen", return_value=FailingStream()), patch("time.sleep"):
                with self.assertRaises(RuntimeError):
                    pdf._download_pdf("https://example.com/broken.pdf", dest)
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_fetch_new_papers_skips_pause_after_last_topic(self):
        pauses: list[float] = []

        def fake_fetch(topic, _max_results, _ingested, *, pause_after=20.0):
            pauses.append(pause_after)
            return []

        with (
            patch.object(config, "ARXIV_TOPICS", ["cs.AI", "cs.LG", "cs.CL"]),
            patch.object(arxiv, "_load_ingested", return_value=set()),
            patch.object(arxiv, "_fetch_topic", side_effect=fake_fetch),
        ):
            arxiv.fetch_new_papers()
        self.assertEqual(pauses, [20.0, 20.0, 0.0])


class StreamlitErrorBoundaryTests(unittest.TestCase):
    def setUp(self):
        from research_assistant.infrastructure import conversation_store

        directory = tempfile.TemporaryDirectory(prefix="rpa-ui-err-test-")
        self.addCleanup(directory.cleanup)
        patcher = patch.object(conversation_store, "DB_PATH", Path(directory.name) / "conversations.sqlite3")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_ui_displays_friendly_errors_when_backend_fails(self):
        from streamlit.testing.v1 import AppTest
        from research_assistant.bootstrap import build_application

        def fail_answer(_query, paper_id=None):
            raise RuntimeError("Cannot reach Ollama at http://localhost:11434.")

        def fail_summary(_pid, _title, aspect=None):
            raise RuntimeError("Ollama request timed out after 120s.")

        def partial_ingest():
            return {"fetched": 2, "ingested": ["p1"], "failed": ["p2"]}

        services = replace(
            build_application(),
            list_papers=lambda: [{"paper_id": "p1", "title": "Sample Paper"}],
            answer=fail_answer,
            summarise_with_sources=fail_summary,
            ingest=partial_ingest,
        )
        script = str(Path(__file__).resolve().parents[1] / "research_assistant" / "presentation" / "streamlit_app.py")
        with patch("research_assistant.bootstrap.build_application", return_value=services):
            app = AppTest.from_file(script).run(timeout=60)
            app.chat_input[0].set_value("测试问答异常").run(timeout=60)
            self.assertFalse(app.exception)
            self.assertTrue(any("生成回答失败" in e.value for e in app.error))

            app.sidebar.radio[0].set_value("单篇精读").run(timeout=60)
            next(b for b in app.button if b.label == "核心方法").click().run(timeout=60)
            self.assertFalse(app.exception)
            self.assertTrue(any("核心方法与算法架构" in e.value and "失败" in e.value for e in app.error))

            next(b for b in app.sidebar.button if b.label == "获取新论文").click().run(timeout=60)
            self.assertFalse(app.exception)
            self.assertTrue(any("另有 1 篇失败" in w.value for w in app.warning))


if __name__ == "__main__":
    unittest.main()
