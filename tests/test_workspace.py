"""State and real-PDF regression checks for the three-pane workspace."""

from pathlib import Path
import tempfile
import unittest

import pymupdf as fitz

from research_assistant import config
from research_assistant.infrastructure import pdf_preview
from research_assistant.presentation.conversations import (
    active_conversation, history_for, name_from_question, new_conversation, select_conversation,
)


class ConversationTests(unittest.TestCase):
    def setUp(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from research_assistant.infrastructure import conversation_store, indexes

        directory = tempfile.TemporaryDirectory(prefix="rpa-conversations-test-")
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        pdf_dir = root / "pdfs"
        pdf_dir.mkdir()
        for pid, body in (
            ("2611.09892", "Agentic Manipulation Policies page 1"),
            ("2611.09891", "CPU-ARC Reasoning Caching page 1"),
        ):
            doc = fitz.open()
            doc.new_page().insert_text((50, 50), body)
            doc.new_page().insert_text((50, 50), f"{pid} page 2")
            doc.save(str(pdf_dir / f"{pid}.pdf"))
            doc.close()

        sample_meta = [
            {"paper_id": "2611.09892", "title": "On Agentic Manipulation Policies"},
            {"paper_id": "2611.09891", "title": "CPU-ARC: Agentic Reasoning Caching"},
        ]
        sample_indexes = (SimpleNamespace(meta=sample_meta), SimpleNamespace(chunks=sample_meta))

        for target, attr, val in (
            (conversation_store, "DB_PATH", root / "conversations.sqlite3"),
            (config, "PDF_DIR", pdf_dir),
            (indexes, "load_indexes", lambda: sample_indexes),
        ):
            patcher = patch.object(target, attr, val)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_conversations_and_scopes_are_independent(self):
        state = {}
        first = active_conversation(state)
        history_for(first, "ask_anything").append({"role": "user", "content": "first"})
        name_from_question(first, "这篇论文的研究方法是什么？")
        second = new_conversation(state)
        self.assertNotEqual(first["id"], second["id"])
        self.assertEqual(history_for(second, "ask_anything"), [])
        self.assertEqual(history_for(first, "deep_dive", "1234.5678"), [])
        select_conversation(state, first["id"])
        self.assertEqual(active_conversation(state)["title"], "这篇论文的研究方法是什么？")
        self.assertEqual(history_for(active_conversation(state), "ask_anything")[0]["content"], "first")
        with self.assertRaises(ValueError):
            select_conversation(state, "missing")

    def test_delete_and_rename_conversation(self):
        from research_assistant.presentation.conversations import (
            delete_conversation, rename_conversation, set_summary, summary_for, clear_summary_for,
        )
        state = {}
        first = active_conversation(state)
        self.assertTrue(rename_conversation(first, "AMP1 策略研究"))
        self.assertEqual(first["title"], "AMP1 策略研究")
        self.assertFalse(rename_conversation(first, "   "))
        set_summary(first, "p1", "Method text", "核心方法")
        self.assertEqual(summary_for(first, "p1")["text"], "Method text")

        second = new_conversation(state)
        self.assertIsNone(summary_for(second, "p1"))

        # Delete first conversation
        delete_conversation(state, first["id"])
        self.assertEqual(len(state["conversations"]), 1)
        self.assertEqual(state["active_conversation_id"], second["id"])

        # Delete only remaining conversation resets to a fresh one
        delete_conversation(state, second["id"])
        self.assertEqual(len(state["conversations"]), 1)
        self.assertEqual(active_conversation(state)["title"], "新对话")

    def test_conversation_stats_and_mode_icons(self):
        from research_assistant.presentation.conversations import conversation_stats, mode_icon, set_summary
        state = {}
        conv = active_conversation(state)
        self.assertEqual(mode_icon("论文问答"), ":material/chat:")
        self.assertEqual(mode_icon("单篇精读"), ":material/article:")
        self.assertEqual(conversation_stats(conv), {"messages": 0, "summaries": 0})
        history_for(conv, "ask_anything").append({"role": "user", "content": "hi"})
        set_summary(conv, "p1", "summary text", "核心方法")
        self.assertEqual(conversation_stats(conv), {"messages": 1, "summaries": 1})


    def test_switch_restores_reading_mode_and_selected_paper(self):
        from streamlit.testing.v1 import AppTest

        root = Path(__file__).resolve().parents[1]
        app = AppTest.from_file(str(root / "research_assistant" / "presentation" / "streamlit_app.py")).run(timeout=60)
        first_id = app.session_state["active_conversation_id"]
        app.sidebar.radio[0].set_value("单篇精读").run(timeout=60)
        selected = app.selectbox[0].value
        next(button for button in app.sidebar.button if button.label == "新建对话").click().run(timeout=60)
        self.assertEqual(app.sidebar.radio[0].value, "论文问答")
        next(button for button in app.sidebar.button if button.key == f"conversation_{first_id}").click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.sidebar.radio[0].value, "单篇精读")
        self.assertEqual(app.selectbox[0].value, selected)
        next(button for button in app.sidebar.button if button.label == "清空对话记录").click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.sidebar.radio[0].value, "论文问答")
        self.assertEqual(len(app.session_state["conversations"]), 1)

    def test_switch_back_and_forth_with_separate_histories(self):
        from streamlit.testing.v1 import AppTest

        root = Path(__file__).resolve().parents[1]
        app = AppTest.from_file(str(root / "research_assistant" / "presentation" / "streamlit_app.py")).run(timeout=60)
        first = app.session_state["conversations"][0]
        first_id = first["id"]
        history_for(first, "ask_anything").append({"role": "user", "content": "第一段对话"})
        next(b for b in app.sidebar.button if b.label == "新建对话").click().run(timeout=60)
        second_id = app.session_state["active_conversation_id"]
        second = next(c for c in app.session_state["conversations"] if c["id"] == second_id)
        history_for(second, "ask_anything").append({"role": "user", "content": "第二段对话"})
        app.run(timeout=60)
        for target, expected in ((first_id, "第一段对话"), (second_id, "第二段对话"), (first_id, "第一段对话")):
            next(b for b in app.sidebar.button if b.key == f"conversation_{target}").click().run(timeout=60)
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["active_conversation_id"], target)
            self.assertEqual(app.chat_message[0].markdown[0].value, expected)

    def test_quick_read_results_are_conversation_records(self):
        from dataclasses import replace
        from unittest.mock import patch
        from streamlit.testing.v1 import AppTest
        from research_assistant.bootstrap import build_application

        services = replace(build_application(), summarise_with_sources=lambda paper_id, title, aspect: {
            "text": f"结果：{aspect}",
            "sources": [{"paper_id": paper_id, "title": title, "page": 1, "chunk_idx": 0, "text": "paper evidence"}],
        })
        root = Path(__file__).resolve().parents[1]
        with patch("research_assistant.bootstrap.build_application", return_value=services):
            app = AppTest.from_file(str(root / "research_assistant" / "presentation" / "streamlit_app.py")).run(timeout=60)
            app.sidebar.radio[0].set_value("单篇精读").run(timeout=60)
            first_id = app.session_state["active_conversation_id"]
            next(b for b in app.button if b.label == "实验对比").click().run(timeout=60)
            self.assertTrue(any("结果：experiments" in m.markdown[0].value for m in app.chat_message))
            self.assertTrue(any("实验对比与评估结果" in b.label for b in app.sidebar.button))
            next(b for b in app.button if b.label == "核心方法").click().run(timeout=60)
            self.assertEqual(len(app.chat_message), 2)
            self.assertEqual(len([b for b in app.button if b.label == "查看原文"]), 2)
            app.sidebar.toggle[0].set_value(False).run(timeout=60)
            next(b for b in app.button if b.key == "message_0_source_1").click().run(timeout=60)
            self.assertTrue(app.sidebar.toggle[0].value)
            self.assertEqual(app.session_state["source_view"], "论文")
            self.assertEqual(app.session_state["preview_page"], 1)
            next(b for b in app.sidebar.button if b.label == "新建对话").click().run(timeout=60)
            next(b for b in app.sidebar.button if b.key == f"conversation_{first_id}").click().run(timeout=60)
            self.assertFalse(app.exception)
            self.assertEqual([m.markdown[0].value.splitlines()[-1] for m in app.chat_message],
                             ["结果：experiments", "结果：methods"])
            self.assertEqual(len([b for b in app.button if b.label == "查看原文"]), 2)

    def test_restart_loads_conversations_and_reading_state(self):
        from dataclasses import replace
        from unittest.mock import patch
        from streamlit.testing.v1 import AppTest
        from research_assistant.bootstrap import build_application

        services = replace(build_application(),
                           summarise_with_sources=lambda paper_id, title, aspect: {
                               "text": f"持久化结果：{aspect}",
                               "sources": [{"paper_id": paper_id, "title": title, "page": 1,
                                            "chunk_idx": 0, "text": "保存的来源"}]},
                           answer=lambda query, paper_id=None: {"answer": "持久化回答", "sources": [], "query": query})
        root = Path(__file__).resolve().parents[1]
        script = str(root / "research_assistant" / "presentation" / "streamlit_app.py")
        with patch("research_assistant.bootstrap.build_application", return_value=services):
            first = AppTest.from_file(script).run(timeout=60)
            first.sidebar.radio[0].set_value("单篇精读").run(timeout=60)
            selected_paper = first.selectbox[0].value
            first_id = first.session_state["active_conversation_id"]
            first.number_input[0].set_value(2).run(timeout=60)
            next(b for b in first.button if b.label == "实验对比").click().run(timeout=60)
            first.chat_input[0].set_value("持久化问题").run(timeout=60)
            next(b for b in first.sidebar.button if b.label == "新建对话").click().run(timeout=60)
            second_id = first.session_state["active_conversation_id"]
            second = AppTest.from_file(script).run(timeout=60)
            self.assertEqual(len(second.session_state["conversations"]), 2)
            self.assertEqual(second.session_state["active_conversation_id"], second_id)
            next(b for b in second.sidebar.button if b.key == f"conversation_{first_id}").click().run(timeout=60)
            self.assertFalse(second.exception)
            self.assertEqual(second.sidebar.radio[0].value, "单篇精读")
            self.assertEqual(second.selectbox[0].value, selected_paper)
            self.assertEqual(second.number_input[0].value, 2)
            self.assertTrue(any("持久化结果：experiments" in m.markdown[0].value for m in second.chat_message))
            self.assertTrue(any(b.label == "查看原文" for b in second.button))
            self.assertTrue(any("持久化问题" in m.markdown[0].value for m in second.chat_message))
            self.assertTrue(any("持久化回答" in m.markdown[0].value for m in second.chat_message))
            next(b for b in second.sidebar.button if b.key == f"btn_delete_{second_id}").click().run(timeout=60)
            third = AppTest.from_file(script).run(timeout=60)
            self.assertEqual([c["id"] for c in third.session_state["conversations"]], [first_id])

    def test_conflicting_session_save_does_not_overwrite(self):
        from research_assistant.infrastructure import conversation_store

        item = {"id": "same", "title": "原始"}
        self.assertEqual(conversation_store.save(item, None), 1)
        item["title"] = "窗口一"
        self.assertEqual(conversation_store.save(item, 1), 2)
        item["title"] = "窗口二"
        with self.assertRaises(RuntimeError):
            conversation_store.save(item, 1)
        self.assertEqual(conversation_store.load()[0][0]["title"], "窗口一")

    def test_manage_each_conversation_without_switching(self):
        from streamlit.testing.v1 import AppTest

        root = Path(__file__).resolve().parents[1]
        app = AppTest.from_file(str(root / "research_assistant" / "presentation" / "streamlit_app.py")).run(timeout=60)
        first_id = app.session_state["active_conversation_id"]
        next(b for b in app.sidebar.button if b.label == "新建对话").click().run(timeout=60)
        second_id = app.session_state["active_conversation_id"]
        self.assertEqual(len(app.sidebar.text_input), 2)
        next(field for field in app.sidebar.text_input if field.key == f"title_in_{first_id}").set_value("第一篇研究").run(timeout=60)
        # Pressing Enter (on_change) updates and persists the sidebar list title without requiring a Save click.
        self.assertEqual(app.session_state["active_conversation_id"], second_id)
        self.assertEqual(next(b for b in app.sidebar.button if b.key == f"conversation_{first_id}").label, "第一篇研究")
        self.assertEqual(next(c for c in app.session_state["conversations"] if c["id"] == first_id)["title"], "第一篇研究")
        reloaded = AppTest.from_file(str(root / "research_assistant" / "presentation" / "streamlit_app.py")).run(timeout=60)
        self.assertEqual(next(b for b in reloaded.sidebar.button if b.key == f"conversation_{first_id}").label, "第一篇研究")
        # Whitespace-only input keeps the previous valid title instead of clearing it.
        next(field for field in app.sidebar.text_input if field.key == f"title_in_{first_id}").set_value("   ").run(timeout=60)
        self.assertEqual(next(b for b in app.sidebar.button if b.key == f"conversation_{first_id}").label, "第一篇研究")
        self.assertEqual(next(field for field in app.sidebar.text_input if field.key == f"title_in_{first_id}").value, "第一篇研究")
        # Clicking 保存名称 remains supported and idempotent.
        next(b for b in app.sidebar.button if b.key == f"btn_save_{first_id}").click().run(timeout=60)
        self.assertEqual(next(c for c in app.session_state["conversations"] if c["id"] == first_id)["title"], "第一篇研究")
        next(b for b in app.sidebar.button if b.key == f"btn_delete_{first_id}").click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual([c["id"] for c in app.session_state["conversations"]], [second_id])
        self.assertEqual(app.session_state["active_conversation_id"], second_id)

        # Rename active conversation
        app.sidebar.text_input[0].set_value("自定义研读").run(timeout=60)
        next(b for b in app.sidebar.button if b.label == "保存名称").click().run(timeout=60)
        self.assertFalse(app.exception)
        conv = next(c for c in app.session_state["conversations"] if c["id"] == app.session_state["active_conversation_id"])
        self.assertEqual(conv["title"], "自定义研读")

        # Create second conversation and delete it
        next(button for button in app.sidebar.button if button.label == "新建对话").click().run(timeout=60)
        self.assertEqual(len(app.session_state["conversations"]), 2)
        next(b for b in app.sidebar.button if b.label == "删除此对话").click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state["conversations"]), 1)

    def test_preview_survives_hiding_panel_and_manual_selection(self):
        from streamlit.testing.v1 import AppTest

        root = Path(__file__).resolve().parents[1]
        app = AppTest.from_file(str(root / "research_assistant" / "presentation" / "streamlit_app.py")).run(timeout=60)
        self.assertGreaterEqual(len(app.selectbox[0].options), 2)
        app.selectbox[0].select(app.selectbox[0].options[1]).run(timeout=60)
        selected_id = app.session_state["preview_paper_id"]
        app.number_input[0].set_value(2).run(timeout=60)
        app.sidebar.toggle[0].set_value(False).run(timeout=60)
        app.sidebar.toggle[0].set_value(True).run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["preview_paper_id"], selected_id)
        self.assertEqual(app.number_input[0].value, 2)

        app.sidebar.radio[0].set_value("单篇精读").run(timeout=60)
        deep_title = app.selectbox[0].value
        deep_preview = app.session_state["preview_paper_id"]
        alternative = next(option for option in app.selectbox[1].options if option != deep_title)
        app.selectbox[1].select(alternative).run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox[0].value, deep_title)
        self.assertNotEqual(app.session_state["preview_paper_id"], deep_preview)

    def test_invalid_source_page_is_not_silently_replaced(self):
        from streamlit.testing.v1 import AppTest

        root = Path(__file__).resolve().parents[1]
        app = AppTest.from_file(str(root / "research_assistant" / "presentation" / "streamlit_app.py")).run(timeout=60)
        for invalid_page in (0, 9999):
            active = next(item for item in app.session_state["conversations"]
                          if item["id"] == app.session_state["active_conversation_id"])
            active["source_panel"]["page"] = invalid_page
            app.run(timeout=60)
            self.assertFalse(app.exception)
            self.assertTrue(any("无法定位该引用" in warning.value for warning in app.warning))
            self.assertFalse(app.get("image"))
            next(button for button in app.button if button.label == "打开第 1 页").click().run(timeout=60)
            self.assertFalse(app.exception)
            self.assertEqual(app.number_input[0].value, 1)
            self.assertTrue(app.get("image"))

    def test_open_source_highlights_text_and_clears(self):
        from streamlit.testing.v1 import AppTest

        root = Path(__file__).resolve().parents[1]
        app = AppTest.from_file(str(root / "research_assistant" / "presentation" / "streamlit_app.py")).run(timeout=60)
        cid = app.session_state["active_conversation_id"]
        conv = next(c for c in app.session_state["conversations"] if c["id"] == cid)
        conv["histories"]["ask_anything:"] = [{
            "role": "assistant",
            "content": "Answer",
            "sources": [{"paper_id": "2611.09892", "title": "Test Paper", "page": 1, "text": "Agentic Manipulation"}]
        }]
        app.get("segmented_control")[0].set_value("引用").run(timeout=60)
        next(b for b in app.button if b.label == "查看原文页面").click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["source_view"], "论文")
        self.assertTrue(any(b.label == "清除高亮" for b in app.button))
        next(b for b in app.button if b.label == "清除高亮").click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertFalse(any(b.label == "清除高亮" for b in app.button))


class PdfPreviewTests(unittest.TestCase):
    def test_preview_uses_requested_page_and_stays_in_library(self):
        previous = config.PDF_DIR
        with tempfile.TemporaryDirectory(prefix="rpa-preview-test-") as temp:
            try:
                config.PDF_DIR = Path(temp)
                document = fitz.open()
                document.new_page().insert_text((50, 50), "Page one")
                document.new_page().insert_text((50, 50), "Page two")
                document.save(str(config.PDF_DIR / "sample.pdf"))
                document.close()
                path = pdf_preview.resolve_paper_pdf("sample")
                self.assertEqual(pdf_preview.page_count(path), 2)
                first = pdf_preview.render_page(path, 1)
                second = pdf_preview.render_page(path, 2)
                self.assertTrue(first.startswith(b"\x89PNG\r\n\x1a\n"))
                self.assertNotEqual(first, second)
                self.assertIsNone(pdf_preview.resolve_paper_pdf("missing"))
                with self.assertRaises(ValueError):
                    pdf_preview.resolve_paper_pdf("../outside")
                with self.assertRaises(ValueError):
                    pdf_preview.render_page(path, 3)
            finally:
                config.PDF_DIR = previous

    def test_render_page_with_highlight_snippet(self):
        previous = config.PDF_DIR
        with tempfile.TemporaryDirectory(prefix="rpa-highlight-test-") as temp:
            try:
                config.PDF_DIR = Path(temp)
                document = fitz.open()
                page = document.new_page()
                page.insert_text((50, 50), "Neural networks require large amounts of annotated data.")
                document.save(str(config.PDF_DIR / "sample.pdf"))
                document.close()
                path = pdf_preview.resolve_paper_pdf("sample")
                plain = pdf_preview.render_page(path, 1)
                highlighted = pdf_preview.render_page(path, 1, highlight_text="annotated data")
                missing_hl = pdf_preview.render_page(path, 1, highlight_text="completely absent text")
                self.assertTrue(plain.startswith(b"\x89PNG\r\n\x1a\n"))
                self.assertTrue(highlighted.startswith(b"\x89PNG\r\n\x1a\n"))
                self.assertTrue(missing_hl.startswith(b"\x89PNG\r\n\x1a\n"))
                self.assertNotEqual(plain, highlighted)
                self.assertEqual(plain, missing_hl)
            finally:
                config.PDF_DIR = previous


if __name__ == "__main__":
    unittest.main()
