"""Exercise the real local model and indexed paper through the Streamlit UI."""

import re
import sys
import tempfile
from pathlib import Path

from streamlit.testing.v1 import AppTest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from research_assistant.infrastructure import conversation_store

temporary_store = tempfile.TemporaryDirectory(prefix="rpa-live-ui-")
conversation_store.DB_PATH = Path(temporary_store.name) / "conversations.sqlite3"
app = AppTest.from_file(str(root / "research_assistant" / "presentation" / "streamlit_app.py")).run(timeout=90)
assert not app.exception and app.title[0].value == "论文问答"
assert any(item.label == "📖 使用说明" for item in app.expander)
assert any(button.label == "获取新论文" for button in app.sidebar.button)

next(button for button in app.button if button.label == "梳理研究方法 ↗").click().run(timeout=180)
assert not app.exception and len(app.chat_message) == 2
assert app.chat_message[-1].expander and app.chat_message[-1].expander[0].label.startswith("参考来源")
print("QUICK_QUESTION_OK", flush=True)

first_id = app.session_state["active_conversation_id"]
conversation = next(item for item in app.session_state["conversations"] if item["id"] == first_id)
first_source = conversation["histories"]["ask_anything:"][-1]["sources"][0]
assert "_confidence" in first_source and "_confidence_badge" in first_source
print(f"CONFIDENCE_BADGE_OK: {first_source['_confidence_badge']}", flush=True)
app.get("segmented_control")[0].set_value("引用").run(timeout=90)
assert any(button.label == "查看原文页面" for button in app.button)
next(button for button in app.button if button.label == "查看原文页面").click().run(timeout=90)
assert not app.exception and app.session_state["source_view"] == "论文"
assert app.session_state["preview_paper_id"] == first_source["paper_id"]
assert app.session_state["preview_page"] == first_source["page"]
assert app.get("image")
print("SOURCE_TO_PDF_PAGE_OK", flush=True)

next(button for button in app.sidebar.button if button.label == "新建对话").click().run(timeout=90)
assert not app.exception and len(app.chat_message) == 0
next(button for button in app.sidebar.button if button.key == f"conversation_{first_id}").click().run(timeout=90)
assert not app.exception and len(app.chat_message) == 2
print("CONVERSATION_SWITCH_OK", flush=True)

app.sidebar.radio[0].set_value("单篇精读").run(timeout=90)
app.selectbox[0].select("On Agentic Manipulation Policies").run(timeout=90)
assert not app.exception

for question, language in [("这篇论文的主要方法是什么？", "zh"),
                           ("What method does the paper propose?", "en")]:
    app.chat_input[0].set_value(question).run(timeout=180)
    assert not app.exception, str(app.exception)
    message = app.chat_message[-1]
    answer = message.markdown[0].value
    assert len(answer) > 15
    assert re.search(r"[\u4e00-\u9fff]" if language == "zh" else r"[A-Za-z]", answer)
    assert message.expander and message.expander[0].label.startswith("参考来源")
    print(f"QUESTION_{language.upper()}_OK {message.expander[0].label}", flush=True)

next(button for button in app.button if button.label == "生成摘要").click().run(timeout=240)
assert not app.exception
summary = app.session_state["summary_2611.09892"]
assert summary and re.search(r"[\u4e00-\u9fff]", summary)
assert app.chat_message[-1].expander and app.chat_message[-1].expander[0].label.startswith("速读参考片段")
assert any(button.label == "查看原文" for button in app.chat_message[-1].button)
print("SUMMARY_SOURCES_OK", flush=True)
print("CHINESE_SUMMARY_OK UI_EXCEPTIONS=0", flush=True)

next(button for button in app.button if button.label == "核心方法").click().run(timeout=240)
assert not app.exception
methods = app.session_state["summary_2611.09892"]
assert methods and re.search(r"[\u4e00-\u9fff]", methods)
print("ASPECT_METHODS_OK", flush=True)

next(button for button in app.button if button.label == "局限与未来").click().run(timeout=240)
assert not app.exception
limitations = app.session_state["summary_2611.09892"]
assert limitations and len(limitations) > 100 and re.search(r"[\u4e00-\u9fff]", limitations)
assert any(kw in limitations for kw in ["延迟", "速度", "等待", "静态", "滑动", "动态"])
assert any(kw in limitations for kw in ["控制器", "反馈", "重叠", "监督", "复用", "调用"])
print("ASPECT_LIMITATIONS_OK", flush=True)

deep_id = app.session_state["active_conversation_id"]
next(button for button in app.sidebar.button if button.label == "新建对话").click().run(timeout=90)
assert app.sidebar.radio[0].value == "论文问答"
next(button for button in app.sidebar.button if button.key == f"conversation_{deep_id}").click().run(timeout=90)
assert not app.exception and app.sidebar.radio[0].value == "单篇精读"
assert app.selectbox[0].value == "On Agentic Manipulation Policies"
assert len(app.chat_message) == 7
print("READING_SCOPE_RESTORED", flush=True)
temporary_store.cleanup()
