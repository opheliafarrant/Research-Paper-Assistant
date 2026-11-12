"""
Streamlit front-end for the Research Paper Assistant
Two modes toggled from the sidebar:
    Ask Anything   –  RAG Q&A across all indexed papers
    Deep Dive      –  pick one paper, ask questions or get a summary
"""

import streamlit as st
from research_assistant import config
from research_assistant.bootstrap import build_application
from research_assistant.domain.citations import format_page_range
from research_assistant.presentation.components import introduction, library_card, quick_questions, section_label, show_guide
from research_assistant.presentation import conversation_persistence as persistence
from research_assistant.presentation.conversations import (
    active_conversation,
    clear_summary_for,
    conversation_stats,
    delete_conversation,
    history_for,
    mode_icon,
    name_from_question,
    new_conversation,
    rename_conversation,
    select_conversation,
    set_summary,
    summary_for,
)
from research_assistant.presentation.source_panel import _open_source, render_source_panel
from research_assistant.presentation.styles import apply_styles

_services = build_application()


def _sync_conversation_summaries() -> None:
    conversation = active_conversation(st.session_state)
    for key in list(st.session_state.keys()):
        if key.startswith("summary_") or key.startswith("aspect_"):
            st.session_state.pop(key, None)
    for pid, data in conversation.setdefault("summaries", {}).items():
        st.session_state[f"summary_{pid}"] = data["text"]
        st.session_state[f"aspect_{pid}"] = data["aspect"]


def _safe_save(conversation: dict) -> bool:
    """Persist conversation state and surface cross-tab version conflicts gracefully."""
    try:
        persistence.save(st.session_state, conversation)
        return True
    except RuntimeError as exc:
        st.warning(f"{exc} 请刷新页面以同步最新会话状态。")
        return False


def _create_conversation() -> None:
    conversation = new_conversation(st.session_state)
    _safe_save(conversation)
    persistence.select(conversation["id"])
    st.session_state["reading_mode"] = "论文问答"
    st.session_state.pop("deep_paper_title", None)
    _sync_conversation_summaries()


def _activate_conversation(conversation_id: str) -> None:
    select_conversation(st.session_state, conversation_id)
    conversation = active_conversation(st.session_state)
    st.session_state["reading_mode"] = conversation["mode"]
    if conversation["paper_title"]:
        st.session_state["deep_paper_title"] = conversation["paper_title"]
    else:
        st.session_state.pop("deep_paper_title", None)
    _sync_conversation_summaries()
    persistence.select(conversation_id)


def _delete_conversation(conversation_id: str) -> None:
    try:
        persistence.delete(st.session_state, [conversation_id])
    except RuntimeError as exc:
        st.warning(f"{exc} 请刷新页面后重试。")
        return
    delete_conversation(st.session_state, conversation_id)
    _activate_conversation(st.session_state["active_conversation_id"])


def _rename_conversation(conversation_id: str) -> None:
    target = next(
        (item for item in st.session_state.get("conversations", []) if item["id"] == conversation_id),
        None,
    )
    if target is None:
        return
    input_key = f"title_in_{conversation_id}"
    if rename_conversation(target, str(st.session_state.get(input_key, ""))):
        _safe_save(target)
    st.session_state[input_key] = target["title"]
    st.session_state.setdefault("conversation_title_versions", {})[conversation_id] = target["title"]


def _remember_mode() -> None:
    active_conversation(st.session_state)["mode"] = st.session_state["reading_mode"]
    _safe_save(active_conversation(st.session_state))


def _remember_paper() -> None:
    active_conversation(st.session_state)["paper_title"] = st.session_state["deep_paper_title"]
    _safe_save(active_conversation(st.session_state))


def _clear_conversations() -> None:
    try:
        persistence.delete(st.session_state, [item["id"] for item in st.session_state["conversations"]])
    except RuntimeError as exc:
        st.warning(f"{exc} 请刷新页面后重试。")
        return
    st.session_state["conversations"] = []
    _create_conversation()


def _all_papers() -> list[dict]:
    return _services.list_papers()


def _ingest_button() -> None:
    notice = st.session_state.pop("ingest_notice", None)
    if notice:
        level, msg = notice
        getattr(st, level, st.info)(msg)
    if st.button("获取新论文", type="primary", icon=":material/add:", use_container_width=True):
        with st.spinner("正在获取论文并建立索引，请稍候……"):
            try:
                summary = _services.ingest()
            except Exception as exc:
                st.error(f"获取论文失败：{exc}")
                return
        if isinstance(summary, dict):
            fetched = int(summary.get("fetched", 0))
            ingested = summary.get("ingested") or []
            failed = summary.get("failed") or []
            if fetched == 0:
                st.info("未发现新论文，本地论文库已是最新。")
                return
            if failed and not ingested:
                st.error(f"本次获取到 {fetched} 篇论文，但全部导入失败：{', '.join(failed)}")
                return
            if failed:
                st.session_state["ingest_notice"] = (
                    "warning",
                    f"成功导入 {len(ingested)} 篇论文，另有 {len(failed)} 篇失败：{', '.join(failed)}",
                )
            else:
                st.session_state["ingest_notice"] = ("success", f"成功导入 {len(ingested)} 篇新论文。")
        else:
            st.session_state["ingest_notice"] = ("success", "论文获取流程已结束，请查看已收录数量。")
        st.rerun()


def _open_message_source(paper_id: str, page: int, text: str) -> None:
    st.session_state["show_source_panel"] = True
    _open_source(paper_id, page, text)


def _render_sources(sources: list[dict], *, key_prefix: str = "", is_summary: bool = False) -> None:
    if not sources:
        return
    label = "速读参考片段" if is_summary else "参考来源"
    with st.expander(f"{label}（{len(sources)} 个片段）"):
        if is_summary:
            st.caption("以下是生成时使用的论文片段，不代表逐句引用；请结合原文核对。")
        for i, s in enumerate(sources, 1):
            badge = s.get("_confidence_badge", "")
            badge_str = f" · `{badge}`" if badge else ""
            st.markdown(
                f"**[{i}]** `{s['paper_id']}` — *{s['title']}*  "
                f"· 第 {format_page_range(s)} 页{badge_str}"
            )
            st.caption(s["text"][:300] + ("…" if len(s["text"]) > 300 else ""))
            if key_prefix:
                st.button("查看原文", key=f"{key_prefix}_source_{i}", on_click=_open_message_source,
                          args=(s["paper_id"], int(s["page"]), s["text"]))
            if i < len(sources):
                st.divider()


def _record_summary(conversation: dict, paper_id: str, text: str, aspect: str, sources: list[dict]) -> None:
    set_summary(conversation, paper_id, text, aspect)
    conversation["summaries"][paper_id]["in_history"] = True
    history_for(conversation, "deep_dive", paper_id).append({
        "role": "assistant", "content": f"**{aspect}**\n\n{text}", "kind": "summary", "sources": sources,
    })
    name_from_question(conversation, aspect)


def _render_chat(conversation: dict, mode: str, paper_id: str = "", suggested_query: str = "") -> list[dict]:
    history = history_for(conversation, mode, paper_id)
    for message_index, msg in enumerate(history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                _render_sources(msg["sources"], key_prefix=f"message_{message_index}",
                                is_summary=msg.get("kind") == "summary")

    query = st.chat_input("输入关于论文的问题，支持中文和英文……") or suggested_query
    if not query:
        return next((msg.get("sources", []) for msg in reversed(history) if msg["role"] == "assistant"), [])

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("正在查找资料并生成回答……"):
            try:
                result = _services.answer(query, paper_id=paper_id if mode == "deep_dive" else None)
            except Exception as exc:
                st.error(f"生成回答失败：{exc}")
                return next((msg.get("sources", []) for msg in reversed(history) if msg["role"] == "assistant"), [])

        st.markdown(result["answer"])
        _render_sources(result["sources"])

    history.append({"role": "user", "content": query})
    name_from_question(conversation, query)
    history.append({
        "role":    "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
    _safe_save(conversation)
    st.rerun()


def main() -> None:
    st.set_page_config(page_title = "论文阅读助手", page_icon = config.APP_ICON, layout = "wide")
    apply_styles()
    first_load = not st.session_state.get("conversations_loaded")
    persistence.initialize(st.session_state)
    papers = _all_papers()
    conversation = active_conversation(st.session_state)
    _safe_save(conversation)
    if first_load:
        persistence.select(conversation["id"])
        st.session_state["reading_mode"] = conversation["mode"]
        if conversation.get("paper_title"):
            st.session_state["deep_paper_title"] = conversation["paper_title"]
        _sync_conversation_summaries()
    with st.sidebar:
        st.title(config.APP_ICON + "  论文阅读助手")
        st.caption("一个安静的研究空间")
        st.button("新建对话", icon=":material/add:", on_click=_create_conversation, use_container_width=True)

        section_label("最近对话")
        known_titles = st.session_state.setdefault("conversation_title_versions", {})
        for item in reversed(st.session_state["conversations"]):
            item_id = item["id"]
            input_key = f"title_in_{item_id}"
            if item_id in known_titles and known_titles[item_id] != item["title"]:
                st.session_state[input_key] = item["title"]
            elif input_key not in st.session_state:
                st.session_state[input_key] = item["title"]
            known_titles[item_id] = item["title"]
            item_mode = item.get("mode", "论文问答")
            icon = mode_icon(item_mode)
            title_col, menu_col = st.columns([5, 1], vertical_alignment="center")
            with title_col:
                st.button(item["title"], icon=icon, key=f"conversation_{item_id}", on_click=_activate_conversation,
                          args=(item_id,), type="primary" if item_id == conversation["id"] else "secondary",
                          use_container_width=True)
            with menu_col:
                with st.popover("⋯", help=f"管理对话：{item['title']}", use_container_width=True):
                    stats = conversation_stats(item)
                    scope_desc = "单篇精读" if item_mode == "单篇精读" else "全库问答"
                    st.caption(f"**{scope_desc}** · {stats['messages']} 条消息 · {stats['summaries']} 个速读卡片")
                    st.text_input(
                        "会话名称",
                        key=input_key,
                        on_change=_rename_conversation,
                        args=(item_id,),
                    )
                    col_save, col_del = st.columns(2)
                    with col_save:
                        st.button(
                            "保存名称",
                            key=f"btn_save_{item_id}",
                            on_click=_rename_conversation,
                            args=(item_id,),
                            type="primary",
                            use_container_width=True,
                        )
                    with col_del:
                        st.button("删除此对话", key=f"btn_delete_{item_id}",
                                  on_click=_delete_conversation, args=(item_id,),
                                  use_container_width=True, disabled=len(st.session_state["conversations"]) <= 1,
                                  help="删除此对话" if len(st.session_state["conversations"]) > 1 else "仅剩一个对话时不可删除")

        section_label("阅读范围")
        mode = st.radio("阅读模式", ["论文问答", "单篇精读"], key="reading_mode",
                        on_change=_remember_mode, label_visibility="collapsed")
        if mode == "单篇精读" and conversation.get("paper_title"):
            st.caption(f"当前精读：{conversation['paper_title'][:22]}…")

        section_label("资料库")
        library_card(len(papers))
        _ingest_button()

        st.divider()
        st.caption("对话保存在本地")
        if "show_source_panel" not in st.session_state:
            st.session_state["show_source_panel"] = True
        show_panel = st.toggle("显示资料与来源", key="show_source_panel")
        with st.popover("管理全部对话", use_container_width=True):
            st.caption("清空后将重置为初始空白会话")
            st.button("清空对话记录", on_click=_clear_conversations, use_container_width=True)

    if show_panel:
        main_column, source_column = st.columns([2.15, 1], gap="medium")
    else:
        main_column, source_column = st.container(), None

    sources = []
    selected_paper_id = None
    with main_column:
        if mode == "论文问答":
            introduction("论文问答", "从资料中寻找证据，梳理方法、结果与研究思路。")
            show_guide()
            if not papers:
                st.info("还没有收录论文，请点击左侧“获取新论文”。")
            else:
                suggestion = ""
                if not history_for(conversation, "ask_anything"):
                    suggestion = quick_questions() or ""
                sources = _render_chat(conversation, mode="ask_anything", suggested_query=suggestion)
        else:
            introduction("单篇精读", "聚焦一篇论文，边读原文边提问。")
            show_guide()
            if not papers:
                st.info("还没有收录论文，请点击左侧“获取新论文”。")
            else:
                section_label("选择要阅读的论文")
                options = {p["title"]: p for p in papers}
                if st.session_state.get("deep_paper_title") not in options:
                    st.session_state.pop("deep_paper_title", None)
                selected = st.selectbox("选择论文", list(options.keys()), key="deep_paper_title",
                                        on_change=_remember_paper, label_visibility="collapsed")
                conversation["paper_title"] = selected
                paper = options[selected]
                selected_paper_id = paper["paper_id"]
                st.caption(f"论文编号 · {paper['paper_id']}")
                section_label("结构化速读")
                col_m, col_e, col_l = st.columns(3)
                with col_m:
                    btn_methods = st.button("核心方法", key=f"method_{paper['paper_id']}", use_container_width=True)
                with col_e:
                    btn_exp = st.button("实验对比", key=f"exp_{paper['paper_id']}", use_container_width=True)
                with col_l:
                    btn_lim = st.button("局限与未来", key=f"lim_{paper['paper_id']}", use_container_width=True)

                col_s, col_c, _ = st.columns([1, 1, 2])
                with col_s:
                    summarise_btn = st.button("生成摘要", key=f"sum_{paper['paper_id']}", use_container_width=True)
                with col_c:
                    clear_summary = st.button("清除摘要", key=f"clear_sum_{paper['paper_id']}", use_container_width=True)

                summary_key = f"summary_{paper['paper_id']}"
                aspect_key = f"aspect_{paper['paper_id']}"
                if clear_summary:
                    current_summary = summary_for(conversation, paper["paper_id"])
                    if current_summary and current_summary.get("in_history"):
                        history = history_for(conversation, "deep_dive", paper["paper_id"])
                        for index in range(len(history) - 1, -1, -1):
                            if history[index].get("kind") == "summary":
                                history.pop(index)
                                break
                    clear_summary_for(conversation, paper["paper_id"])
                    st.session_state.pop(summary_key, None)
                    st.session_state.pop(aspect_key, None)
                aspect_actions = (
                    (summarise_btn, "summary", "综合结构化摘要", "正在生成综合结构化摘要……"),
                    (btn_methods, "methods", "核心方法与算法架构", "正在提炼核心方法与算法架构……"),
                    (btn_exp, "experiments", "实验对比与评估结果", "正在提炼实验对比与评估结果……"),
                    (btn_lim, "limitations", "研究局限性与未来方向", "正在提炼局限性与未来研究方向……"),
                )
                triggered_aspect = False
                for clicked, aspect_id, aspect_title, spinner_msg in aspect_actions:
                    if not clicked:
                        continue
                    with st.spinner(spinner_msg):
                        try:
                            res = _services.summarise_with_sources(paper["paper_id"], paper["title"], aspect=aspect_id)
                        except Exception as exc:
                            st.error(f"生成{aspect_title}失败：{exc}")
                            break
                        _record_summary(conversation, paper["paper_id"], res["text"], aspect_title, res["sources"])
                        st.session_state[summary_key] = res["text"]
                        st.session_state[aspect_key] = aspect_title
                        triggered_aspect = True
                    break

                active_summary = summary_for(conversation, paper["paper_id"]) or (
                    {"text": st.session_state[summary_key], "aspect": st.session_state.get(aspect_key, "论文提炼")}
                    if summary_key in st.session_state else None
                )
                if active_summary and not active_summary.get("in_history"):
                    with st.expander(f"📑 {active_summary['aspect']}", expanded=True):
                        st.markdown(active_summary["text"])
                if triggered_aspect:
                    _safe_save(conversation)
                    st.rerun()
                sources = _render_chat(conversation, mode="deep_dive", paper_id=paper["paper_id"])

    if source_column is not None:
        with source_column:
            with st.container(border=True, key="source_panel_card"):
                render_source_panel(papers, sources, _services, forced_paper_id=selected_paper_id)
    _safe_save(conversation)


if __name__ == "__main__":
    main()


