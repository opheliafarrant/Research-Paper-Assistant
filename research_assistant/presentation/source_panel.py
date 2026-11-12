"""Paper and citation panel for the research workspace."""

from pathlib import Path

import streamlit as st
from research_assistant.domain.citations import chunk_pages, format_page_range
from research_assistant.presentation.conversations import active_conversation


@st.cache_data(show_spinner=False, max_entries=24)
def _page_count(path: str, modified_ns: int, _services) -> int:
    return _services.pdf_page_count(Path(path))


@st.cache_data(show_spinner=False, max_entries=24)
def _page_image(path: str, modified_ns: int, page: int, highlight: str, _services) -> bytes:
    return _services.render_pdf_page(Path(path), page, highlight_text=highlight)


def _panel_state() -> dict:
    conversation = active_conversation(st.session_state)
    return conversation.setdefault("source_panel", {
        "view": "论文", "paper_id": None, "page": 1, "forced_paper_id": None,
        "highlight_text": "",
    })


def _remember_view() -> None:
    _panel_state()["view"] = st.session_state["source_view"]


def _remember_paper() -> None:
    panel = _panel_state()
    panel["paper_id"] = st.session_state["preview_paper_id"]
    panel["page"] = 1
    panel["highlight_text"] = ""


def _remember_page() -> None:
    panel = _panel_state()
    new_page = st.session_state["preview_page"]
    if new_page != panel.get("page"):
        panel["highlight_text"] = ""
    panel["page"] = new_page


def _open_source(paper_id: str, page: int, highlight_text: str = "") -> None:
    panel = _panel_state()
    panel.update(
        view="论文", paper_id=paper_id, page=page, forced_paper_id=paper_id,
        highlight_text=highlight_text,
    )


def _open_first_page() -> None:
    panel = _panel_state()
    panel["page"] = 1
    panel["highlight_text"] = ""


def _clear_highlight() -> None:
    _panel_state()["highlight_text"] = ""


def render_source_panel(papers: list[dict], sources: list[dict], services,
                        forced_paper_id: str | None = None) -> None:
    panel = _panel_state()
    by_id = {paper["paper_id"]: paper for paper in papers}
    if forced_paper_id in by_id and panel["forced_paper_id"] != forced_paper_id:
        panel.update(paper_id=forced_paper_id, page=1)
    panel["forced_paper_id"] = forced_paper_id
    st.session_state["source_view"] = panel["view"]
    st.markdown("#### 资料与来源")
    view = st.segmented_control(
        "资料视图", ["论文", "引用"], required=True, key="source_view",
        on_change=_remember_view, label_visibility="collapsed", width="stretch",
    )
    if view == "引用":
        if not sources:
            st.caption("提问或速读后，最近一次使用的片段会显示在这里。")
            return
        st.caption(f"最近一次回答或速读 · {len(sources)} 个参考片段")
        for index, source in enumerate(sources, 1):
            badge = source.get("_confidence_badge", "")
            badge_label = f" ({badge})" if badge else ""
            page_label = format_page_range(source)
            start_page, end_page = chunk_pages(source)
            with st.expander(f"[{index}] {source['title']} · 第 {page_label} 页{badge_label}"):
                if badge:
                    st.caption(f"🎯 检索匹配置信度：`{badge}`")
                st.write(source["text"])
                st.button(
                    "查看原文页面", key=f"open_source_{index}",
                    on_click=_open_source,
                    args=(source["paper_id"], start_page, source.get("text", "")),
                    use_container_width=True,
                )
                if end_page > start_page:
                    st.button(
                        f"查看续页（第 {end_page} 页）", key=f"open_source_cont_{index}",
                        on_click=_open_source,
                        args=(source["paper_id"], end_page, source.get("text", "")),
                        use_container_width=True,
                    )
        return

    if not papers:
        st.caption("还没有论文。请先在左侧获取新论文。")
        return
    ids = list(by_id)
    if panel["paper_id"] not in by_id:
        panel.update(paper_id=ids[0], page=1)
    st.session_state["preview_paper_id"] = panel["paper_id"]

    paper_id = st.selectbox(
        "查看论文", ids, format_func=lambda item: by_id[item]["title"],
        key="preview_paper_id", on_change=_remember_paper,
    )
    st.caption(f"论文编号 · {paper_id}")
    try:
        pdf_path = services.resolve_paper_pdf(paper_id)
        if pdf_path is None:
            st.info("本地没有这篇论文的 PDF，仍可查看已索引的引用片段。")
            return
        modified_ns = pdf_path.stat().st_mtime_ns
        count = _page_count(str(pdf_path), modified_ns, services)
        if not count:
            st.warning("该 PDF 没有可预览的页面。")
            return
        if not 1 <= int(panel["page"]) <= count:
            st.warning(f"引用标注第 {panel['page']} 页，但本地 PDF 共 {count} 页，无法定位该引用。")
            st.button("打开第 1 页", on_click=_open_first_page, use_container_width=True)
            return
        st.session_state["preview_page"] = panel["page"]
        st.number_input("预览页码", min_value=1, max_value=count, step=1,
                        key="preview_page", on_change=_remember_page)
        highlight = panel.get("highlight_text", "") if panel.get("page") == st.session_state["preview_page"] else ""
        image = _page_image(str(pdf_path), modified_ns, st.session_state["preview_page"], highlight, services)
        if highlight:
            col_hl, col_clear = st.columns([3, 1])
            with col_hl:
                st.caption("正在尝试在页面中高亮匹配的片段；请核对原文。")
            with col_clear:
                st.button("清除高亮", key="clear_highlight", on_click=_clear_highlight, use_container_width=True)
        else:
            st.caption("可使用页面右上角的放大按钮查看细节。")
        st.image(image, width="stretch", caption=f"原文第 {st.session_state['preview_page']} / {count} 页")
    except (OSError, RuntimeError, ValueError) as exc:
        st.warning(f"无法预览本地 PDF：{exc}")
