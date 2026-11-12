"""Reusable Streamlit components for navigation and first-use guidance."""

import streamlit as st
from html import escape


def section_label(text: str) -> None:
    st.markdown(f'<div class="section-label">{escape(text)}</div>', unsafe_allow_html=True)


def library_card(count: int) -> None:
    note = "已建立本地检索索引" if count else "获取论文后即可开始阅读"
    st.markdown(
        '<div class="library-card"><div class="library-label">我的论文库</div>'
        f'<div class="library-value">{count} 篇论文</div>'
        f'<div class="library-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def introduction(title: str, description: str) -> None:
    st.markdown('<div class="workspace-kicker">你的研究工作台 · 本地知识库</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(description)


def show_guide() -> None:
    with st.expander("📖 使用说明", expanded=False):
        st.markdown(
            "1. **获取论文**：点击左侧“获取新论文”；已有论文可直接提问。\n"
            "2. **论文问答**：输入中文或英文问题，回答会尽量跟随提问语言。\n"
            "3. **单篇精读**：选择论文，生成中文摘要或针对它提问。\n"
            "4. **核对来源**：展开回答或速读中的参考片段，点击“查看原文”；也可在右侧“引用”中查看最近使用的片段。\n"
            "5. **管理对话**：左侧可新建与切换对话，对话自动保存在本地。"
        )
        st.caption("获取论文需要联网；本地模型生成答案可能需要一些时间。")


def quick_questions() -> str | None:
    section_label("不知从哪里开始？")
    st.markdown('<div class="section-note">选一个方向，或直接在下方输入你的问题。</div>', unsafe_allow_html=True)
    choices = (
        ("梳理研究方法 ↗", "这些论文主要使用了哪些研究方法？请注明论文编号。"),
        ("比较不同思路 ↗", "请比较已收录论文的研究思路，并标明论文编号。"),
        ("寻找研究局限 ↗", "已收录论文提到了哪些研究局限或未来方向？请注明论文编号。"),
    )
    selected = None
    for column, (label, question) in zip(st.columns(3), choices):
        with column:
            if st.button(label, key=f"suggest_{label}", use_container_width=True):
                selected = question
    return selected
