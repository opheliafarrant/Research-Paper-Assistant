"""Presentation-only styling for the research workspace."""

import streamlit as st


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] { background: #fff; color: #232932; }
        [data-testid="stSidebar"] { background: #f5f6f8; border-right: 1px solid #e7e9ed; }
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding: 1.35rem .85rem; }
        [data-testid="stMainBlockContainer"] { max-width: 1600px; padding: 2.2rem 2rem 5rem; container-type: inline-size; }
        [data-testid="stMain"] h1 { font-family: -apple-system, 'Segoe UI', 'Microsoft YaHei', sans-serif;
            font-size: 1.85rem; font-weight: 650; letter-spacing: -.03em; color: #252b34; }
        [data-testid="stSidebar"] h1 { font-size: 1.13rem; letter-spacing: -.02em; }
        [data-testid="stCaptionContainer"] { color: #737d8b; }
        .workspace-kicker { color: #7d8796; font-size: .7rem; font-weight: 700;
            letter-spacing: .12em; margin-bottom: .25rem; }
        .section-label { color: #7a8492; font-size: .74rem; font-weight: 700;
            letter-spacing: .04em; margin: 1.3rem 0 .45rem; }
        .section-note { color: #838d9b; font-size: .82rem; margin: -.15rem 0 .8rem; }
        .library-card { background: #fff; border: 1px solid #e6e9ee; border-radius: 12px;
            padding: .7rem .85rem; margin: .45rem 0 .65rem; }
        .library-label { color: #84909e; font-size: .72rem; margin-bottom: .2rem; }
        .library-value { color: #333d4a; font-size: 1rem; font-weight: 650; }
        .library-note { color: #8a94a1; font-size: .7rem; margin-top: .15rem; }
        div.stButton > button { border: 1px solid #e4e7eb; border-radius: 9px;
            background: #fff; color: #3d4856; min-height: 2.55rem; }
        div.stButton > button:hover { border-color: #9aa9bb; color: #344c68; background: #f8fafc; }
        div.stButton > button[kind="primary"] { background: #53647f; border-color: #53647f; color: white; }
        div.stButton > button[kind="primary"]:hover { background: #43546e; color: white; }
        [class*="st-key-conversation_"] button { border: 0 !important;
            justify-content: flex-start; text-align: left; min-height: 2.35rem !important;
            padding: .4rem .65rem; border-radius: 9px !important; }
        [class*="st-key-conversation_"] button[kind="secondary"] { background: transparent; color: #4a5568; }
        [class*="st-key-conversation_"] button[kind="secondary"]:hover { background: #eaedf2; color: #1a202c; }
        [class*="st-key-conversation_"] button[kind="primary"] { background: #e2e8f0; color: #1e293b; font-weight: 600; }
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has([class*="st-key-conversation_"]) { gap: .2rem; align-items: center; margin-bottom: .2rem; }
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has([class*="st-key-conversation_"]) [data-testid="stPopover"] button {
            border: 1px solid transparent !important; background: transparent !important;
            color: #8a96a6 !important; min-height: 2.35rem !important; padding: .25rem .45rem !important;
            border-radius: 8px !important; box-shadow: none !important; }
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has([class*="st-key-conversation_"]) [data-testid="stPopover"] button:hover {
            background: #e9edf3 !important; color: #2d3748 !important; border-color: #cbd5e1 !important; }
        [data-testid="stExpander"] { border: 1px solid #e8ebef; border-radius: 10px; background: #fff; }
        [data-testid="stChatMessage"] { border: 0; border-radius: 12px; background: #f8f9fb; padding: 1rem; }
        [data-testid="stColumn"]:has([data-testid="stChatInput"]) { container-type: inline-size; }
        [data-testid="stChatInput"] { position: fixed; bottom: 1.1rem; z-index: 10; width: 100cqw;
            border-radius: 13px; border: 1px solid #dfe4eb; box-shadow: 0 5px 24px #1b283b0c; }
        [data-testid="stSelectbox"] > div { border-radius: 9px; }
        hr { border-color: #e9ebef; }
        /* 窄屏：右侧来源面板改为主栏下方整行排列，避免被压缩到 150-200px */
        @media (max-width: 1150px) {
            [data-testid="stHorizontalBlock"]:has(.st-key-source_panel_card) { row-gap: 1rem !important; }
            [data-testid="stHorizontalBlock"]:has(.st-key-source_panel_card) > [data-testid="stColumn"] {
                flex: 1 1 100% !important; width: 100% !important; max-width: 100% !important; }
            .st-key-source_panel_card [data-testid="stImage"] img {
                width: auto !important; max-width: min(460px, 100%) !important;
                display: block; margin: 0 auto; }
        }
        @media (max-width: 768px) {
            [data-testid="stMainBlockContainer"] { padding: 1.6rem 1rem 4rem; }
            [data-testid="stMain"] h1 { font-size: 1.55rem; }
            [data-testid="stChatInput"] { position: static; width: auto; box-shadow: none; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
