from dataclasses import dataclass

import streamlit as st

from ui.theme import THEME_OPTIONS, normalize_theme


@dataclass
class SidebarConfig:
    theme_mode: str
    frame_interval: int
    max_frames: int
    summary_style: str
    search_query: str


def render_sidebar() -> SidebarConfig:
    st.sidebar.divider()
    st.sidebar.header("Appearance")
    current_theme = normalize_theme(st.session_state.theme_mode)
    theme_mode = st.sidebar.radio(
        "Theme",
        THEME_OPTIONS,
        index=THEME_OPTIONS.index(current_theme),
        help="Choose between Light Mode and Dark Mode.",
    )
    st.sidebar.divider()

    st.sidebar.header("Analysis Settings")
    frame_interval = st.sidebar.slider(
        "Frame sampling interval (seconds)",
        min_value=1,
        max_value=10,
        value=2,
        help="Extract one frame at the selected interval.",
    )

    max_frames = st.sidebar.slider(
        "Maximum frames to analyze",
        min_value=5,
        max_value=50,
        value=20,
        help="Cap the number of frames to keep processing time predictable.",
    )

    summary_style = st.sidebar.selectbox(
        "Summary style",
        ["Detailed", "Concise", "Bullet Points"],
        help="Select the output structure used for the written summary.",
    )

    st.sidebar.divider()
    st.sidebar.subheader("Archive Search")
    search_query = st.sidebar.text_input(
        "Search saved summaries",
        placeholder="Example: pedestrian crossing",
        help="Run semantic search across previously saved summaries.",
        key="sidebar_search_query",
    )
    st.sidebar.caption("Other pages: open **Semantic search** below.")

    return SidebarConfig(
        theme_mode=theme_mode,
        frame_interval=frame_interval,
        max_frames=max_frames,
        summary_style=summary_style,
        search_query=search_query,
    )
