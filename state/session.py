import os

import streamlit as st

from ui.theme import normalize_theme


def init_session_state() -> None:
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "summary" not in st.session_state:
        st.session_state.summary = None
    if "frames" not in st.session_state:
        st.session_state.frames = None
    if "analysis_stats" not in st.session_state:
        st.session_state.analysis_stats = None
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = normalize_theme(os.getenv("UI_THEME"))
    else:
        st.session_state.theme_mode = normalize_theme(st.session_state.theme_mode)
