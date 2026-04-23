"""Shared Streamlit login for the main app and multipage routes."""

from __future__ import annotations

import os

import streamlit as st


def ensure_auth_state() -> None:
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = None


def get_credentials() -> dict[str, str]:
    """Username -> password. From env (single user) or Streamlit secrets."""
    try:
        if "passwords" in st.secrets:
            return st.secrets["passwords"]
    except Exception:
        pass
    user = os.getenv("LOGIN_USERNAME")
    pwd = os.getenv("LOGIN_PASSWORD")
    if user and pwd:
        return {user: pwd}
    return {}


def require_login() -> None:
    """If not logged in, render login form and stop."""
    ensure_auth_state()
    if st.session_state.logged_in:
        return

    credentials = get_credentials()
    if not credentials:
        st.warning(
            "Set LOGIN_USERNAME and LOGIN_PASSWORD in the environment, or add a "
            "'passwords' dict in Streamlit secrets."
        )
        st.stop()

    with st.form("login"):
        st.subheader("Login")
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted and u and p and credentials.get(u) == p:
            st.session_state.logged_in = True
            st.session_state.username = u
            st.rerun()
        elif submitted:
            st.error("Invalid username or password.")
    st.stop()


def render_user_sidebar() -> None:
    """Caption + logout when logged in."""
    if st.session_state.get("logged_in"):
        st.sidebar.caption(f"Logged in as **{st.session_state.username}**")
        if st.sidebar.button("Log out", key="auth_logout_button"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.rerun()
        st.sidebar.divider()
