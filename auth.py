"""Shared Streamlit login for the main app and multipage routes."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time

import streamlit as st

from ui.theme import apply_theme, normalize_theme

AUTH_QUERY_PARAM = "auth"
AUTH_TOKEN_TTL_SECONDS = 60 * 60 * 24 * 7  # 7 days


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


def _get_auth_secret() -> str:
    return (
        os.getenv("AUTH_SECRET")
        or os.getenv("STREAMLIT_AUTH_SECRET")
        or os.getenv("LOGIN_PASSWORD")
        or "streamlit-auth-fallback-secret"
    )


def _build_auth_token(username: str) -> str:
    payload = {"u": username, "exp": int(time.time()) + AUTH_TOKEN_TTL_SECONDS}
    payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload_b64 = base64.urlsafe_b64encode(payload_json).decode("utf-8").rstrip("=")
    signature = hmac.new(_get_auth_secret().encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload_b64}.{signature}"


def _restore_user_from_token() -> None:
    token = st.query_params.get(AUTH_QUERY_PARAM)
    if not token or st.session_state.get("logged_in"):
        return
    try:
        payload_b64, provided_signature = token.split(".", 1)
        expected_signature = hmac.new(_get_auth_secret().encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(provided_signature, expected_signature):
            st.query_params.pop(AUTH_QUERY_PARAM, None)
            return

        padded_payload = payload_b64 + "=" * (-len(payload_b64) % 4)
        payload_json = base64.urlsafe_b64decode(padded_payload.encode("utf-8")).decode("utf-8")
        payload = json.loads(payload_json)
        username = payload.get("u")
        expires_at = int(payload.get("exp", 0))
        if not username or expires_at <= int(time.time()):
            st.query_params.pop(AUTH_QUERY_PARAM, None)
            return

        st.session_state.logged_in = True
        st.session_state.username = username
    except Exception:
        st.query_params.pop(AUTH_QUERY_PARAM, None)


def require_login() -> None:
    """If not logged in, render login form and stop."""
    ensure_auth_state()
    _restore_user_from_token()
    # Ensure theme exists even before login so auth UI matches app styling.
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = normalize_theme(os.getenv("UI_THEME"))
    else:
        st.session_state.theme_mode = normalize_theme(st.session_state.theme_mode)
    apply_theme(st.session_state.theme_mode)

    if st.session_state.logged_in:
        # Multipage navigation can drop query params on some routes; restore token
        # whenever an authenticated session exists so hard refresh remains logged in.
        if not st.query_params.get(AUTH_QUERY_PARAM) and st.session_state.username:
            st.query_params[AUTH_QUERY_PARAM] = _build_auth_token(st.session_state.username)
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
            st.query_params[AUTH_QUERY_PARAM] = _build_auth_token(u)
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
            st.query_params.pop(AUTH_QUERY_PARAM, None)
            st.rerun()
        st.sidebar.divider()
