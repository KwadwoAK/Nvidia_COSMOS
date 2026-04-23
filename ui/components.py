from typing import Optional

import streamlit as st


def format_filesize(num_bytes: Optional[int]) -> str:
    """Return a human-readable file size."""
    if not num_bytes:
        return "Unavailable"
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def format_duration(seconds: Optional[float]) -> str:
    """Return a readable minutes:seconds string."""
    if not seconds or seconds <= 0:
        return "Not available"
    total_seconds = int(round(seconds))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def render_metric_card(label: str, value: str, caption: str) -> None:
    """Render a compact presentation card."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-card-label">{label}</div>
            <div class="metric-card-value">{value}</div>
            <div class="metric-card-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
