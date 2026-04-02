"""
Turn Cosmos per-frame captions into a structured summary via a local Ollama model.

Uses the same markdown template as the heuristic path (summary_templates).
Env: OLLAMA_HOST (optional), OLLAMA_MODEL (default llama3.2), COSMOS_MODEL_LABEL (display only).
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List

from summary_templates import (
    DEFAULT_VISION_MODEL_LABEL,
    metadata_line,
    ollama_user_prompt,
)


def _format_ts(seconds: float) -> str:
    from video_processor import VideoProcessor

    return VideoProcessor().format_timestamp(seconds)


def build_frame_transcript(
    frame_descriptions: List[Dict[str, str]],
    timestamps: List[float],
    format_timestamp: Callable[[float], str] | None = None,
) -> str:
    fmt = format_timestamp or _format_ts
    lines: List[str] = []
    for fd, ts in zip(frame_descriptions, timestamps):
        desc = (fd.get("description") or "").strip()
        lines.append(f"[{fmt(ts)}] {desc}")
    return "\n".join(lines)


def summarize_frames_with_ollama(
    frame_descriptions: List[Dict[str, str]],
    timestamps: List[float],
    style: str = "formal",
    model: str | None = None,
    host: str | None = None,
    vision_model: str | None = None,
) -> str:
    import ollama

    resolved_model = model or os.getenv("OLLAMA_MODEL", "llama3.2")
    resolved_host = host or os.getenv("OLLAMA_HOST")
    vision = vision_model or os.getenv("COSMOS_MODEL_LABEL", DEFAULT_VISION_MODEL_LABEL)

    client = ollama.Client(host=resolved_host) if resolved_host else ollama.Client()

    transcript = build_frame_transcript(frame_descriptions, timestamps)
    if not transcript.strip():
        return metadata_line(style, "ollama", vision) + "\n\n_No content to summarize._"

    user_prompt = ollama_user_prompt(transcript, style, vision_model=vision)

    response = client.chat(
        model=resolved_model,
        messages=[{"role": "user", "content": user_prompt}],
    )
    message = response.get("message") or {}
    body = (message.get("content") or "").strip()
    if not body:
        return metadata_line(style, "ollama", vision) + "\n\n_Ollama returned an empty response._"

    header = metadata_line(style, "ollama", vision)
    return f"{header}\n\n{body}"
