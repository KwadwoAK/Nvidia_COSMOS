"""
Turn Cosmos per-frame captions into a structured summary via a local Ollama model.

Uses prompts from ``summary_templates`` (including the long municipal / incident-record style).

Environment variables:
  OLLAMA_HOST — optional API base
  OLLAMA_MODEL — model name (default llama3.2)
  OLLAMA_NUM_PREDICT — max new tokens (optional)
  OLLAMA_NUM_PREDICT_MUNICIPAL — override for ``municipal_report`` style (default 8192)
  COSMOS_MODEL_LABEL — label embedded in prompts
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

    sk = (style or "").lower().strip().replace(" ", "_")
    if os.getenv("OLLAMA_NUM_PREDICT"):
        num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "4096"))
    elif sk == "municipal_report":
        num_predict = int(os.getenv("OLLAMA_NUM_PREDICT_MUNICIPAL", "8192"))
    else:
        num_predict = int(os.getenv("OLLAMA_NUM_PREDICT_DEFAULT", "4096"))
    options = {"num_predict": num_predict}

    kwargs: dict = {
        "model": resolved_model,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    if options:
        kwargs["options"] = options

    response = client.chat(**kwargs)
    message = response.get("message") or {}
    body = (message.get("content") or "").strip()
    if not body:
        return metadata_line(style, "ollama", vision) + "\n\n_Ollama returned an empty response._"

    header = metadata_line(style, "ollama", vision)
    return f"{header}\n\n{body}"
