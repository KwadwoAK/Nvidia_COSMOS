"""
Turn Cosmos per-frame captions into a structured summary via Gemma 4.
"""
from __future__ import annotations

import os
from typing import Dict, List

from openai import OpenAI

from summarys.summary_templates import (
    DEFAULT_VISION_MODEL_LABEL,
    gemma_user_prompt,
    metadata_line,
)


def summarize_frames_with_gemma(
    frame_descriptions: List[Dict[str, str]],
    timestamps: List[float],
    style: str = "formal",
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    vision_model: str | None = None,
) -> str:
    resolved_model = model or os.getenv("SUMMARY_MODEL", "gemma-4-26B-MoE")
    resolved_base_url = base_url or os.getenv("SUMMARY_BASE_URL", "http://10.20.1.28:8010/v1")
    resolved_api_key = api_key or os.getenv("SUMMARY_API_KEY", "not-required")
    vision = vision_model or os.getenv("COSMOS_MODEL_LABEL", DEFAULT_VISION_MODEL_LABEL)

    client = OpenAI(base_url=resolved_base_url, api_key=resolved_api_key)

    lines: List[str] = []
    for fd, ts in zip(frame_descriptions, timestamps):
        desc = (fd.get("description") or "").strip()
        lines.append(f"[{ts:.2f}s] {desc}")
    transcript = "\n".join(lines).strip()
    if not transcript:
        return metadata_line(style, "gemma4", vision) + "\n\n_No content to summarize._"

    user_prompt = gemma_user_prompt(transcript, style, vision_model=vision)
    response = client.chat.completions.create(
        model=resolved_model,
        messages=[{"role": "user", "content": user_prompt}],
        max_tokens=int(os.getenv("SUMMARY_MAX_TOKENS", "4096")),
    )
    body = (response.choices[0].message.content or "").strip() if response.choices else ""
    if not body:
        return metadata_line(style, "gemma4", vision) + "\n\n_Gemma returned an empty response._"

    return f"{metadata_line(style, 'gemma4', vision)}\n\n{body}"
