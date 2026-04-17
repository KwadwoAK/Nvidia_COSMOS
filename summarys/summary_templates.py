"""
Canonical summarization layout for Ollama-generated summaries.

Metadata helpers and prompts; embeddings/search reuse the same template header.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List

# Bump when you change section names or semantics
TEMPLATE_ID = "cosmos_summary_v1"
DEFAULT_VISION_MODEL_LABEL = "Cosmos-Reason2-8B"

# Sidebar labels -> internal keys (stored in DB / metadata)
ANALYSIS_STYLES: list[tuple[str, str]] = [
    ("Bullet points", "bullet_points"),
    ("Concise", "concise"),
    ("Formal", "formal"),
    ("Municipal report (detailed)", "municipal_report"),
]


def style_key_from_label(label: str) -> str:
    for display, key in ANALYSIS_STYLES:
        if display == label:
            return key
    return "formal"


def metadata_line(
    style: str,
    engine: str,
    vision_model: str = DEFAULT_VISION_MODEL_LABEL,
) -> str:
    """Machine-readable first line (safe for embeddings; strip if you need plain prose only)."""
    return (
        f"<!-- summary_template:{TEMPLATE_ID} | style:{style} | engine:{engine} | "
        f"vision:{vision_model} -->"
    )


def _ollama_municipal_report_prompt(transcript: str, vision_model: str) -> str:
    return f"""You are drafting an official field observation record from timestamped visual descriptions produced by {vision_model}.

Use only facts supported by the notes below.

Frame notes:
{transcript}

Write markdown with these sections:
## Record identification
## Executive summary
## Detailed chronological account
## Persons, vehicles, objects, and environment
## Actions and sequence of events
## Uncertainties, occlusions, and limitations
## Administrative closing
"""


def ollama_user_prompt(
    transcript: str,
    style: str,
    vision_model: str = DEFAULT_VISION_MODEL_LABEL,
) -> str:
    style_key = (style or "formal").lower().strip().replace(" ", "_")
    if style_key == "municipal_report":
        return _ollama_municipal_report_prompt(transcript, vision_model)

    return f"""You turn timestamped frame descriptions into a structured video summary.

Vision captions came from {vision_model}. Use ONLY information supported by the frame notes below.

Frame notes:
{transcript}

Output MUST use this markdown shape:
## Overview
## Chronological highlights
## Takeaways

Keep wording concise and factual. No speculation.
"""


def extract_keywords_from_frames(frame_descriptions: List[Dict[str, str]]) -> List[str]:
    """Public alias for topic-like tokens from frame captions (search hints)."""
    return _rough_topics(frame_descriptions)


def _rough_topics(frame_descriptions: List[Dict[str, str]]) -> List[str]:
    common = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
        "is", "are", "was", "were", "this", "that", "these", "those", "there", "here",
    }
    freq: Dict[str, int] = {}
    for fd in frame_descriptions:
        for w in (fd.get("description") or "").lower().split():
            w = re.sub(r"[^a-z0-9]", "", w)
            if len(w) > 5 and w not in common:
                freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: -x[1])
    return [w for w, _ in ranked[:8]]


def parse_template_id_from_summary(text: str) -> str | None:
    m = re.search(r"summary_template:([a-zA-Z0-9_\-]+)", text[:500])
    return m.group(1) if m else None


def record_for_storage(
    *,
    summary_text: str,
    filename: str | None,
    duration_sec: float | None,
    style: str,
    engine: str,
    vision_model: str = DEFAULT_VISION_MODEL_LABEL,
    search_text: str | None = None,
) -> dict[str, Any]:
    """Row shape for JSONL local store (and optional future API)."""
    row: dict[str, Any] = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "template_id": TEMPLATE_ID,
        "vision_model": vision_model,
        "summary_style": style,
        "summary_engine": engine,
        "filename": filename,
        "duration_sec": duration_sec,
        "summary_text": summary_text,
    }
    if search_text:
        row["search_text"] = search_text
    return row


def jsonl_dumps(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False) + "\n"

