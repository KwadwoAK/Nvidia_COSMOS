"""
Canonical summarization layout for all video summaries (heuristic + Ollama).

Every stored summary uses the same markdown skeleton so DB rows and local files stay consistent.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List

# Bump when you change section names or semantics
TEMPLATE_ID = "cosmos_summary_v1"
DEFAULT_VISION_MODEL_LABEL = "Cosmos-Reason2-2B"

# Sidebar labels -> internal keys (stored in DB / metadata)
ANALYSIS_STYLES: list[tuple[str, str]] = [
    ("Bullet points", "bullet_points"),
    ("Concise", "concise"),
    ("Formal", "formal"),
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


def ollama_user_prompt(
    transcript: str,
    style: str,
    vision_model: str = DEFAULT_VISION_MODEL_LABEL,
) -> str:
    """
    Instructions for the LLM: same three sections for every analysis type; tone/length changes by style.
    """
    style_key = (style or "formal").lower().strip().replace(" ", "_")
    # Normalize legacy values
    if style_key == "bulletpoints":
        style_key = "bullet_points"
    if style_key in ("detailed", "deep"):
        style_key = "formal"

    style_rules = {
        "bullet_points": (
            "## Overview\n"
            "One short paragraph (2–3 sentences max) setting context.\n\n"
            "## Chronological highlights\n"
            "Prefer bullets: one line per important moment with timestamp. Include as many distinct moments as the notes support.\n\n"
            "## Takeaways\n"
            "Bullet list only; each line one clear takeaway.",
            "Tone: direct and scannable. Prefer bullets over prose outside Overview.",
        ),
        "concise": (
            "## Overview\n"
            "At most 2 short sentences: what the clip is and the main arc.\n\n"
            "## Chronological highlights\n"
            "3–6 bullets with timestamps; merge redundant adjacent frames.\n\n"
            "## Takeaways\n"
            "1–3 bullets: only the essentials.",
            "Tone: brief and neutral; no filler.",
        ),
        "formal": (
            "## Overview\n"
            "2–4 complete sentences in a neutral, professional register (third person, no slang, no contractions). Summarize scope and purpose of the footage.\n\n"
            "## Chronological highlights\n"
            "5–10 bullets with timestamps; each bullet a full clause in formal language suitable for a report or memo.\n\n"
            "## Takeaways\n"
            "2–5 formal bullet points; suitable for stakeholders or documentation.",
            "Tone: formal report — clear, impersonal, precise.",
        ),
    }
    structure_text, tone_rule = style_rules.get(
        style_key,
        style_rules["formal"],
    )

    return f"""You turn timestamped frame descriptions into a structured video summary.

Vision captions came from {vision_model}. Use ONLY information supported by the frame notes below.

Frame notes:
{transcript}

Output MUST use this exact markdown shape (these three headings, in order). Match the language of the frame notes unless they are mixed, then use English.

{structure_text}

Rules:
- Do not invent objects, people, or events missing from the frame notes.
- {tone_rule}
"""


def format_heuristic_summary(
    frame_descriptions: List[Dict[str, str]],
    timestamps: List[float],
    style: str,
    format_timestamp,
    vision_model: str = DEFAULT_VISION_MODEL_LABEL,
) -> str:
    """Deterministic summary that mirrors the Ollama section layout."""
    if not frame_descriptions:
        return metadata_line(style, "heuristic", vision_model) + "\n\n_No content to summarize._"

    style_key = (style or "formal").lower().strip().replace(" ", "_")
    if style_key in ("detailed", "deep"):
        style_key = "formal"
    if style_key == "bulletpoints":
        style_key = "bullet_points"

    lines: List[str] = [metadata_line(style_key, "heuristic", vision_model), ""]

    # Overview: first + last + count
    first = (frame_descriptions[0].get("description") or "").strip()
    last = (frame_descriptions[-1].get("description") or "").strip()
    dur = format_timestamp(timestamps[-1]) if timestamps else "unknown"
    lines.append("## Overview")
    if style_key == "concise":
        lines.append(
            f"The clip runs through **{dur}** (sampled frames). It begins with: {first[:280]}{'…' if len(first) > 280 else ''} "
            f"It ends with: {last[:280]}{'…' if len(last) > 280 else ''}"
        )
    elif style_key == "formal":
        lines.append(
            f"The audiovisual material (approximately **{dur}** of sampled content) initially presents the following. "
            f"{first} "
            f"The sequence concludes as follows. {last}"
        )
    else:
        # bullet_points: short context paragraph
        lines.append(
            f"**Duration (sampled):** {dur}. **Start:** {first[:350]}{'…' if len(first) > 350 else ''} "
            f"**End:** {last[:350]}{'…' if len(last) > 350 else ''}"
        )
    lines.append("")

    lines.append("## Chronological highlights")
    if style_key == "bullet_points":
        max_bullets = min(len(frame_descriptions), 24)
    elif style_key == "concise":
        max_bullets = 6
    else:
        max_bullets = 10
    step = max(1, len(frame_descriptions) // max_bullets) if max_bullets < len(frame_descriptions) else 1
    for i in range(0, len(frame_descriptions), step):
        ts = format_timestamp(timestamps[i]) if i < len(timestamps) else "?"
        desc = (frame_descriptions[i].get("description") or "").strip()
        short = desc if len(desc) <= 400 else desc[:397] + "…"
        lines.append(f"- **[{ts}]** {short}")
    lines.append("")

    lines.append("## Takeaways")
    topics = extract_keywords_from_frames(frame_descriptions)
    if topics:
        for t in topics[:5]:
            lines.append(f"- Recurring theme: **{t}**")
    else:
        lines.append("- No strong recurring keywords detected from sampled frames.")
    lines.append("")
    lines.append(f"_Generated with template `{TEMPLATE_ID}` (heuristic)._")

    return "\n".join(lines)


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
