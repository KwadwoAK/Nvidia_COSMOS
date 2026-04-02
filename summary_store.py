"""Append structured summaries to a local JSONL file (no DB required)."""
from __future__ import annotations

import os
from pathlib import Path

from summary_templates import DEFAULT_VISION_MODEL_LABEL, jsonl_dumps, record_for_storage

_STORE_DIR = Path(__file__).resolve().parent / "data" / "summaries"
_DEFAULT_FILE = "summaries.jsonl"


def append_local_summary(
    *,
    summary_text: str,
    filename: str | None,
    duration_sec: float | None,
    style: str,
    engine: str,
    vision_model: str | None = None,
    search_text: str | None = None,
) -> Path | None:
    if os.getenv("SAVE_SUMMARIES_LOCAL", "1").lower() in ("0", "false", "no"):
        return None

    _STORE_DIR.mkdir(parents=True, exist_ok=True)
    path = _STORE_DIR / os.getenv("SUMMARIES_JSONL", _DEFAULT_FILE)
    rec = record_for_storage(
        summary_text=summary_text,
        filename=filename,
        duration_sec=duration_sec,
        style=style,
        engine=engine,
        vision_model=vision_model or DEFAULT_VISION_MODEL_LABEL,
        search_text=search_text,
    )
    with path.open("a", encoding="utf-8") as f:
        f.write(jsonl_dumps(rec))
    return path
