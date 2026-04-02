"""
Optional on-disk copy of uploaded videos (metadata + summarization still go to Postgres).

Set SAVE_UPLOADED_VIDEOS=1 (or true) to enable. Files land under data/uploads/.
"""
from __future__ import annotations

import os
import re
import uuid
from pathlib import Path


def _sanitize_filename(name: str) -> str:
    base = Path(name).name
    base = re.sub(r"[^\w.\-]", "_", base, flags=re.UNICODE)
    return base[:180] if len(base) > 180 else base


def persist_uploaded_video(file_bytes: bytes, original_filename: str | None) -> Path | None:
    if os.getenv("SAVE_UPLOADED_VIDEOS", "").lower() not in ("1", "true", "yes"):
        return None

    root = Path(__file__).resolve().parent / "data" / "uploads"
    root.mkdir(parents=True, exist_ok=True)

    safe = _sanitize_filename(original_filename or "video.mp4")
    stem = Path(safe).stem
    suffix = Path(safe).suffix or ".mp4"
    out = root / f"{uuid.uuid4().hex[:12]}_{stem}{suffix}"
    out.write_bytes(file_bytes)
    return out
