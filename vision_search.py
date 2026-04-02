"""
Build searchable text from frame-level captions + summary so users can find videos by
objects / scenes (e.g. "red car", "cocina") via the same embedding model as summaries.

True pixel-level detection would need a separate detector; here we search what Cosmos (or mock) wrote.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List

# Sentence-transformers context limit — stay safely under token cap
_MAX_EMBED_CHARS = 10_000


def build_search_text(
    summary: str,
    frame_descriptions: List[Dict[str, str]],
) -> str:
    """Concatenate summary + every frame caption so embeddings catch object/scene queries."""
    parts: List[str] = [summary.strip(), "", "=== Per-frame captions ===", ""]
    for fd in frame_descriptions:
        desc = (fd.get("description") or "").strip()
        if desc:
            parts.append(desc)
    return "\n".join(parts).strip()[:_MAX_EMBED_CHARS]


def suggest_search_terms(
    frame_descriptions: List[Dict[str, str]],
    max_terms: int = 15,
) -> List[str]:
    """Lightweight keyword hints from captions (nouns-ish tokens + 2-word phrases)."""
    from summary_templates import extract_keywords_from_frames

    topics = extract_keywords_from_frames(frame_descriptions)
    text = " ".join((fd.get("description") or "") for fd in frame_descriptions).lower()
    # simple bigrams on words length > 3
    words = re.findall(r"[a-záéíóúñ]{4,}", text)
    bigrams: Dict[str, int] = {}
    for i in range(len(words) - 1):
        bg = f"{words[i]} {words[i + 1]}"
        bigrams[bg] = bigrams.get(bg, 0) + 1
    top_bigrams = sorted(bigrams.items(), key=lambda x: -x[1])[:8]
    out = list(topics[: max_terms // 2])
    for bg, _ in top_bigrams:
        if bg not in out and len(out) < max_terms:
            out.append(bg)
    return out[:max_terms]


def _cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def search_local_summaries_semantic(
    query: str,
    *,
    jsonl_path: Path | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    Semantic search over local JSONL when Supabase is not configured.
    Uses the same embedding model as the DB path.
    """
    from embeddings.embedder import embed_text

    root = Path(__file__).resolve().parent / "data" / "summaries"
    path = jsonl_path or (root / "summaries.jsonl")
    if not path.is_file():
        return []

    q = embed_text(query.strip())
    scored: list[tuple[float, dict[str, Any]]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            blob = rec.get("search_text") or rec.get("summary_text") or ""
            if not blob.strip():
                continue
            doc_emb = embed_text(blob[:_MAX_EMBED_CHARS])
            sim = _cosine_sim(q, doc_emb)
            scored.append((sim, rec))

    scored.sort(key=lambda x: -x[0])
    out: list[dict[str, Any]] = []
    for sim, rec in scored[:limit]:
        row = dict(rec)
        row["distance"] = 1.0 - sim
        row["similarity"] = sim
        out.append(row)
    return out
