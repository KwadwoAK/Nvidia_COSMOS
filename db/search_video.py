"""
Search video summaries in PostgreSQL with pgvector and prepare searchable text.
Caller is responsible for computing query embeddings (e.g. via embeddings.embedder).
"""

from __future__ import annotations

import re
from typing import Any

_MAX_EMBED_CHARS = 10_000

from db.connection import get_connection


def _ensure_vector_registered(conn: Any) -> None:
    """Register pgvector type on the connection so vector columns work."""
    try:
        from pgvector.psycopg2 import register_vector

        register_vector(conn)
    except ImportError:
        pass  


def build_search_text(
    summary: str,
    frame_descriptions: list[dict[str, str]],
) -> str:
    """Build the text blob that gets embedded and stored for DB-backed search."""
    parts: list[str] = [summary.strip(), "", "=== Per-frame captions ===", ""]
    for fd in frame_descriptions:
        desc = (fd.get("description") or "").strip()
        if desc:
            parts.append(desc)
    return "\n".join(parts).strip()[:_MAX_EMBED_CHARS]


def suggest_search_terms(
    frame_descriptions: list[dict[str, str]],
    max_terms: int = 15,
) -> list[str]:
    """Return lightweight keyword hints derived from frame captions."""
    from summary_templates import extract_keywords_from_frames

    topics = extract_keywords_from_frames(frame_descriptions)
    text = " ".join((fd.get("description") or "") for fd in frame_descriptions).lower()
    words = re.findall(r"[a-záéíóúñ]{4,}", text)
    bigrams: dict[str, int] = {}
    for i in range(len(words) - 1):
        bg = f"{words[i]} {words[i + 1]}"
        bigrams[bg] = bigrams.get(bg, 0) + 1
    top_bigrams = sorted(bigrams.items(), key=lambda item: -item[1])[:8]
    out = list(topics[: max_terms // 2])
    for bg, _ in top_bigrams:
        if bg not in out and len(out) < max_terms:
            out.append(bg)
    return out[:max_terms]


def search_similar(
    query_embedding: list[float],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    Return video summaries most similar to the query embedding (cosine distance).

    Args:
        query_embedding: List of 384 floats from the same model as stored embeddings.
        limit: Max number of results (default 10).

    Returns:
        List of dicts with keys: id, created_at, filename, duration_sec,
        summary_style, summary_text, distance (cosine distance; lower = more similar).
    """
    conn = None
    try:
        conn = get_connection()
        if conn is None:
            return []
        _ensure_vector_registered(conn)

        from pgvector import Vector

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, created_at, filename, duration_sec, summary_style,
                       summary_text, embedding <=> %s AS distance
                FROM video_summaries
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (Vector(query_embedding), Vector(query_embedding), limit),
            )
            columns = [d[0] for d in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]
    except Exception:
        raise
    finally:
        if conn:
            conn.close()
