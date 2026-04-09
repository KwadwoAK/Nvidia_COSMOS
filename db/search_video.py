"""
Search video summaries in PostgreSQL with pgvector (cosine similarity).
Caller is responsible for computing query embeddings (e.g. via embeddings.embedder).
"""
# query Supabase database for video_summaries

from __future__ import annotations

from typing import Any

from db.connection import get_connection


def _ensure_vector_registered(conn: Any) -> None:
    """Register pgvector type on the connection so vector columns work."""
    try:
        from pgvector.psycopg2 import register_vector

        register_vector(conn)
    except ImportError:
        pass  # pgvector not installed; raw SQL may still work with cast


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
