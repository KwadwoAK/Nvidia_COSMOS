"""
Store video summaries in PostgreSQL with pgvector.
Search is in db.search_video. Caller is responsible for computing embeddings
(e.g. via embeddings.embedder).
"""
# write 
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


def _table_columns(conn: Any, table_name: str) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
            """,
            (table_name,),
        )
        return {row[0] for row in cur.fetchall()}


def insert_summary(
    filename: str | None,
    duration_sec: float | None,
    summary_style: str,
    summary_text: str,
    embedding: list[float],
    summary_engine: str | None = None,
    vision_model: str | None = None,
    template_id: str | None = None,
    search_text: str | None = None,
) -> int | None:
    """
    Insert a video summary row. Returns the new row id, or None on failure.

    Args:
        filename: Original video filename (optional).
        duration_sec: Video duration in seconds (optional).
        summary_style: e.g. "detailed", "concise", "bullet points".
        summary_text: Full summary text (required).
        embedding: List of 384 floats (must match table vector(384)).
        summary_engine: e.g. "ollama" (optional; inserted if column exists).
        vision_model: Label of vision model used (optional; inserted if column exists).
        template_id: Summary template version (optional; inserted if column exists).
        search_text: Text used for semantic search (optional; inserted if column exists).
    """
    if not summary_text.strip():
        return None

    conn = None
    try:
        conn = get_connection()
        if conn is None:
            return None
        _ensure_vector_registered(conn)

        from pgvector import Vector

        row_data: dict[str, Any] = {
            "filename": filename,
            "duration_sec": duration_sec,
            "summary_style": summary_style,
            "summary_text": summary_text,
            "embedding": Vector(embedding),
            "summary_engine": summary_engine,
            "vision_model": vision_model,
            "template_id": template_id,
            "search_text": search_text,
        }
        supported_cols = _table_columns(conn, "video_summaries")
        insert_cols = [k for k in row_data.keys() if k in supported_cols]
        insert_values = [row_data[k] for k in insert_cols]
        if not insert_cols:
            return None
        placeholders = ", ".join(["%s"] * len(insert_cols))
        cols_sql = ", ".join(insert_cols)

        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO video_summaries
                  ({cols_sql})
                VALUES ({placeholders})
                RETURNING id
                """,
                insert_values,
            )
            row = cur.fetchone()
            conn.commit()
            return row[0] if row else None
    except Exception:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()
