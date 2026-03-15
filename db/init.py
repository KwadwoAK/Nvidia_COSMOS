"""
Initialize the database: enable pgvector and create video_summaries if missing.
Requires DATABASE_URL (e.g. postgresql://user:pass@localhost:5432/cosmos_videos).
The database must already exist; this does not create it.
Run once: python -m db.init  or  from db.init import run_init; run_init()
"""

"""
Initialize the database: create the app DB if missing, enable pgvector, create table.
Uses DATABASE_URL (e.g. postgresql://user:pass@localhost:5432/cosmos_videos).
If the database doesn't exist, connects to the same server with database=postgres
to create it, then runs extension + table on the target DB.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse, urlunparse

import psycopg2  # use same as db.connection

# Load env so DATABASE_URL is set when run as __main__
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DATABASE_URL = os.getenv("DATABASE_URL")
DB_NAME_DEFAULT = "cosmos_videos"


def _get_server_url(url: str, default_db: str = "postgres") -> str:
    """Same host/user/password as url, but connect to default_db (e.g. postgres)."""
    parsed = urlparse(url)
    path = f"/{default_db}" if not parsed.path or parsed.path == "/" else f"/{default_db}"
    return urlunparse(parsed._replace(path=path))


def _ensure_database_exists(target_url: str, default_db: str = "postgres") -> None:
    """Create the target database if it doesn't exist (connects to default_db to run CREATE DATABASE)."""
    parsed = urlparse(target_url)
    dbname = (parsed.path or "").strip("/") or DB_NAME_DEFAULT
    server_url = _get_server_url(target_url, default_db)

    conn = None
    try:
        conn = psycopg2.connect(server_url)
        conn.autocommit = True  # CREATE DATABASE cannot run inside a transaction
        with conn.cursor() as cur:
            cur.execute(f'CREATE DATABASE "{dbname}"')
    except psycopg2.errors.DuplicateDatabase:  # 42P04
        pass  # already exists
    except Exception:
        raise
    finally:
        if conn:
            conn.close()


def run_init() -> None:
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set")

    # Create the app database if it doesn't exist
    _ensure_database_exists(DATABASE_URL)

    # Run extension + table on the app database
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS video_summaries (
                    id            BIGSERIAL PRIMARY KEY,
                    created_at    TIMESTAMPTZ DEFAULT NOW(),
                    filename      TEXT,
                    duration_sec  NUMERIC(10,2),
                    summary_style TEXT,
                    summary_text  TEXT NOT NULL,
                    embedding     vector(384)
                );
            """)
        conn.commit()
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    run_init()