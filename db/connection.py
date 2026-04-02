import os

# Optional: load .env so SUPABASE_DB_URL is set (pip install python-dotenv)
try:
    from dotenv import load_dotenv  # type: ignore[import-untyped]
    load_dotenv()
except ImportError:
    pass

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")


def get_connection():
    """Return a new connection to the database. Caller must close it."""
    if not SUPABASE_DB_URL:
        raise RuntimeError(
            "Missing SUPABASE_DB_URL. Set it in your environment or .env file."
        )
    import psycopg2  # type: ignore[import-untyped]

    return psycopg2.connect(SUPABASE_DB_URL)
