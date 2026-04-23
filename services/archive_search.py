from typing import Optional

from db.search_video import search_similar_by_text


def run_archive_search(query: str, limit: int = 10) -> tuple[list[dict], Optional[str]]:
    try:
        results = search_similar_by_text(query.strip(), limit=limit)
        return results, None
    except Exception as exc:
        return [], str(exc)
