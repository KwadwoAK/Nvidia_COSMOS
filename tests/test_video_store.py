from __future__ import annotations

import re
from typing import Any

import db.video_store as video_store


class FakeCursor:
    def __init__(self, conn: "FakeConnection"):
        self.conn = conn
        self.description: list[tuple[str]] = []
        self._fetchall: list[tuple[Any, ...]] = []
        self._fetchone: tuple[Any, ...] | None = None

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> None:
        sql_clean = " ".join(sql.strip().split()).lower()
        self.conn.executed.append((sql, params))

        if "information_schema.columns" in sql_clean:
            self._fetchall = [(c,) for c in self.conn.table_columns]
            self.description = [("column_name",)]
            return

        if "insert into video_summaries" in sql_clean:
            self._fetchone = (self.conn.next_id,)
            self.description = [("id",)]
            return

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._fetchall

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._fetchone


class FakeConnection:
    def __init__(self, table_columns: set[str], next_id: int = 1001):
        self.table_columns = table_columns
        self.next_id = next_id
        self.executed: list[tuple[str, tuple[Any, ...] | None]] = []
        self.committed = False
        self.closed = False
        self.rolled_back = False

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True

    def close(self) -> None:
        self.closed = True


def _extract_insert_columns(sql: str) -> list[str]:
    m = re.search(r"INSERT INTO\s+video_summaries\s*\((.*?)\)\s*VALUES", sql, re.I | re.S)
    if not m:
        return []
    return [c.strip() for c in m.group(1).split(",")]


def test_insert_summary_old_schema_omits_template_aware_fields(monkeypatch):
    old_cols = {"id", "filename", "duration_sec", "summary_style", "summary_text", "embedding"}
    conn = FakeConnection(old_cols, next_id=2001)

    monkeypatch.setattr(video_store, "get_connection", lambda: conn)
    monkeypatch.setattr(video_store, "_ensure_vector_registered", lambda _: None)
    monkeypatch.setattr(video_store, "Vector", lambda values: values, raising=False)

    row_id = video_store.insert_summary(
        filename="clip.mp4",
        duration_sec=7.5,
        summary_style="concise",
        summary_text="Summary text",
        embedding=[0.1, 0.2, 0.3],
        summary_engine="gemma4",
        vision_model="Cosmos-Reason2-8B",
        template_id="cosmos_summary_v1",
        search_text="summary blob",
        storage_object_path="user/file.mp4",
    )

    assert row_id == 2001
    insert_sql = next(sql for sql, _ in conn.executed if "INSERT INTO video_summaries" in sql)
    used_cols = set(_extract_insert_columns(insert_sql))
    assert "summary_engine" not in used_cols
    assert "vision_model" not in used_cols
    assert "template_id" not in used_cols
    assert "search_text" not in used_cols
    assert "storage_object_path" not in used_cols
    assert conn.committed is True
    assert conn.closed is True


def test_insert_summary_new_schema_includes_template_aware_fields(monkeypatch):
    base_cols = {"id", "filename", "duration_sec", "summary_style", "summary_text", "embedding"}
    new_cols = base_cols | {
        "summary_engine",
        "vision_model",
        "template_id",
        "search_text",
        "storage_object_path",
    }
    conn = FakeConnection(new_cols, next_id=2002)

    monkeypatch.setattr(video_store, "get_connection", lambda: conn)
    monkeypatch.setattr(video_store, "_ensure_vector_registered", lambda _: None)
    monkeypatch.setattr(video_store, "Vector", lambda values: values, raising=False)

    row_id = video_store.insert_summary(
        filename="clip2.mp4",
        duration_sec=12.0,
        summary_style="formal",
        summary_text="Detailed summary text",
        embedding=[0.4, 0.5, 0.6],
        summary_engine="gemma4",
        vision_model="Cosmos-Reason2-8B",
        template_id="cosmos_summary_v1",
        search_text="searchable summary and captions",
        storage_object_path="user/file2.mp4",
    )

    assert row_id == 2002
    insert_sql = next(sql for sql, _ in conn.executed if "INSERT INTO video_summaries" in sql)
    used_cols = set(_extract_insert_columns(insert_sql))
    assert {"summary_engine", "vision_model", "template_id", "search_text", "storage_object_path"}.issubset(
        used_cols
    )
    assert conn.committed is True
    assert conn.closed is True
