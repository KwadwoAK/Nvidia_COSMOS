#!/usr/bin/env python3
"""
Single-file smoke checks for current summary/storage wiring.

What this validates:
1) app.py uses summarys.gemma_summarizer for summaries
2) summary_templates defaults + metadata parsing behave as expected
3) vision_search search text builder works
4) db.video_store.insert_summary works with:
   - old schema (minimal columns)
   - new schema (template-aware + storage_object_path when present)
5) pages/2_Semantic_search.py wires semantic search + video URL display

Run:
  python smoke_check_pipeline.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def ok(msg: str) -> None:
    print(f"[PASS] {msg}")


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    raise AssertionError(msg)


def check_app_routing() -> None:
    app_path = ROOT / "app.py"
    text = app_path.read_text(encoding="utf-8")

    if "from services.pipeline import run_generate_summary_workflow" not in text:
        fail("app.py is not routing generation through services.pipeline")
    if "from services.archive_search import run_archive_search" not in text:
        fail("app.py is not routing archive search through services.archive_search")
    if "from state.session import init_session_state" not in text:
        fail("app.py is not using centralized session state initialization")
    if "from ui.sidebar import render_sidebar" not in text:
        fail("app.py is not using ui.sidebar rendering")
    if "from ui.theme import apply_theme" not in text:
        fail("app.py is not applying theme through ui.theme")

    pipeline_path = ROOT / "services" / "pipeline.py"
    if not pipeline_path.is_file():
        fail("missing services/pipeline.py")
    pipeline_text = pipeline_path.read_text(encoding="utf-8")
    if "summarize_frames_with_gemma(" not in pipeline_text:
        fail("services/pipeline.py is not calling summarize_frames_with_gemma")
    if "style_key_from_label(summary_style)" not in pipeline_text:
        fail("services/pipeline.py is not normalizing style label via style_key_from_label")
    if "build_search_text(summary, frame_descriptions)" not in pipeline_text:
        fail("services/pipeline.py is not building search_text for embedding/storage")
    if 'summary_engine="gemma4"' not in pipeline_text:
        fail("services/pipeline.py is not marking stored summaries as gemma4 engine")
    if "upload_local_file_to_video_bucket" not in pipeline_text:
        fail("services/pipeline.py is not uploading videos to Supabase Storage when configured")
    if "storage_object_path=storage_object_path" not in pipeline_text:
        fail("services/pipeline.py is not passing storage_object_path into insert_summary")
    page = ROOT / "pages" / "2_Semantic_search.py"
    if not page.is_file():
        fail("missing pages/2_Semantic_search.py")
    ptext = page.read_text(encoding="utf-8")
    if "search_similar_by_text" not in ptext:
        fail("semantic search page is not calling search_similar_by_text")
    ok("app.py delegates orchestration and pipeline keeps Gemma/template-aware storage behavior")


def check_templates_and_search_text() -> None:
    from summarys.summary_templates import (
        DEFAULT_VISION_MODEL_LABEL,
        metadata_line,
        parse_template_id_from_summary,
    )
    from vision_search import build_search_text

    if DEFAULT_VISION_MODEL_LABEL != "Cosmos-Reason2-8B":
        fail("DEFAULT_VISION_MODEL_LABEL is not Cosmos-Reason2-8B")

    header = metadata_line("formal", "gemma4", DEFAULT_VISION_MODEL_LABEL)
    template_id = parse_template_id_from_summary(header + "\n\nBody")
    if template_id != "cosmos_summary_v1":
        fail("parse_template_id_from_summary failed to parse cosmos_summary_v1")

    blob = build_search_text(
        "## Overview\nSample summary",
        [
            {"frame_index": 0, "description": "A red car is parked near a sidewalk."},
            {"frame_index": 1, "description": "A person walks past the red car."},
        ],
    )
    if "Per-frame captions" not in blob or "red car" not in blob.lower():
        fail("build_search_text did not include expected frame caption content")
    ok("template metadata + search_text builder are working")


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
    def __init__(self, table_columns: set[str], next_id: int = 101):
        self.table_columns = table_columns
        self.next_id = next_id
        self.executed: list[tuple[str, tuple[Any, ...] | None]] = []
        self.closed = False
        self.committed = False
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


def check_video_store_insert_schemas() -> None:
    import db.video_store as video_store

    original_get_connection = video_store.get_connection
    original_ensure = video_store._ensure_vector_registered
    original_vector = getattr(video_store, "Vector", None)

    class _Vector:
        def __init__(self, values: list[float]):
            self.values = values

    # Monkeypatch local imports used inside function
    sys.modules.setdefault("pgvector", type("pgvector", (), {"Vector": _Vector}))

    try:
        video_store._ensure_vector_registered = lambda conn: None  # type: ignore[assignment]

        # Case A: old schema
        old_cols = {"id", "filename", "duration_sec", "summary_style", "summary_text", "embedding"}
        old_conn = FakeConnection(old_cols, next_id=201)
        video_store.get_connection = lambda: old_conn  # type: ignore[assignment]
        row_id = video_store.insert_summary(
            filename="clip.mp4",
            duration_sec=12.5,
            summary_style="formal",
            summary_text="hello",
            embedding=[0.1, 0.2, 0.3],
            summary_engine="gemma4",
            vision_model="Cosmos-Reason2-8B",
            template_id="cosmos_summary_v1",
            search_text="hello world",
        )
        if row_id != 201:
            fail("insert_summary failed for old schema")
        insert_sql = next(sql for sql, _ in old_conn.executed if "INSERT INTO video_summaries" in sql)
        used_cols = set(_extract_insert_columns(insert_sql))
        forbidden = {"summary_engine", "vision_model", "template_id", "search_text"}
        if used_cols & forbidden:
            fail("old schema insert unexpectedly included template-aware columns")
        ok("db insert works with old schema")

        # Case B: new schema (metadata + storage path)
        new_cols = old_cols | {
            "summary_engine",
            "vision_model",
            "template_id",
            "search_text",
            "storage_object_path",
        }
        new_conn = FakeConnection(new_cols, next_id=202)
        video_store.get_connection = lambda: new_conn  # type: ignore[assignment]
        row_id = video_store.insert_summary(
            filename="clip2.mp4",
            duration_sec=18.0,
            summary_style="concise",
            summary_text="world",
            embedding=[0.4, 0.5, 0.6],
            summary_engine="gemma4",
            vision_model="Cosmos-Reason2-8B",
            template_id="cosmos_summary_v1",
            search_text="world captions",
            storage_object_path="alice/uuid_clip2.mp4",
        )
        if row_id != 202:
            fail("insert_summary failed for new schema")
        insert_sql = next(sql for sql, _ in new_conn.executed if "INSERT INTO video_summaries" in sql)
        used_cols = set(_extract_insert_columns(insert_sql))
        required = {"summary_engine", "vision_model", "template_id", "search_text", "storage_object_path"}
        if not required.issubset(used_cols):
            fail("new schema insert did not include all template-aware + storage columns")
        ok("db insert uses template-aware fields when schema supports them")

    finally:
        video_store.get_connection = original_get_connection  # type: ignore[assignment]
        video_store._ensure_vector_registered = original_ensure  # type: ignore[assignment]
        if original_vector is not None:
            setattr(video_store, "Vector", original_vector)


def main() -> int:
    checks = [
        check_app_routing,
        check_templates_and_search_text,
        check_video_store_insert_schemas,
    ]
    try:
        for fn in checks:
            fn()
    except Exception as exc:
        print(f"\nSmoke check failed: {exc}")
        return 1

    print("\nAll smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
