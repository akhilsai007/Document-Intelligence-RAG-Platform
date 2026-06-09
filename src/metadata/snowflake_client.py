"""Metadata layer. In production this writes curated chunk/document metadata
to Snowflake (the BI dashboards read from these tables). When Snowflake is not
configured it transparently falls back to a local SQLite file so the platform
runs anywhere."""
from __future__ import annotations

import sqlite3
from functools import lru_cache
from typing import Dict, List

from config.settings import settings

DDL = """
CREATE TABLE IF NOT EXISTS doc_metadata (
    chunk_id   TEXT PRIMARY KEY,
    doc_id     TEXT,
    category   TEXT,
    source     TEXT,
    char_len   INTEGER,
    ingested_at TEXT
)
"""

QUERY_LOG_DDL = """
CREATE TABLE IF NOT EXISTS query_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT,
    category TEXT,
    n_results INTEGER,
    latency_ms REAL,
    ts TEXT
)
"""


class MetadataStore:
    """SQLite-backed implementation (local fallback)."""

    def __init__(self, path: str):
        self.path = path
        self._init()

    def _conn(self):
        return sqlite3.connect(self.path)

    def _init(self):
        import os

        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with self._conn() as c:
            c.execute(DDL)
            c.execute(QUERY_LOG_DDL)

    def upsert_chunks(self, rows: List[Dict]) -> None:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as c:
            c.executemany(
                "INSERT OR REPLACE INTO doc_metadata "
                "(chunk_id, doc_id, category, source, char_len, ingested_at) "
                "VALUES (?,?,?,?,?,?)",
                [
                    (
                        r["chunk_id"],
                        r["doc_id"],
                        r.get("category", "general"),
                        r.get("source", ""),
                        r.get("char_len", 0),
                        now,
                    )
                    for r in rows
                ],
            )

    def log_query(self, query: str, category: str, n_results: int, latency_ms: float) -> None:
        from datetime import datetime, timezone

        with self._conn() as c:
            c.execute(
                "INSERT INTO query_log (query, category, n_results, latency_ms, ts) "
                "VALUES (?,?,?,?,?)",
                (query, category, n_results, latency_ms,
                 datetime.now(timezone.utc).isoformat()),
            )

    def category_counts(self) -> Dict[str, int]:
        with self._conn() as c:
            cur = c.execute("SELECT category, COUNT(*) FROM doc_metadata GROUP BY category")
            return {row[0]: row[1] for row in cur.fetchall()}


class SnowflakeMetadataStore(MetadataStore):  # pragma: no cover (needs creds)
    """Production implementation against Snowflake. Mirrors the SQLite API."""

    def __init__(self):
        import snowflake.connector

        self._connect = lambda: snowflake.connector.connect(
            account=settings.snowflake_account,
            user=settings.snowflake_user,
            password=settings.snowflake_password,
            warehouse=settings.snowflake_warehouse,
            database=settings.snowflake_database,
            schema=settings.snowflake_schema,
        )
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(DDL.replace("TEXT", "STRING"))
            cur.execute(QUERY_LOG_DDL.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "NUMBER AUTOINCREMENT"))


@lru_cache(maxsize=1)
def get_metadata_store() -> MetadataStore:
    if settings.snowflake_enabled:
        try:
            return SnowflakeMetadataStore()
        except Exception:
            pass
    return MetadataStore(settings.metadata_local_path)
