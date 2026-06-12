"""Ingestion audit log: every fetch attempt is recorded, success or not."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import duckdb
import pandas as pd

from qtdata.models import Dataset


@dataclass
class ManifestEntry:
    run_id: str
    provider: str
    dataset: Dataset
    ticker: str
    requested_start: date | None
    requested_end: date | None
    rows_fetched: int
    payload_sha256: str | None
    status: str  # 'success' | 'empty' | 'skipped' | 'failed'
    error: str | None
    started_at: datetime
    finished_at: datetime


def record_fetch(conn: duckdb.DuckDBPyConnection, entry: ManifestEntry) -> None:
    conn.execute(
        "INSERT INTO ingestion_manifest VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            entry.run_id,
            entry.provider,
            str(entry.dataset),
            entry.ticker,
            entry.requested_start,
            entry.requested_end,
            entry.rows_fetched,
            entry.payload_sha256,
            entry.status,
            entry.error,
            entry.started_at,
            entry.finished_at,
        ],
    )


def recent_runs(conn: duckdb.DuckDBPyConnection, limit: int = 20) -> pd.DataFrame:
    return conn.execute(
        """
        SELECT run_id, provider, dataset,
               MIN(started_at) AS started_at,
               COUNT(*) AS fetches,
               SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS ok,
               SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed,
               SUM(rows_fetched) AS rows
        FROM ingestion_manifest
        GROUP BY run_id, provider, dataset
        ORDER BY started_at DESC
        LIMIT ?
        """,
        [limit],
    ).df()


def failures_for_run(conn: duckdb.DuckDBPyConnection, run_id: str) -> pd.DataFrame:
    return conn.execute(
        "SELECT ticker, dataset, error FROM ingestion_manifest "
        "WHERE run_id = ? AND status = 'failed'",
        [run_id],
    ).df()
