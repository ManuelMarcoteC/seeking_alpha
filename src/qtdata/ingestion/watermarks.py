"""Per (provider, dataset, ticker) high-water marks for incremental ingestion."""

from __future__ import annotations

from datetime import date

import duckdb

from qtdata.models import Dataset


def get_watermark(
    conn: duckdb.DuckDBPyConnection, provider: str, dataset: Dataset, ticker: str
) -> date | None:
    row = conn.execute(
        "SELECT high_water_date FROM watermarks "
        "WHERE provider = ? AND dataset = ? AND ticker = ?",
        [provider, str(dataset), ticker],
    ).fetchone()
    return row[0] if row else None


def set_watermark(
    conn: duckdb.DuckDBPyConnection,
    provider: str,
    dataset: Dataset,
    ticker: str,
    high_water: date,
    run_id: str,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO watermarks "
        "(provider, dataset, ticker, high_water_date, last_run_id, updated_at) "
        "VALUES (?, ?, ?, ?, ?, current_timestamp)",
        [provider, str(dataset), ticker, high_water, run_id],
    )
