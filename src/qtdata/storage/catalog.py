"""DuckDB catalog: operational state (watermarks, manifest, curation ledger)
plus research views over the curated parquet tables.

The parquet files remain the single source of truth; catalog.duckdb holds only
mutable bookkeeping and CREATE VIEW definitions.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from qtdata.config import Settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS watermarks (
    provider TEXT NOT NULL,
    dataset TEXT NOT NULL,
    ticker TEXT NOT NULL,
    high_water_date DATE NOT NULL,
    last_run_id TEXT,
    updated_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (provider, dataset, ticker)
);
CREATE TABLE IF NOT EXISTS ingestion_manifest (
    run_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    dataset TEXT NOT NULL,
    ticker TEXT NOT NULL,
    requested_start DATE,
    requested_end DATE,
    rows_fetched BIGINT,
    payload_sha256 TEXT,
    status TEXT NOT NULL,
    error TEXT,
    started_at TIMESTAMP,
    finished_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS curated_files (
    path TEXT PRIMARY KEY,
    curated_at TIMESTAMP DEFAULT current_timestamp
);
"""

# curated table name -> (hive partitioned?)
_CURATED_TABLES = {
    "ohlcv_daily": True,
    "corporate_actions": False,
    "universe_membership": False,
    "validation_flags": True,
    "adjustment_factors": True,
    "listing_directory": True,
    "fundamentals_snapshot": False,
    "news_articles": True,
    "news_ticker_sentiment": True,
    "sentiment_daily": True,
}


class Catalog:
    def __init__(self, settings: Settings, read_only: bool = False):
        self.settings = settings
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(settings.catalog_path), read_only=read_only)

    # -- lifecycle ---------------------------------------------------------
    def init_schema(self) -> None:
        self.conn.execute(_SCHEMA)

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:  # noqa: BLE001 — double-close is a no-op
            pass

    def __enter__(self) -> Catalog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- curation ledger ----------------------------------------------------
    def is_file_curated(self, path: Path) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM curated_files WHERE path = ?", [str(path)]
        ).fetchone()
        return row is not None

    def mark_file_curated(self, path: Path) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO curated_files (path) VALUES (?)", [str(path)]
        )

    # -- views ---------------------------------------------------------------
    def refresh_views(self) -> list[str]:
        """(Re)create research views for every curated table that has data."""
        created: list[str] = []
        available: set[str] = set()
        for table, partitioned in _CURATED_TABLES.items():
            table_dir = self.settings.curated_dir / table
            if not table_dir.exists() or not any(table_dir.rglob("*.parquet")):
                continue
            glob = (table_dir / "**" / "*.parquet").as_posix()
            hive = "true" if partitioned else "false"
            self.conn.execute(
                f"CREATE OR REPLACE VIEW {table} AS "
                f"SELECT * FROM read_parquet('{glob}', hive_partitioning={hive}, "
                f"union_by_name=true)"
            )
            created.append(table)
            available.add(table)

        if {"ohlcv_daily", "adjustment_factors"} <= available:
            self.conn.execute(
                """
                CREATE OR REPLACE VIEW ohlcv_daily_adj AS
                SELECT
                    o.ticker,
                    o.date,
                    o.open  * COALESCE(f.adj_factor, 1.0) AS open,
                    o.high  * COALESCE(f.adj_factor, 1.0) AS high,
                    o.low   * COALESCE(f.adj_factor, 1.0) AS low,
                    o.close * COALESCE(f.adj_factor, 1.0) AS close,
                    CAST(round(o.volume / COALESCE(f.split_factor, 1.0)) AS BIGINT) AS volume,
                    o.close AS close_raw,
                    COALESCE(f.adj_factor, 1.0) AS adj_factor,
                    o.source, o.run_id
                FROM ohlcv_daily o
                LEFT JOIN adjustment_factors f USING (ticker, date)
                """
            )
            created.append("ohlcv_daily_adj")

        if {"sentiment_daily", "ohlcv_daily"} <= available:
            # decayed carry-forward factor, derived on read (never materialized):
            # last observed sentiment per ticker decayed by exp(-days_since/tau)
            tau = float(self.settings.news_decay_tau_days)
            self.conn.execute(
                f"""
                CREATE OR REPLACE VIEW sentiment_daily_decayed AS
                WITH spine AS (
                    SELECT DISTINCT o.ticker, o.date
                    FROM ohlcv_daily o
                ),
                joined AS (
                    SELECT s.ticker, s.date,
                           sd.sent_av, sd.sent_finbert, sd.n_articles,
                           CASE WHEN sd.ticker IS NOT NULL THEN s.date END AS obs_date
                    FROM spine s
                    LEFT JOIN sentiment_daily sd USING (ticker, date)
                ),
                filled AS (
                    SELECT *,
                           MAX(obs_date) OVER (
                               PARTITION BY ticker ORDER BY date
                               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                           ) AS last_obs_date,
                           LAST_VALUE(sent_av IGNORE NULLS) OVER (
                               PARTITION BY ticker ORDER BY date
                               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                           ) AS last_sent_av,
                           LAST_VALUE(sent_finbert IGNORE NULLS) OVER (
                               PARTITION BY ticker ORDER BY date
                               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                           ) AS last_sent_finbert
                    FROM joined
                )
                SELECT ticker, date, sent_av, sent_finbert, n_articles,
                       last_sent_av * exp(-date_diff('day', last_obs_date, date) / {tau})
                           AS sent_av_decayed,
                       last_sent_finbert * exp(-date_diff('day', last_obs_date, date) / {tau})
                           AS sent_finbert_decayed
                FROM filled
                """
            )
            created.append("sentiment_daily_decayed")

        if {"ohlcv_daily", "validation_flags"} <= available:
            self.conn.execute(
                """
                CREATE OR REPLACE VIEW ohlcv_daily_clean AS
                SELECT o.*, COALESCE(fl.n_flags, 0) AS n_flags, fl.flag_types
                FROM ohlcv_daily o
                LEFT JOIN (
                    SELECT ticker, date, COUNT(*) AS n_flags,
                           string_agg(DISTINCT flag_type, ',') AS flag_types
                    FROM validation_flags
                    GROUP BY ticker, date
                ) fl USING (ticker, date)
                """
            )
            created.append("ohlcv_daily_clean")
        return created

    # -- queries --------------------------------------------------------------
    def query(self, sql: str) -> pd.DataFrame:
        return self.conn.execute(sql).df()
