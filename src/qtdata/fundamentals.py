"""Fundamentals snapshot ingestion (stockanalysis.com screener export).

The webinar's screener_us.csv (~5,300 tickers x 308 columns) is a CURRENT
snapshot — survivorship-biased and not point-in-time. It is ingested for the
agent layer and research queries only, never as a PIT factor source; the bias
is recorded in every row's `note` column and keyed by `as_of`.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from uuid import uuid4

import pandas as pd

from qtdata.config import Settings
from qtdata.models import FUNDAMENTALS_KEY
from qtdata.storage import parquet_store
from qtdata.storage.catalog import Catalog

logger = logging.getLogger(__name__)

SNAPSHOT_NOTE = (
    "STATIC SNAPSHOT (stockanalysis.com screener export): current-state, "
    "survivorship-biased; research/agent use only — never a PIT factor source."
)

# columns that must stay text even if most values look numeric
_TEXT_COLUMNS = {
    "symbol", "name", "industry", "sector", "exchange", "country", "usState",
    "country_code", "marketCapCategory", "analystRatings", "analystRatingsTop",
    "tags", "website", "financialCurrency", "priceCurrency", "fiscalYearEnd",
    "earningsTime", "payoutFrequency", "lastSplitType", "sic", "cik", "cusip", "isin",
}


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Adopt a numeric dtype only when >=80% of non-null values convert cleanly."""
    out = df.copy()
    for col in out.columns:
        if col in _TEXT_COLUMNS or col.lower().endswith("date"):
            continue
        if out[col].dtype != object:
            continue
        original_nonnull = out[col].notna().sum()
        if original_nonnull == 0:
            continue
        converted = pd.to_numeric(out[col], errors="coerce")
        if converted.notna().sum() >= 0.8 * original_nonnull:
            out[col] = converted
    return out


def ingest_screener_csv(
    settings: Settings,
    catalog: Catalog,
    csv_path: Path,
    as_of: date,
    note: str = SNAPSHOT_NOTE,
) -> int:
    """Snapshot to raw, coerce, upsert curated `fundamentals_snapshot`. Returns rows."""
    run_id = uuid4().hex[:12]
    raw = pd.read_csv(csv_path, low_memory=False)
    if "symbol" not in raw.columns:
        raise ValueError(f"{csv_path} does not look like a screener export (no 'symbol')")

    df = _coerce_numeric(raw)
    df.insert(0, "ticker", df["symbol"].astype(str).str.upper().str.strip())
    df = df.drop_duplicates(subset=["ticker"], keep="first")
    df["as_of"] = pd.Timestamp(as_of)
    df["note"] = note
    df["source"] = "stockanalysis_screener"
    df["run_id"] = run_id
    df["ingested_at"] = pd.Timestamp.now(tz="UTC")

    raw_path = (
        settings.raw_dir / "provider=stockanalysis" / "dataset=fundamentals_snapshot"
        / f"as_of={as_of}" / f"{run_id}.parquet"
    )
    parquet_store.write_raw(df, raw_path)

    res = parquet_store.upsert(
        df, settings.curated_dir / "fundamentals_snapshot", FUNDAMENTALS_KEY, partition_col=None
    )
    catalog.refresh_views()
    logger.info("Fundamentals snapshot %s: %d tickers ingested", as_of, res.rows_written)
    return res.rows_written
