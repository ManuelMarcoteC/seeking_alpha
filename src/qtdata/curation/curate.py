"""Raw -> curated promotion.

Reads not-yet-curated raw payloads, normalizes to the canonical schema,
quarantines schema violations, dedupes on the primary key (latest ingest wins),
upserts into the curated tables, recomputes adjustment factors and runs the
anomaly detectors for the affected tickers. Raw files are never modified.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import pandas as pd

from qtdata.config import Settings
from qtdata.curation.adjustments import compute_adjustment_factors
from qtdata.models import (
    ACTIONS_COLUMNS,
    ACTIONS_KEY,
    FACTORS_KEY,
    OHLCV_COLUMNS,
    OHLCV_KEY,
    Dataset,
)
from qtdata.storage import parquet_store
from qtdata.storage.catalog import Catalog
from qtdata.validation.anomalies import run_detectors
from qtdata.validation.report import ValidationReport, persist_quarantine, persist_report
from qtdata.validation.schemas import ACTIONS_SCHEMA, OHLCV_SCHEMA, validate_frame

logger = logging.getLogger(__name__)


@dataclass
class CurationSummary:
    run_id: str
    files_processed: int = 0
    rows_upserted: int = 0
    rows_quarantined: int = 0
    flags_written: int = 0
    tickers: list[str] = field(default_factory=list)


def _uncurated_files(
    settings: Settings, catalog: Catalog, dataset: Dataset, tickers: list[str] | None
) -> list[Path]:
    pattern = f"provider=*/dataset={dataset}/ticker=*/*.parquet"
    files = sorted(settings.raw_dir.glob(pattern))
    if tickers is not None:
        wanted = {f"ticker={t}" for t in tickers}
        files = [f for f in files if f.parent.name in wanted]
    return [f for f in files if not catalog.is_file_curated(f)]


def _normalize_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")
    df["ticker"] = df["ticker"].astype(str).str.upper()
    return df[OHLCV_COLUMNS]


def _normalize_actions(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["ex_date"] = pd.to_datetime(df["ex_date"]).dt.tz_localize(None).dt.normalize()
    df["value"] = pd.to_numeric(df["value"], errors="coerce").astype(float)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["action_type"] = df["action_type"].astype(str)
    return df[ACTIONS_COLUMNS]


def curate_corporate_actions(
    settings: Settings, catalog: Catalog, tickers: list[str] | None = None
) -> CurationSummary:
    run_id = uuid4().hex[:12]
    summary = CurationSummary(run_id=run_id)
    files = _uncurated_files(settings, catalog, Dataset.CORPORATE_ACTIONS, tickers)
    if not files:
        return summary

    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = _normalize_actions(raw)
    df = (
        df.sort_values("ingested_at")
        .drop_duplicates(subset=ACTIONS_KEY, keep="last")
        .reset_index(drop=True)
    )
    valid, failures = validate_frame(df, ACTIONS_SCHEMA)
    persist_quarantine(failures, run_id, settings)

    res = parquet_store.upsert(
        valid, settings.curated_dir / "corporate_actions", ACTIONS_KEY, partition_col=None
    )
    for f in files:
        catalog.mark_file_curated(f)
    summary.files_processed = len(files)
    summary.rows_upserted = res.rows_written
    summary.rows_quarantined = len(failures["index"].unique()) if not failures.empty else 0
    summary.tickers = sorted(valid["ticker"].unique())
    return summary


def curate_ohlcv(
    settings: Settings, catalog: Catalog, tickers: list[str] | None = None
) -> CurationSummary:
    run_id = uuid4().hex[:12]
    summary = CurationSummary(run_id=run_id)
    files = _uncurated_files(settings, catalog, Dataset.OHLCV_DAILY, tickers)
    if not files:
        return summary

    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = _normalize_ohlcv(raw)
    df = (
        df.sort_values("ingested_at")
        .drop_duplicates(subset=OHLCV_KEY, keep="last")
        .reset_index(drop=True)
    )
    valid, failures = validate_frame(df, OHLCV_SCHEMA)
    persist_quarantine(failures, run_id, settings)
    if valid.empty:
        logger.warning("All %d rows quarantined; nothing promoted", len(df))
        for f in files:
            catalog.mark_file_curated(f)
        summary.files_processed = len(files)
        summary.rows_quarantined = len(df)
        return summary

    out = valid.copy()
    out["volume"] = out["volume"].astype("int64")
    out["year"] = out["date"].dt.year
    res = parquet_store.upsert(
        out, settings.curated_dir / "ohlcv_daily", OHLCV_KEY, partition_col="year"
    )

    affected = sorted(out["ticker"].unique())

    # Re-derive adjustment factors and anomaly flags over the FULL curated series
    # of the affected tickers (detectors need history, not just the increment).
    curated = parquet_store.read(
        settings.curated_dir / "ohlcv_daily", filters=[("ticker", "in", affected)]
    )
    actions = parquet_store.read(settings.curated_dir / "corporate_actions")
    if not actions.empty:
        actions = actions[actions["ticker"].isin(affected)]

    factors = compute_adjustment_factors(curated, actions)
    if not factors.empty:
        factors["year"] = pd.to_datetime(factors["date"]).dt.year
        parquet_store.upsert(
            factors, settings.curated_dir / "adjustment_factors", FACTORS_KEY, partition_col="year"
        )

    flags = run_detectors(curated, actions, settings)
    report = ValidationReport(run_id=run_id, flags=flags, quarantined=failures)
    persist_report(report, settings)

    for f in files:
        catalog.mark_file_curated(f)

    summary.files_processed = len(files)
    summary.rows_upserted = res.rows_written
    summary.rows_quarantined = len(failures["index"].unique()) if not failures.empty else 0
    summary.flags_written = len(flags)
    summary.tickers = affected
    return summary


def curate_all(
    settings: Settings, catalog: Catalog, tickers: list[str] | None = None
) -> tuple[CurationSummary, CurationSummary]:
    """Actions first (factors depend on them), then OHLCV."""
    actions_summary = curate_corporate_actions(settings, catalog, tickers)
    ohlcv_summary = curate_ohlcv(settings, catalog, tickers)
    catalog.refresh_views()
    return actions_summary, ohlcv_summary
