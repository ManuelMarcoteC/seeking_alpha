"""Incremental ingestion orchestrator: provider -> immutable raw layer.

Per (ticker, dataset): resolve the effective start from the watermark (plus one
trading session), fetch, append the payload to the raw layer, record the
manifest row, and advance the watermark. One failing ticker never aborts the
run — failures are isolated and logged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from uuid import uuid4

import pandas as pd

from qtdata import calendars
from qtdata.config import Settings
from qtdata.ingestion.manifest import ManifestEntry, record_fetch
from qtdata.ingestion.watermarks import get_watermark, set_watermark
from qtdata.models import Dataset
from qtdata.providers import get_provider
from qtdata.storage import parquet_store
from qtdata.storage.catalog import Catalog

logger = logging.getLogger(__name__)

DATASET_ALIASES = {
    "ohlcv": Dataset.OHLCV_DAILY,
    "ohlcv_daily": Dataset.OHLCV_DAILY,
    "actions": Dataset.CORPORATE_ACTIONS,
    "corporate_actions": Dataset.CORPORATE_ACTIONS,
}

_DATE_COL = {Dataset.OHLCV_DAILY: "date", Dataset.CORPORATE_ACTIONS: "ex_date"}


@dataclass
class IngestSummary:
    run_id: str
    ok: int = 0
    empty: int = 0
    skipped: int = 0
    failed: int = 0
    rows: int = 0
    failures: list[tuple[str, str, str]] = field(default_factory=list)  # ticker, dataset, error


def _raw_path(settings: Settings, provider: str, dataset: Dataset, ticker: str, run_id: str):
    return (
        settings.raw_dir
        / f"provider={provider}"
        / f"dataset={dataset}"
        / f"ticker={ticker}"
        / f"{run_id}.parquet"
    )


def ingest(
    settings: Settings,
    catalog: Catalog,
    tickers: list[str],
    provider_name: str | None = None,
    start: date | None = None,
    end: date | None = None,
    datasets: tuple[Dataset, ...] = (Dataset.OHLCV_DAILY, Dataset.CORPORATE_ACTIONS),
    full_refresh: bool = False,
) -> IngestSummary:
    provider = get_provider(provider_name or settings.default_provider, settings)
    run_id = uuid4().hex[:12]
    if end is None:
        # never ingest today's partial bar: it would freeze under the watermark
        end = calendars.last_completed_session(date.today(), settings.default_calendar).date()
    summary = IngestSummary(run_id=run_id)

    for ticker in tickers:
        for dataset in datasets:
            if dataset not in provider.supported_datasets():
                summary.skipped += 1
                continue
            started_at = datetime.now(UTC)

            eff_start = start
            if eff_start is None and not full_refresh:
                wm = get_watermark(catalog.conn, provider.name, dataset, ticker)
                if wm is not None:
                    try:
                        nxt = calendars.next_session(wm, settings.default_calendar).date()
                    except Exception:
                        nxt = None
                    if nxt is None or nxt > end:
                        summary.skipped += 1
                        continue
                    eff_start = nxt
            if eff_start is None:
                eff_start = settings.default_start_date

            try:
                if dataset is Dataset.OHLCV_DAILY:
                    result = provider.fetch_ohlcv(ticker, eff_start, end)
                else:
                    result = provider.fetch_corporate_actions(ticker, eff_start, end)
                df = result.df

                if df.empty:
                    status, rows = "empty", 0
                    summary.empty += 1
                else:
                    df = df.assign(
                        source=provider.name,
                        run_id=run_id,
                        ingested_at=pd.Timestamp.now(tz="UTC"),
                    )
                    parquet_store.write_raw(
                        df, _raw_path(settings, provider.name, dataset, ticker, run_id)
                    )
                    high_water = pd.Timestamp(df[_DATE_COL[dataset]].max()).date()
                    if dataset is Dataset.CORPORATE_ACTIONS:
                        # the vendor answered through `end`; don't refetch the same window
                        high_water = max(high_water, end)
                    set_watermark(catalog.conn, provider.name, dataset, ticker, high_water, run_id)
                    status, rows = "success", len(df)
                    summary.ok += 1
                    summary.rows += rows

                record_fetch(
                    catalog.conn,
                    ManifestEntry(
                        run_id, provider.name, dataset, ticker, eff_start, end,
                        rows, result.payload_sha256, status, None,
                        started_at, datetime.now(UTC),
                    ),
                )
            except Exception as exc:  # noqa: BLE001 — failure isolation is the point
                logger.exception("Ingestion failed for %s/%s", ticker, dataset)
                summary.failed += 1
                summary.failures.append((ticker, str(dataset), str(exc)))
                record_fetch(
                    catalog.conn,
                    ManifestEntry(
                        run_id, provider.name, dataset, ticker, eff_start, end,
                        0, None, "failed", str(exc), started_at, datetime.now(UTC),
                    ),
                )
    return summary
