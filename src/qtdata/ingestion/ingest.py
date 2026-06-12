"""Incremental ingestion orchestrator: provider -> immutable raw layer.

Per (ticker, dataset): resolve the effective start from the watermark (plus one
trading session), fetch, append the payload to the raw layer, record the
manifest row, and advance the watermark. One failing ticker never aborts the
run — failures are isolated and logged.

Providers that implement BatchFetchProtocol (yfinance, synthetic) are driven
through a batched path: tickers with identical effective starts are grouped and
fetched in one provider call (yfinance: chunked yf.download serving OHLCV and
corporate actions together). Manifest rows and watermarks stay per (ticker,
dataset); a failing batch falls back to the per-ticker path for that group.
"""

from __future__ import annotations

import logging
from collections import defaultdict
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
from qtdata.providers.base import (
    BatchFetchProtocol,
    DataProviderProtocol,
    FetchResult,
    make_fetch_result,
)
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


def _effective_start(
    catalog: Catalog,
    settings: Settings,
    provider_name: str,
    dataset: Dataset,
    ticker: str,
    start: date | None,
    end: date,
    full_refresh: bool,
) -> date | None:
    """Resolve the fetch start for one (ticker, dataset); None means up to date."""
    if start is not None:
        return start
    if not full_refresh:
        wm = get_watermark(catalog.conn, provider_name, dataset, ticker)
        if wm is not None:
            try:
                nxt = calendars.next_session(wm, settings.default_calendar).date()
            except Exception:
                return None
            if nxt > end:
                return None
            return nxt
    return settings.default_start_date


def _record_result(
    settings: Settings,
    catalog: Catalog,
    provider_name: str,
    dataset: Dataset,
    ticker: str,
    eff_start: date,
    end: date,
    run_id: str,
    result: FetchResult,
    started_at: datetime,
    summary: IngestSummary,
) -> None:
    """Bookkeeping for one successful fetch: raw write, watermark, manifest."""
    df = result.df
    if df.empty:
        status, rows = "empty", 0
        summary.empty += 1
    else:
        df = df.assign(
            source=provider_name,
            run_id=run_id,
            ingested_at=pd.Timestamp.now(tz="UTC"),
        )
        parquet_store.write_raw(df, _raw_path(settings, provider_name, dataset, ticker, run_id))
        high_water = pd.Timestamp(df[_DATE_COL[dataset]].max()).date()
        if dataset is Dataset.CORPORATE_ACTIONS:
            # the vendor answered through `end`; don't refetch the same window
            high_water = max(high_water, end)
        set_watermark(catalog.conn, provider_name, dataset, ticker, high_water, run_id)
        status, rows = "success", len(df)
        summary.ok += 1
        summary.rows += rows

    record_fetch(
        catalog.conn,
        ManifestEntry(
            run_id, provider_name, dataset, ticker, eff_start, end,
            rows, result.payload_sha256, status, None,
            started_at, datetime.now(UTC),
        ),
    )


def _record_failure(
    catalog: Catalog,
    provider_name: str,
    dataset: Dataset,
    ticker: str,
    eff_start: date | None,
    end: date,
    run_id: str,
    exc: Exception,
    started_at: datetime,
    summary: IngestSummary,
) -> None:
    summary.failed += 1
    summary.failures.append((ticker, str(dataset), str(exc)))
    record_fetch(
        catalog.conn,
        ManifestEntry(
            run_id, provider_name, dataset, ticker, eff_start, end,
            0, None, "failed", str(exc), started_at, datetime.now(UTC),
        ),
    )


def _slice_from(df: pd.DataFrame, dataset: Dataset, eff_start: date) -> pd.DataFrame:
    """Trim a batch payload to this ticker's own effective start (truthful manifest)."""
    if df.empty:
        return df
    col = _DATE_COL[dataset]
    return df[df[col] >= pd.Timestamp(eff_start)].reset_index(drop=True)


def _ingest_per_ticker(
    settings: Settings,
    catalog: Catalog,
    provider: DataProviderProtocol,
    plan: dict[str, dict[Dataset, date]],
    end: date,
    run_id: str,
    summary: IngestSummary,
) -> None:
    for ticker, per_dataset in plan.items():
        for dataset, eff_start in per_dataset.items():
            started_at = datetime.now(UTC)
            try:
                if dataset is Dataset.OHLCV_DAILY:
                    result = provider.fetch_ohlcv(ticker, eff_start, end)
                else:
                    result = provider.fetch_corporate_actions(ticker, eff_start, end)
                _record_result(
                    settings, catalog, provider.name, dataset, ticker,
                    eff_start, end, run_id, result, started_at, summary,
                )
            except Exception as exc:  # noqa: BLE001 — failure isolation is the point
                logger.exception("Ingestion failed for %s/%s", ticker, dataset)
                _record_failure(
                    catalog, provider.name, dataset, ticker,
                    eff_start, end, run_id, exc, started_at, summary,
                )


def _ingest_batched(
    settings: Settings,
    catalog: Catalog,
    provider: DataProviderProtocol,
    plan: dict[str, dict[Dataset, date]],
    end: date,
    run_id: str,
    summary: IngestSummary,
) -> None:
    """Group tickers by identical effective-start signature and batch-fetch each group."""
    groups: dict[tuple, list[str]] = defaultdict(list)
    for ticker, per_dataset in plan.items():
        signature = tuple(sorted((str(ds), es) for ds, es in per_dataset.items()))
        groups[signature].append(ticker)

    for signature, tickers in groups.items():
        per_dataset = plan[tickers[0]]
        batch_start = min(per_dataset.values())
        started_at = datetime.now(UTC)
        try:
            results = provider.fetch_batch(tickers, batch_start, end)
        except Exception:
            logger.exception(
                "Batch fetch failed for %d tickers; falling back to per-ticker", len(tickers)
            )
            _ingest_per_ticker(
                settings, catalog, provider,
                {t: plan[t] for t in tickers}, end, run_id, summary,
            )
            continue

        empty_result_cache: dict[Dataset, FetchResult] = {}
        for ticker in tickers:
            ticker_results = results.get(ticker, {})
            for dataset, eff_start in per_dataset.items():
                result = ticker_results.get(dataset)
                if result is None:
                    if dataset not in empty_result_cache:
                        empty_result_cache[dataset] = make_fetch_result(
                            pd.DataFrame(), provider.name, dataset, ticker, eff_start, end
                        )
                    result = empty_result_cache[dataset]
                else:
                    sliced = _slice_from(result.df, dataset, eff_start)
                    if len(sliced) != len(result.df):
                        result = make_fetch_result(
                            sliced, provider.name, dataset, ticker, eff_start, end
                        )
                try:
                    _record_result(
                        settings, catalog, provider.name, dataset, ticker,
                        eff_start, end, run_id, result, started_at, summary,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Bookkeeping failed for %s/%s", ticker, dataset)
                    _record_failure(
                        catalog, provider.name, dataset, ticker,
                        eff_start, end, run_id, exc, started_at, summary,
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

    # planning pass: effective start per (ticker, dataset); skips counted here
    plan: dict[str, dict[Dataset, date]] = {}
    for ticker in tickers:
        per_dataset: dict[Dataset, date] = {}
        for dataset in datasets:
            if dataset not in provider.supported_datasets():
                summary.skipped += 1
                continue
            eff = _effective_start(
                catalog, settings, provider.name, dataset, ticker, start, end, full_refresh
            )
            if eff is None:
                summary.skipped += 1
                continue
            per_dataset[dataset] = eff
        if per_dataset:
            plan[ticker] = per_dataset

    if not plan:
        return summary

    if isinstance(provider, BatchFetchProtocol) and len(plan) > 1:
        _ingest_batched(settings, catalog, provider, plan, end, run_id, summary)
    else:
        _ingest_per_ticker(settings, catalog, provider, plan, end, run_id, summary)
    return summary
