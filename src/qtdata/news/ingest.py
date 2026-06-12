"""News ingestion -> immutable raw layer (date-partitioned layout).

Firehose (alpha_vantage_news): one watermark for the whole stream, keyed
(provider, 'news', '_FIREHOSE_') = last fully ingested UTC day. Budget guard:
each day costs up to `av_news_page_limit` of the 25 free requests/day.

yfinance_news: per-ticker harvest of the recent stream; dedupe at curation
makes overlapping pulls harmless.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta
from uuid import uuid4

import pandas as pd

from qtdata import calendars
from qtdata.config import Settings
from qtdata.ingestion.ingest import IngestSummary
from qtdata.ingestion.manifest import ManifestEntry, record_fetch
from qtdata.ingestion.watermarks import get_watermark, set_watermark
from qtdata.models import Dataset
from qtdata.storage import parquet_store
from qtdata.storage.catalog import Catalog

logger = logging.getLogger(__name__)

FIREHOSE_KEY = "_FIREHOSE_"


def _raw_news_path(settings: Settings, provider: str, day: date, run_id: str):
    return (
        settings.raw_dir / f"provider={provider}" / "dataset=news"
        / f"date={day}" / f"{run_id}.parquet"
    )


def ingest_news(
    settings: Settings,
    catalog: Catalog,
    *,
    date_from: date | None = None,
    date_to: date | None = None,
    provider_name: str = "alpha_vantage_news",
    tickers: list[str] | None = None,
) -> IngestSummary:
    run_id = uuid4().hex[:12]
    summary = IngestSummary(run_id=run_id)

    if provider_name == "alpha_vantage_news":
        _ingest_firehose(settings, catalog, run_id, date_from, date_to, summary)
    elif provider_name == "yfinance_news":
        if not tickers:
            raise ValueError("yfinance_news requires --tickers")
        _ingest_yfinance(settings, catalog, run_id, tickers, summary)
    else:
        raise ValueError(f"Unknown news provider {provider_name!r}")
    return summary


def _ingest_firehose(
    settings: Settings,
    catalog: Catalog,
    run_id: str,
    date_from: date | None,
    date_to: date | None,
    summary: IngestSummary,
) -> None:
    from qtdata.providers.alpha_vantage_news import AlphaVantageNewsProvider

    provider = AlphaVantageNewsProvider(settings)
    if date_to is None:
        date_to = calendars.last_completed_session(
            date.today(), settings.default_calendar
        ).date()
    if date_from is None:
        wm = get_watermark(catalog.conn, provider.name, Dataset.NEWS, FIREHOSE_KEY)
        date_from = (wm + timedelta(days=1)) if wm else date_to
    if date_from > date_to:
        summary.skipped += 1
        return

    n_days = (date_to - date_from).days + 1
    budget = n_days * settings.av_news_page_limit
    if budget > 25:
        logger.warning(
            "Firehose plan needs up to %d AV requests (%d days x %d pages) — "
            "free tier allows 25/day; consider a narrower window.",
            budget, n_days, settings.av_news_page_limit,
        )

    day = date_from
    while day <= date_to:
        started_at = datetime.now(UTC)
        try:
            frame, pages = provider.fetch_news_day(day, settings.av_news_page_limit)
            if frame.empty:
                status, rows = "empty", 0
                summary.empty += 1
            else:
                frame = frame.assign(
                    run_id=run_id, ingested_at=pd.Timestamp.now(tz="UTC")
                )
                parquet_store.write_raw(
                    frame, _raw_news_path(settings, provider.name, day, run_id)
                )
                status, rows = "success", len(frame)
                summary.ok += 1
                summary.rows += rows
            # day fully paged (success or genuinely empty) -> advance watermark
            set_watermark(catalog.conn, provider.name, Dataset.NEWS, FIREHOSE_KEY, day, run_id)
            record_fetch(
                catalog.conn,
                ManifestEntry(
                    run_id, provider.name, Dataset.NEWS, FIREHOSE_KEY, day, day,
                    rows, None, status, None, started_at, datetime.now(UTC),
                ),
            )
        except Exception as exc:  # noqa: BLE001 — quota/network: stop, don't burn budget
            logger.exception("Firehose ingestion failed on %s; stopping", day)
            summary.failed += 1
            summary.failures.append((FIREHOSE_KEY, str(Dataset.NEWS), str(exc)))
            record_fetch(
                catalog.conn,
                ManifestEntry(
                    run_id, provider.name, Dataset.NEWS, FIREHOSE_KEY, day, day,
                    0, None, "failed", str(exc), started_at, datetime.now(UTC),
                ),
            )
            return
        day += timedelta(days=1)


def _ingest_yfinance(
    settings: Settings,
    catalog: Catalog,
    run_id: str,
    tickers: list[str],
    summary: IngestSummary,
) -> None:
    from qtdata.providers.yfinance_news import YFinanceNewsProvider

    provider = YFinanceNewsProvider(settings)
    today = date.today()
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        started_at = datetime.now(UTC)
        frame = provider.fetch_news(ticker)
        if frame.empty:
            summary.empty += 1
            status, rows = "empty", 0
        else:
            frames.append(frame)
            summary.ok += 1
            summary.rows += len(frame)
            status, rows = "success", len(frame)
            set_watermark(catalog.conn, provider.name, Dataset.NEWS, ticker, today, run_id)
        record_fetch(
            catalog.conn,
            ManifestEntry(
                run_id, provider.name, Dataset.NEWS, ticker, today, today,
                rows, None, status, None, started_at, datetime.now(UTC),
            ),
        )
    if frames:
        combined = pd.concat(frames, ignore_index=True).assign(
            run_id=run_id, ingested_at=pd.Timestamp.now(tz="UTC")
        )
        parquet_store.write_raw(
            combined, _raw_news_path(settings, provider.name, today, run_id)
        )
