"""Daily sentiment factor (gold): news rows -> sentiment_daily per (ticker, session).

PIT rules that make the factor backtestable:
- effective_ts = max(published_at, ingested_at): you cannot trade on an article
  before you possessed it (backfilled archives get attributed to ingestion day).
- Cutoff: articles after 15:30 ET (configurable) or on non-sessions map to the
  NEXT trading session — a close-execution backtest only sees what existed
  30 minutes before its fill.
- Aggregation: relevance-weighted mean with a relevance floor; news volume
  (log(1+N)) kept as a companion feature. Decay lives in the derived view
  `sentiment_daily_decayed`, never materialized.
"""

from __future__ import annotations

import logging
from datetime import date, time
from uuid import uuid4

import numpy as np
import pandas as pd

from qtdata import calendars
from qtdata.config import Settings
from qtdata.models import SENTIMENT_DAILY_KEY
from qtdata.storage import parquet_store
from qtdata.storage.catalog import Catalog

logger = logging.getLogger(__name__)

NY_TZ = "America/New_York"


def _parse_cutoff(cutoff_local: str) -> time:
    hour, minute = cutoff_local.split(":")
    return time(int(hour), int(minute))


def trading_day_for(
    effective_ts_utc: pd.Timestamp,
    cutoff_local: str = "15:30",
    calendar: str = "XNYS",
) -> pd.Timestamp:
    """Map an effective timestamp to the first session that could trade on it."""
    local = effective_ts_utc.tz_convert(NY_TZ)
    local_date = local.date()
    cutoff = _parse_cutoff(cutoff_local)
    if calendars.is_session(local_date, calendar) and local.time() <= cutoff:
        return pd.Timestamp(local_date)
    return calendars.next_session(local_date, calendar)


def build_sentiment_daily(
    settings: Settings, catalog: Catalog, since: date | None = None
) -> int:
    """(Re)build the daily factor; idempotent upsert per (ticker, date). Returns rows."""
    rows = parquet_store.read(settings.curated_dir / "news_ticker_sentiment")
    if rows.empty:
        return 0

    rows = rows.copy()
    rows["published_at"] = pd.to_datetime(rows["published_at"], utc=True)
    rows["ingested_at"] = pd.to_datetime(rows["ingested_at"], utc=True)
    effective = rows[["published_at", "ingested_at"]].max(axis=1)

    cutoff = settings.news_cutoff_local
    calendar = settings.default_calendar
    unique_ts = effective.drop_duplicates()
    mapping = {ts: trading_day_for(ts, cutoff, calendar) for ts in unique_ts}
    rows["date"] = effective.map(mapping)

    # relevance floor; null relevance (yfinance harvester) weights 1.0 — documented
    weight = rows["relevance"].astype(float).fillna(1.0)
    rows = rows[weight >= settings.news_relevance_floor].copy()
    if rows.empty:
        return 0
    rows["weight"] = weight[rows.index]

    if since is not None:
        rows = rows[rows["date"] >= pd.Timestamp(since)]
        if rows.empty:
            return 0

    def _aggregate(group: pd.DataFrame) -> pd.Series:
        out: dict[str, float] = {}
        for src_col, dst in (("score_av", "sent_av"), ("score_finbert", "sent_finbert")):
            scored = group.dropna(subset=[src_col])
            if scored.empty:
                out[dst] = np.nan
            else:
                out[dst] = float(
                    np.average(scored[src_col], weights=scored["weight"])
                )
        n = group["article_id"].nunique()
        out["n_articles"] = float(n)
        out["log_n_articles"] = float(np.log1p(n))
        out["rel_sum"] = float(group["weight"].sum())
        return pd.Series(out)

    daily = (
        rows.groupby(["ticker", "date"], as_index=False)
        .apply(_aggregate, include_groups=False)
        .reset_index(drop=True)
    )
    daily["n_articles"] = daily["n_articles"].astype("int64")
    daily["built_at"] = pd.Timestamp.now(tz="UTC")
    daily["year"] = pd.to_datetime(daily["date"]).dt.year

    res = parquet_store.upsert(
        daily, settings.curated_dir / "sentiment_daily", SENTIMENT_DAILY_KEY,
        partition_col="year",
    )
    catalog.refresh_views()
    logger.info("sentiment_daily rebuilt: %d (ticker, day) rows (run %s)",
                res.rows_written, uuid4().hex[:8])
    return res.rows_written
