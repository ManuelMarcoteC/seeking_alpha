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

# Per-market attribution: a foreign ticker's news must be bucketed against its
# OWN exchange session, in its OWN timezone, with a cutoff relative to its OWN
# close. Keyed by ticker suffix; the default (no suffix = NASDAQ/US) preserves
# the original NY_TZ / XNYS / 15:30 behaviour exactly, so the US path is unchanged.
# (tz, calendar, cutoff_local) — cutoff ~30 min before each market's local close.
_MARKET_BY_SUFFIX: dict[str, tuple[str, str, str]] = {
    ".HK": ("Asia/Hong_Kong", "XHKG", "15:30"),  # SEHK closes 16:00 HKT
    ".KS": ("Asia/Seoul", "XKRX", "15:00"),       # KOSPI closes 15:30 KST
    ".KQ": ("Asia/Seoul", "XKRX", "15:00"),       # KOSDAQ closes 15:30 KST
}
_DEFAULT_MARKET: tuple[str, str, str] = (NY_TZ, "XNYS", "15:30")


def market_for_ticker(ticker: str) -> tuple[str, str, str]:
    """Resolve (tz, calendar, cutoff_local) from a ticker's market suffix.

    No recognized suffix -> the US default (NY_TZ, XNYS, 15:30), so every NASDAQ
    ticker keeps its exact prior attribution. Foreign suffixes (.HK/.KS/.KQ) get
    their own market's timezone, calendar, and local cutoff.
    """
    up = ticker.upper()
    for suffix, market in _MARKET_BY_SUFFIX.items():
        if up.endswith(suffix):
            return market
    return _DEFAULT_MARKET


def _parse_cutoff(cutoff_local: str) -> time:
    hour, minute = cutoff_local.split(":")
    return time(int(hour), int(minute))


def trading_day_for(
    effective_ts_utc: pd.Timestamp,
    cutoff_local: str = "15:30",
    calendar: str = "XNYS",
    tz: str = NY_TZ,
) -> pd.Timestamp:
    """Map an effective timestamp to the first session that could trade on it.

    `tz` is the market-local timezone the cutoff is expressed in. Defaults to
    NY_TZ so existing US callers are unaffected; foreign markets pass their own.
    """
    local = effective_ts_utc.tz_convert(tz)
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
    rows["effective_ts"] = effective

    # Per-market attribution: resolve (tz, calendar, cutoff) from each ticker's
    # suffix, then map its effective_ts to its OWN market's session. NASDAQ
    # tickers fall through to the US default, so their attribution is unchanged.
    # The US-configured cutoff (settings) still overrides the US default cutoff.
    us_cutoff = settings.news_cutoff_local
    cache: dict[tuple[str, pd.Timestamp], pd.Timestamp] = {}

    def _attribute(ticker: str, ts: pd.Timestamp) -> pd.Timestamp:
        key = (ticker.upper(), ts)
        hit = cache.get(key)
        if hit is not None:
            return hit
        tz, calendar, cutoff = market_for_ticker(ticker)
        if (tz, calendar) == _DEFAULT_MARKET[:2]:
            cutoff = us_cutoff  # honour the US-configured cutoff for NASDAQ
        day = trading_day_for(ts, cutoff, calendar, tz)
        cache[key] = day
        return day

    rows["date"] = [
        _attribute(t, ts) for t, ts in zip(rows["ticker"], effective, strict=True)
    ]

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
