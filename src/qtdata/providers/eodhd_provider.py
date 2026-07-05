"""EODHD intraday adapter (FASE GRATIS — validación del proyecto seeking-alpha).

Adds an INTRADAY (1-minute) data source to qtdata, the dimension the daily-only
lake lacks. Chosen in seeking-alpha/plan/decisions/A1-datos-pit.md (best
quality/price for a multi-market AI/semis basket: US 1-min since 2004 + global
coverage + delisted API, 29.99 EUR/mo, free tier 20 calls/day for validation).

CONTRACT NOTES (load-bearing):
- EODHD intraday returns UTC timestamps (``gmtoffset:0``) — exactly what the
  no-look-ahead intraday discipline needs. We keep a tz-aware UTC ``ts`` column;
  NEVER store a naive local timestamp (that is how look-ahead creeps in).
- The vendor appends a trailing SNAPSHOT bar with ``volume:null`` and flat OHLC
  (the not-yet-closed/last bar). We DROP rows with null volume — flag-never-mutate
  applies to real bars, not to a vendor placeholder that was never a real print.
- Prices are as-traded intraday; split adjustment is DERIVED downstream by the
  same ``curation/adjustments.py`` machinery as daily (adjust-on-read), so we do
  NOT adjust here. Corporate actions still come from the daily provider.
- US symbols use the ``.US`` suffix. Foreign suffixes differ from yfinance
  (EODHD: Korea KOSPI = ``.KO`` (not ``.KS``), Hong Kong = ``.HK``). The mapping
  for the Tier-2 foreign names is [VERIFICAR] in the paid phase; this free-phase
  module is validated on US (Tier 1) only.

This module is intentionally self-contained for the FREE PHASE: it proves the
data ingests and normalizes to qtdata's conventions. Wiring it into the medallion
(a new ``Dataset.OHLCV_INTRADAY``, parquet partition by ticker+month, a DuckDB
``ohlcv_intraday`` view) is task B1 of the plan, applied by the human.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from qtdata.config import Settings
from qtdata.providers.base import RateLimiter, retry_transient

_NY = ZoneInfo("America/New_York")
_BASE = "https://eodhd.com/api/intraday"
# EODHD caps each 1-min intraday request at 120 days. Use 119 to stay safely under
# the inclusive boundary when we add a day at the window edge.
_MAX_INTRADAY_WINDOW_DAYS = 119
# US Regular Trading Hours in local ET (handles DST automatically via zoneinfo).
_RTH_OPEN = (9, 30)
_RTH_CLOSE = (16, 0)

INTRADAY_COLUMNS = ["ticker", "ts", "open", "high", "low", "close", "volume", "segment"]


def _segment(ts_utc: pd.Timestamp) -> str:
    """RTH vs extended-hours, decided in ET so DST is handled correctly."""
    et = ts_utc.tz_convert(_NY)
    hm = (et.hour, et.minute)
    if _RTH_OPEN <= hm < _RTH_CLOSE:
        return "rth"
    return "ext"


def normalize_intraday(rows: list[dict], ticker: str) -> pd.DataFrame:
    """Vendor JSON -> canonical intraday frame (tz-aware UTC ts, as-traded OHLCV).

    Drops the trailing null-volume snapshot bar; keeps only real prints.
    """
    if not rows:
        return pd.DataFrame(columns=INTRADAY_COLUMNS)
    df = pd.DataFrame(rows)
    # UTC from the unix epoch column (gmtoffset is 0 but we never trust the string)
    df["ts"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    # Drop vendor placeholder / not-yet-closed bars (volume null) — never a real print.
    df = df[df["volume"].notna()].copy()
    df = df.dropna(subset=["open", "high", "low", "close"])
    if df.empty:
        return pd.DataFrame(columns=INTRADAY_COLUMNS)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
    df["ticker"] = ticker.upper()
    df["segment"] = [_segment(ts) for ts in df["ts"]]
    df = df[INTRADAY_COLUMNS].sort_values("ts").reset_index(drop=True)
    return df


class EODHDProvider:
    """Minimal EODHD intraday client for the free-phase validation.

    Reads ``QT_EODHD_API_KEY``; falls back to the public ``demo`` token (works for
    a handful of US symbols like AAPL/TSLA/MSFT — enough to validate the pipeline
    shape without an account).
    """

    name = "eodhd"

    def __init__(self, settings: Settings, *, rate_limit_per_min: int = 60):
        key = getattr(settings, "eodhd_api_key", None)
        self._token = key.get_secret_value() if key else "demo"
        self._limiter = RateLimiter(rate_limit_per_min)

    @retry_transient
    def _get(self, symbol: str, interval: str, start: date, end: date) -> list[dict]:
        self._limiter.acquire()
        params = {
            "interval": interval,
            "api_token": self._token,
            "fmt": "json",
            "from": int(datetime(start.year, start.month, start.day, tzinfo=UTC).timestamp()),
            "to": int(
                (datetime(end.year, end.month, end.day, tzinfo=UTC) + timedelta(days=1)).timestamp()
            ),
        }
        resp = requests.get(f"{_BASE}/{symbol}", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []

    def fetch_intraday_1m(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """1-minute bars for a US ticker in [start, end]. Returns canonical frame.

        EODHD caps each 1-min request at 120 days ("Max period length is 120 days"),
        so a multi-year pull MUST be chunked. We request <=120-day windows and
        concatenate; chunk boundaries can overlap a bar, so we dedup on ts
        (first-wins) after the merge.
        """
        symbol = ticker if "." in ticker else f"{ticker.upper()}.US"
        frames: list[pd.DataFrame] = []
        win_start = start
        while win_start <= end:
            win_end = min(win_start + timedelta(days=_MAX_INTRADAY_WINDOW_DAYS), end)
            rows = self._get(symbol, "1m", win_start, win_end)
            frames.append(normalize_intraday(rows, ticker))
            win_start = win_end + timedelta(days=1)
        if not frames:
            return pd.DataFrame(columns=INTRADAY_COLUMNS)
        out = pd.concat(frames, ignore_index=True)
        out = out.drop_duplicates(subset=["ts"], keep="first").sort_values("ts")
        return out.reset_index(drop=True)
