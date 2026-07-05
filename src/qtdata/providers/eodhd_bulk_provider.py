"""EODHD daily adapter: bulk EOD (whole market, one call) + per-symbol history.

A1c of the seeking-alpha plan: with the lake widened to ~10k US names (C13),
yfinance is demoted to spot cross-validation and EODHD becomes the primary
daily-ingestion engine. Two access patterns, both normalized to qtdata's
canonical OHLCV frame (as-traded prices — adjust-on-read stays downstream):

- ``fetch_bulk_day(day)``      -> the WHOLE US market for one date, one call
                                  (``/api/eod-bulk-last-day/US``). This is the
                                  daily-drip pattern: ~1 call/day total.
- ``fetch_symbol_history(sym)`` -> full history for one symbol, one call
                                  (``/api/eod/{SYM}.US``). This is the backfill
                                  pattern: ~1 call/ticker, run once.

CONTRACT NOTES (load-bearing):
- ``close`` from both endpoints is the RAW (as-traded) close; ``adjusted_close``
  is vendor-derived and is kept ONLY as a cross-check column in the raw layer —
  the curated lake stores unadjusted prices per invariant #4 (adjust-on-read).
- Corporate actions keep coming from the existing daily pipeline; this module
  does not write splits/dividends.
- QUALITY GATE before trusting at scale (B0 spec): random sample >=100 NASDAQ
  tickers must reconcile with the already-validated yfinance as-traded closes
  (rel tol 0.1%) + split-day spot checks. Run via ``qt reconcile`` machinery.
- The public ``demo`` token only serves AAPL/TSLA/MSFT-class symbols; the bulk
  endpoint requires a paid tier ([VERIFICAR] price before committing B1).
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import requests

from qtdata.config import Settings
from qtdata.models import ProviderNotConfiguredError
from qtdata.providers.base import RateLimiter, retry_transient

logger = logging.getLogger(__name__)

_BULK_BASE = "https://eodhd.com/api/eod-bulk-last-day"
_EOD_BASE = "https://eodhd.com/api/eod"

DAILY_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "volume"]


def _normalize_rows(rows: list[dict], *, ticker: str | None = None) -> pd.DataFrame:
    """Vendor JSON -> canonical daily frame (as-traded OHLCV, date-typed)."""
    if not rows:
        return pd.DataFrame(columns=DAILY_COLUMNS)
    df = pd.DataFrame(rows)
    if ticker is not None:
        df["ticker"] = ticker.upper()
    else:
        # bulk rows carry a `code` column (symbol without the .US suffix)
        df["ticker"] = df["code"].astype(str).str.upper()
    df["date"] = pd.to_datetime(df["date"])
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
    df = df.dropna(subset=["open", "high", "low", "close"])
    keep = DAILY_COLUMNS + (["adjusted_close"] if "adjusted_close" in df.columns else [])
    return df[keep].sort_values(["ticker", "date"]).reset_index(drop=True)


class EODHDBulkProvider:
    """Daily EOD client for the widened lake. Requires ``QT_EODHD_API_KEY``.

    Unlike the intraday free-phase client this one REFUSES to run on the demo
    token for bulk calls: silently ingesting a demo-limited subset into a ~10k
    lake would be a data-integrity failure, not a convenience.
    """

    name = "eodhd_bulk"

    def __init__(self, settings: Settings, *, rate_limit_per_min: int | None = None):
        key = getattr(settings, "eodhd_api_key", None)
        self._token = key.get_secret_value() if key else None
        limit = rate_limit_per_min or settings.eodhd_rate_limit_per_min
        self._limiter = RateLimiter(limit)

    def _require_token(self) -> str:
        if not self._token:
            raise ProviderNotConfiguredError(
                "EODHD bulk requires QT_EODHD_API_KEY (paid tier); the demo token "
                "would silently truncate the universe."
            )
        return self._token

    @retry_transient
    def _get(self, url: str, params: dict) -> list[dict]:
        self._limiter.acquire()
        resp = requests.get(url, params=params, timeout=120)
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, list) else []

    def fetch_bulk_day(self, day: date | None = None, exchange: str = "US") -> pd.DataFrame:
        """Whole-market EOD for one date (defaults to the vendor's last session)."""
        token = self._require_token()
        params: dict = {"api_token": token, "fmt": "json"}
        if day is not None:
            params["date"] = day.isoformat()
        rows = self._get(f"{_BULK_BASE}/{exchange}", params)
        df = _normalize_rows(rows)
        logger.info("EODHD bulk %s %s: %d rows", exchange, day or "last", len(df))
        return df

    def fetch_symbol_history(
        self, ticker: str, start: date | None = None, end: date | None = None,
        exchange_suffix: str = "US",
    ) -> pd.DataFrame:
        """Full (or windowed) daily history for one symbol — the backfill pattern."""
        token = self._require_token()
        params: dict = {"api_token": token, "fmt": "json", "period": "d"}
        if start is not None:
            params["from"] = start.isoformat()
        if end is not None:
            params["to"] = end.isoformat()
        symbol = f"{ticker.upper()}.{exchange_suffix}"
        rows = self._get(f"{_EOD_BASE}/{symbol}", params)
        return _normalize_rows(rows, ticker=ticker)
