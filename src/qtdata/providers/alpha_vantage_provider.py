"""Alpha Vantage adapter (requires QT_ALPHA_VANTAGE_API_KEY).

Endpoint mapping (free tier as of 2026; TIME_SERIES_DAILY_ADJUSTED is premium):
- OHLCV (unadjusted, which is what our raw layer wants): TIME_SERIES_DAILY
- Splits:    SPLITS
- Dividends: DIVIDENDS

Free tier is ~25 requests/day — keep the universe small or upgrade before
using this as the primary price source. Without a key every fetch raises
ProviderNotConfiguredError; the rest of the pipeline is unaffected.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import requests

from qtdata.config import Settings
from qtdata.models import ActionType, Dataset, ProviderNotConfiguredError
from qtdata.providers.base import FetchResult, RateLimiter, make_fetch_result, retry_transient

BASE_URL = "https://www.alphavantage.co/query"
_TIMEOUT = 30


class AlphaVantageProvider:
    name = "alpha_vantage"

    def __init__(self, settings: Settings):
        key = settings.alpha_vantage_api_key
        self._key = key.get_secret_value() if key else None
        self._limiter = RateLimiter(settings.alpha_vantage_rate_limit_per_min)

    def supported_datasets(self) -> frozenset[Dataset]:
        return frozenset({Dataset.OHLCV_DAILY, Dataset.CORPORATE_ACTIONS})

    def _require_key(self) -> None:
        if not self._key:
            raise ProviderNotConfiguredError(
                "Alpha Vantage requires QT_ALPHA_VANTAGE_API_KEY (set it in .env)."
            )

    @retry_transient
    def _get(self, **params: str) -> dict:
        self._limiter.acquire()
        resp = requests.get(
            BASE_URL, params={**params, "apikey": self._key}, timeout=_TIMEOUT
        )
        resp.raise_for_status()
        payload = resp.json()
        for key in ("Error Message", "Information", "Note"):
            if key in payload:
                raise RuntimeError(f"Alpha Vantage refused the request: {payload[key]}")
        return payload

    def fetch_ohlcv(self, ticker: str, start: date, end: date) -> FetchResult:
        self._require_key()
        payload = self._get(
            function="TIME_SERIES_DAILY", symbol=ticker, outputsize="full", datatype="json"
        )
        series = payload.get("Time Series (Daily)", {})
        rows = [
            {
                "ticker": ticker.upper(),
                "date": pd.Timestamp(d),
                "open": float(v["1. open"]),
                "high": float(v["2. high"]),
                "low": float(v["3. low"]),
                "close": float(v["4. close"]),
                "volume": int(v["5. volume"]),
            }
            for d, v in series.items()
        ]
        df = pd.DataFrame(rows)
        if not df.empty:
            s, e = pd.Timestamp(start), pd.Timestamp(end)
            df = (
                df[(df["date"] >= s) & (df["date"] <= e)]
                .sort_values("date")
                .reset_index(drop=True)
            )
        return make_fetch_result(df, self.name, Dataset.OHLCV_DAILY, ticker, start, end)

    def fetch_corporate_actions(self, ticker: str, start: date, end: date) -> FetchResult:
        self._require_key()
        rows: list[dict] = []

        splits = self._get(function="SPLITS", symbol=ticker).get("data", [])
        for item in splits:
            rows.append(
                {
                    "ticker": ticker.upper(),
                    "ex_date": pd.Timestamp(item["effective_date"]),
                    "action_type": ActionType.SPLIT.value,
                    "value": float(item["split_factor"]),
                }
            )
        dividends = self._get(function="DIVIDENDS", symbol=ticker).get("data", [])
        for item in dividends:
            rows.append(
                {
                    "ticker": ticker.upper(),
                    "ex_date": pd.Timestamp(item["ex_dividend_date"]),
                    "action_type": ActionType.DIVIDEND.value,
                    "value": float(item["amount"]),
                }
            )
        df = pd.DataFrame(rows, columns=["ticker", "ex_date", "action_type", "value"])
        if not df.empty:
            s, e = pd.Timestamp(start), pd.Timestamp(end)
            df = (
                df[(df["ex_date"] >= s) & (df["ex_date"] <= e)]
                .sort_values("ex_date")
                .reset_index(drop=True)
            )
        return make_fetch_result(df, self.name, Dataset.CORPORATE_ACTIONS, ticker, start, end)
