"""yfinance adapter.

`auto_adjust=False` is load-bearing: the raw layer stores prices AS TRADED;
adjustments are derived downstream from the corporate-actions table. Splits
and dividends come from the same history call (`actions=True`).

yfinance scrapes Yahoo and breaks periodically — treat it as a prototyping
source and reconcile against a second provider before trusting it.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from qtdata.config import Settings
from qtdata.models import ActionType, Dataset
from qtdata.providers.base import FetchResult, RateLimiter, make_fetch_result, retry_transient

_OHLCV_RENAME = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}


def normalize_history_ohlcv(hist: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Vendor frame (tz-aware index, capitalized columns) -> canonical raw shape."""
    if hist.empty:
        return pd.DataFrame(columns=["ticker", "date", *(_OHLCV_RENAME.values())])
    df = hist.rename(columns=_OHLCV_RENAME).copy()
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df = df[list(_OHLCV_RENAME.values())].dropna(subset=["open", "high", "low", "close"])
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
    df = df.reset_index(names="date")
    df.insert(0, "ticker", ticker.upper())
    return df


def normalize_history_actions(hist: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Extract long-format corporate actions from a history frame with actions=True."""
    cols = ["ticker", "ex_date", "action_type", "value"]
    if hist.empty:
        return pd.DataFrame(columns=cols)
    idx = pd.to_datetime(hist.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    dates = idx.normalize()
    rows = []
    if "Dividends" in hist.columns:
        mask = hist["Dividends"].fillna(0) > 0
        for d, v in zip(dates[mask], hist.loc[mask.to_numpy(), "Dividends"], strict=True):
            rows.append((ticker.upper(), d, ActionType.DIVIDEND.value, float(v)))
    if "Stock Splits" in hist.columns:
        mask = hist["Stock Splits"].fillna(0) > 0
        for d, v in zip(dates[mask], hist.loc[mask.to_numpy(), "Stock Splits"], strict=True):
            rows.append((ticker.upper(), d, ActionType.SPLIT.value, float(v)))
    return pd.DataFrame(rows, columns=cols)


class YFinanceProvider:
    name = "yfinance"

    def __init__(self, settings: Settings):
        self._limiter = RateLimiter(settings.yfinance_rate_limit_per_min)

    def supported_datasets(self) -> frozenset[Dataset]:
        return frozenset({Dataset.OHLCV_DAILY, Dataset.CORPORATE_ACTIONS})

    @retry_transient
    def _history(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        self._limiter.acquire()
        # yfinance treats `end` as exclusive
        return yf.Ticker(ticker).history(
            start=start,
            end=end + timedelta(days=1),
            interval="1d",
            auto_adjust=False,
            actions=True,
        )

    def fetch_ohlcv(self, ticker: str, start: date, end: date) -> FetchResult:
        df = normalize_history_ohlcv(self._history(ticker, start, end), ticker)
        return make_fetch_result(df, self.name, Dataset.OHLCV_DAILY, ticker, start, end)

    def fetch_corporate_actions(self, ticker: str, start: date, end: date) -> FetchResult:
        df = normalize_history_actions(self._history(ticker, start, end), ticker)
        return make_fetch_result(df, self.name, Dataset.CORPORATE_ACTIONS, ticker, start, end)
