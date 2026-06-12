"""Deterministic synthetic market-data provider.

Generates a seeded geometric random walk per ticker over real exchange sessions,
with injectable events (gaps, splits, dividends, stale runs, zero-volume runs,
missing sessions). Used by the test suite and for keyless end-to-end demos
(`qt ingest --provider synthetic`).

The full path is always generated from a fixed BASE_START and sliced, so two
fetches over different windows agree on their overlap — this is what makes
incremental ingestion testable offline.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from qtdata import calendars
from qtdata.models import ActionType, Dataset
from qtdata.providers.base import FetchResult, make_fetch_result

BASE_START = pd.Timestamp("2015-01-02")
BASE_END = pd.Timestamp("2030-12-31")

_PRICE_COLS = ["open", "high", "low", "close"]


@dataclass(frozen=True)
class Gap:
    """One-day jump of `pct` at `on` (path continues from the new level)."""

    on: date
    pct: float


@dataclass(frozen=True)
class SplitEvent:
    ex_date: date
    ratio: float  # 4.0 means a 4:1 split (price quarters at ex-date)


@dataclass(frozen=True)
class DividendEvent:
    ex_date: date
    amount: float  # cash per share; price drops by ~amount at ex-date


@dataclass(frozen=True)
class StaleRun:
    start: date
    length: int


@dataclass(frozen=True)
class ZeroVolumeRun:
    start: date
    length: int


@dataclass(frozen=True)
class MissingSessions:
    dates: tuple[date, ...] = field(default_factory=tuple)


Event = Gap | SplitEvent | DividendEvent | StaleRun | ZeroVolumeRun | MissingSessions


def _ticker_seed(ticker: str, base_seed: int) -> int:
    return (zlib.crc32(ticker.encode("utf-8")) ^ base_seed) & 0xFFFFFFFF


class SyntheticProvider:
    name = "synthetic"

    def __init__(
        self,
        seed: int = 42,
        events: dict[str, list[Event]] | None = None,
        calendar: str = "XNYS",
    ):
        self.seed = seed
        self.events = events or {}
        self.calendar = calendar

    def supported_datasets(self) -> frozenset[Dataset]:
        return frozenset({Dataset.OHLCV_DAILY, Dataset.CORPORATE_ACTIONS})

    def _full_path(self, ticker: str) -> pd.DataFrame:
        sessions = calendars.sessions_between(BASE_START, BASE_END, self.calendar)
        n = len(sessions)
        rng = np.random.default_rng(_ticker_seed(ticker, self.seed))
        rets = rng.normal(0.0004, 0.015, size=n)
        intraday = np.abs(rng.normal(0.0, 0.006, size=(n, 2)))
        open_noise = rng.normal(0.0, 0.004, size=n)
        volume = rng.lognormal(15.0, 0.5, size=n)

        close = 100.0 * np.exp(np.cumsum(rets))
        open_ = np.empty(n)
        open_[0] = 100.0
        open_[1:] = close[:-1] * (1.0 + open_noise[1:])
        high = np.maximum(open_, close) * (1.0 + intraday[:, 0])
        low = np.minimum(open_, close) * (1.0 - intraday[:, 1])

        df = pd.DataFrame(
            {
                "ticker": ticker,
                "date": sessions,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume.astype("int64"),
            }
        )
        for ev in self.events.get(ticker, []):
            df = self._apply_event(df, ev)
        return df.reset_index(drop=True)

    def _apply_event(self, df: pd.DataFrame, ev: Event) -> pd.DataFrame:
        dates = df["date"]
        if isinstance(ev, Gap):
            mask = dates >= pd.Timestamp(ev.on)
            df.loc[mask, _PRICE_COLS] *= 1.0 + ev.pct
        elif isinstance(ev, SplitEvent):
            mask = dates >= pd.Timestamp(ev.ex_date)
            df.loc[mask, _PRICE_COLS] /= ev.ratio
            df.loc[mask, "volume"] = (df.loc[mask, "volume"] * ev.ratio).astype("int64")
        elif isinstance(ev, DividendEvent):
            pos = int(dates.searchsorted(pd.Timestamp(ev.ex_date)))
            if pos > 0:
                prev_close = float(df["close"].iloc[pos - 1])
                factor = max(1.0 - ev.amount / prev_close, 0.01)
                df.loc[dates >= pd.Timestamp(ev.ex_date), _PRICE_COLS] *= factor
        elif isinstance(ev, StaleRun):
            pos = int(dates.searchsorted(pd.Timestamp(ev.start)))
            stop = min(pos + ev.length, len(df))
            level = float(df["close"].iloc[pos]) if pos < len(df) else None
            if level is not None:
                df.loc[df.index[pos:stop], _PRICE_COLS] = level
        elif isinstance(ev, ZeroVolumeRun):
            pos = int(dates.searchsorted(pd.Timestamp(ev.start)))
            stop = min(pos + ev.length, len(df))
            df.loc[df.index[pos:stop], "volume"] = 0
        elif isinstance(ev, MissingSessions):
            drop = {pd.Timestamp(d) for d in ev.dates}
            df = df[~df["date"].isin(drop)]
        return df

    def fetch_ohlcv(self, ticker: str, start: date, end: date) -> FetchResult:
        path = self._full_path(ticker)
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        df = path[(path["date"] >= s) & (path["date"] <= e)].reset_index(drop=True)
        return make_fetch_result(df, self.name, Dataset.OHLCV_DAILY, ticker, start, end)

    def fetch_batch(
        self, tickers: Sequence[str], start: date, end: date
    ) -> dict[str, dict[Dataset, FetchResult]]:
        """Batch capability mirror — lets the batched ingest path run fully offline."""
        out: dict[str, dict[Dataset, FetchResult]] = {}
        for ticker in tickers:
            ohlcv = self.fetch_ohlcv(ticker, start, end)
            actions = self.fetch_corporate_actions(ticker, start, end)
            if ohlcv.df.empty and actions.df.empty:
                continue
            out[ticker] = {
                Dataset.OHLCV_DAILY: ohlcv,
                Dataset.CORPORATE_ACTIONS: actions,
            }
        return out

    def fetch_corporate_actions(self, ticker: str, start: date, end: date) -> FetchResult:
        rows = []
        for ev in self.events.get(ticker, []):
            if isinstance(ev, SplitEvent):
                rows.append((ticker, pd.Timestamp(ev.ex_date), ActionType.SPLIT.value, ev.ratio))
            elif isinstance(ev, DividendEvent):
                rows.append(
                    (ticker, pd.Timestamp(ev.ex_date), ActionType.DIVIDEND.value, ev.amount)
                )
        df = pd.DataFrame(rows, columns=["ticker", "ex_date", "action_type", "value"])
        if not df.empty:
            s, e = pd.Timestamp(start), pd.Timestamp(end)
            df = df[(df["ex_date"] >= s) & (df["ex_date"] <= e)].reset_index(drop=True)
        return make_fetch_result(df, self.name, Dataset.CORPORATE_ACTIONS, ticker, start, end)
