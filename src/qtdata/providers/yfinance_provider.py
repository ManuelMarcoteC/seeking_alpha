"""yfinance adapter.

The raw layer's contract is AS TRADED: prices exactly as printed on the tape
each day, with split discontinuities intact. Adjustments are derived downstream
from the corporate-actions table (CRSP-style, ``ohlcv_daily_adj``).

CAVEAT (load-bearing): ``auto_adjust=False`` only governs *dividends*. yfinance
STILL returns split-back-adjusted Close/OHLC and split-inflated Volume — verified
live (AAPL 2020-08-27 prints 125.01, the real as-traded close was 500.04; volume
155.5M vs the real 38.9M, both off by the 4:1 ratio). Storing that vendor frame
verbatim violates the as-traded contract and makes the downstream split factor
apply the split a SECOND time. So we invert the vendor's split back-adjustment
here, in the provider, via ``reconstruct_as_traded`` — the single place that
touches the wire — keeping the raw layer honest and every downstream invariant
(flag-never-mutate, adjusted-on-read) intact. Dividends are left untouched.

yfinance scrapes Yahoo and breaks periodically — treat it as a prototyping
source and reconcile against a second provider before trusting it.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from qtdata.config import Settings
from qtdata.curation.adjustments import _dedup_splits
from qtdata.models import ActionType, Dataset
from qtdata.providers.base import FetchResult, RateLimiter, make_fetch_result, retry_transient

_OHLCV_RENAME = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}


def reconstruct_as_traded(ohlcv: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Invert yfinance's split back-adjustment to recover AS-TRADED OHLCV.

    yfinance returns Close/OHLC already divided by every split that occurred on
    or after each row's date, and Volume multiplied by the same factor. To undo
    it we compute, per row, ``future_mult = product of split ratios with ex_date
    STRICTLY AFTER that date`` and multiply OHLC by it / divide Volume by it.

    The split set is deduplicated with the SAME ``_dedup_splits`` guard the
    downstream factor computation uses, so this un-adjust is the EXACT inverse of
    ``adjustments.compute_adjustment_factors``' ``split_factor``. That lock-step
    is what guarantees ``ohlcv_daily_adj`` re-derives the continuous vendor series
    even for vendor-duplicated splits (e.g. Samsung's 50:1 emitted twice → a
    single 50x here and a single 1/50 downstream cancel; a naive 2500x would not).

    Exactness requires every split the vendor applied to be present in ``actions``
    — true whenever the fetch window extends to the present, which holds for
    watermark-forward incremental ingest and full backfills (the only paths that
    write the lake). A partial historical window ending *before* an already-past
    split is the one unsupported case; ingest never issues one.

    Single-ticker frame in, single-ticker frame out (matches ``_dedup_splits``'
    per-ticker assumption and the per-ticker call sites below).
    """
    if ohlcv.empty or actions is None or actions.empty:
        return ohlcv
    splits = actions[actions["action_type"] == ActionType.SPLIT.value]
    splits = _dedup_splits(splits)
    if splits.empty:
        return ohlcv

    ex = pd.to_datetime(splits["ex_date"]).to_numpy()
    ratios = splits["value"].to_numpy(dtype=float)
    order = np.argsort(ex)
    ex_sorted, r_sorted = ex[order], ratios[order]
    # suffix[i] = prod(r_sorted[i:]); suffix[len] = 1  (product of FUTURE splits)
    suffix = np.ones(len(r_sorted) + 1)
    suffix[:-1] = np.cumprod(r_sorted[::-1])[::-1]

    df = ohlcv.copy()
    dates = pd.to_datetime(df["date"]).to_numpy()
    # side="right": a split on its own ex-date is already in the printed price
    # (the row is post-split), so it must NOT be un-adjusted — mirrors the
    # downstream split_factor convention (factor == 1 on/after the ex-date).
    future_mult = suffix[np.searchsorted(ex_sorted, dates, side="right")]
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].to_numpy(dtype=float) * future_mult
    df["volume"] = np.round(df["volume"].to_numpy(dtype=float) / future_mult).astype("int64")
    return df


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
        self._batch_size = max(settings.yfinance_batch_size, 0)
        self._batch_threads = settings.yfinance_batch_threads

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
        frame = self._history(ticker, start, end)
        ohlcv = normalize_history_ohlcv(frame, ticker)
        actions = normalize_history_actions(frame, ticker)
        ohlcv = reconstruct_as_traded(ohlcv, actions)
        return make_fetch_result(ohlcv, self.name, Dataset.OHLCV_DAILY, ticker, start, end)

    def fetch_corporate_actions(self, ticker: str, start: date, end: date) -> FetchResult:
        df = normalize_history_actions(self._history(ticker, start, end), ticker)
        return make_fetch_result(df, self.name, Dataset.CORPORATE_ACTIONS, ticker, start, end)

    # -- batch path (one yf.download serves OHLCV + corporate actions) ---------
    @retry_transient
    def _download_chunk(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        self._limiter.acquire()
        return yf.download(
            tickers,
            start=start,
            end=end + timedelta(days=1),
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            actions=True,
            threads=self._batch_threads,
            progress=False,
        )

    def fetch_batch(
        self, tickers: Sequence[str], start: date, end: date
    ) -> dict[str, dict[Dataset, FetchResult]]:
        size = self._batch_size if self._batch_size > 1 else len(tickers)
        out: dict[str, dict[Dataset, FetchResult]] = {}
        symbols = [t.upper() for t in tickers]
        for i in range(0, len(symbols), size):
            chunk = symbols[i : i + size]
            frame = self._download_chunk(chunk, start, end)
            if frame is None or frame.empty:
                continue
            for ticker in chunk:
                if isinstance(frame.columns, pd.MultiIndex):
                    if ticker not in frame.columns.get_level_values(0):
                        continue
                    sub = frame[ticker]
                else:  # single-ticker chunk returns flat columns
                    sub = frame
                ohlcv = normalize_history_ohlcv(sub, ticker)
                actions = normalize_history_actions(sub, ticker)
                if ohlcv.empty and actions.empty:
                    continue
                ohlcv = reconstruct_as_traded(ohlcv, actions)
                out[ticker] = {
                    Dataset.OHLCV_DAILY: make_fetch_result(
                        ohlcv, self.name, Dataset.OHLCV_DAILY, ticker, start, end
                    ),
                    Dataset.CORPORATE_ACTIONS: make_fetch_result(
                        actions, self.name, Dataset.CORPORATE_ACTIONS, ticker, start, end
                    ),
                }
        return out
