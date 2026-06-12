"""Event study around extreme-sentiment days.

Abnormal return = the name's daily return minus the equal-weighted
cross-sectional mean return that day (market-adjusted; the event name is part
of the market, negligible at universe breadth). Events align on session
offsets within each ticker's own calendar; incomplete windows are dropped.
Offset 0 is the event session itself — its move can overlap signal formation,
so the clean read is offsets >= +1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EventStudyResult:
    window: tuple[int, int]
    n_pos: int
    n_neg: int
    car_pos: pd.Series  # mean cumulative abnormal return, indexed by session offset
    car_neg: pd.Series


def find_events(
    factor: pd.DataFrame,
    *,
    score_col: str,
    threshold: float = 0.5,
    min_articles: int = 3,
) -> pd.DataFrame:
    """(ticker, date, sign) where |score| >= threshold with enough article support."""
    f = factor.dropna(subset=[score_col])
    mask = (f[score_col].abs() >= threshold) & (f["n_articles"] >= min_articles)
    events = f.loc[mask, ["ticker", "date", score_col]].copy()
    events["sign"] = np.sign(events[score_col]).astype(int)
    return events.loc[events["sign"] != 0, ["ticker", "date", "sign"]].reset_index(drop=True)


def run_event_study(
    closes: pd.DataFrame,
    events: pd.DataFrame,
    window: tuple[int, int] = (-5, 20),
) -> EventStudyResult:
    pre, post = window
    rets = closes.sort_values(["ticker", "date"]).reset_index(drop=True).copy()
    rets["ret"] = rets.groupby("ticker")["close"].pct_change()
    market = rets.groupby("date")["ret"].mean().rename("mkt")
    rets = rets.merge(market, on="date")
    rets["abn"] = rets["ret"] - rets["mkt"]

    frames_pos: list[pd.Series] = []
    frames_neg: list[pd.Series] = []
    for ticker, g in rets.groupby("ticker"):
        ev = events[events["ticker"] == ticker]
        if ev.empty:
            continue
        g = g.reset_index(drop=True)
        position = {d: i for i, d in enumerate(g["date"])}
        for _, e in ev.iterrows():
            i = position.get(e["date"])
            if i is None:
                continue
            lo, hi = i + pre, i + post
            if lo < 0 or hi >= len(g):
                continue  # incomplete window
            abn = g["abn"].iloc[lo : hi + 1].reset_index(drop=True)
            abn.index = range(pre, post + 1)
            (frames_pos if e["sign"] > 0 else frames_neg).append(abn)

    def mean_car(frames: list[pd.Series]) -> pd.Series:
        if not frames:
            return pd.Series(dtype=float)
        return pd.concat(frames, axis=1).mean(axis=1).cumsum()

    return EventStudyResult(
        window, len(frames_pos), len(frames_neg), mean_car(frames_pos), mean_car(frames_neg)
    )
