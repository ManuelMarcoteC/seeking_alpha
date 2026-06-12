"""Adjustment factors derived from the corporate-actions table (CRSP-style).

Adjusted prices are NEVER stored as truth: factors are recomputed from raw
closes + actions whenever either changes, so a vendor restating a split simply
flows through on the next curation pass.

Convention: factor applies multiplicatively to all dates STRICTLY BEFORE the
ex-date. For a 4:1 split, pre-split adjusted close = raw close * (1/4).
For a dividend d with previous close p, pre-ex prices scale by (1 - d/p).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from qtdata.models import ActionType

logger = logging.getLogger(__name__)

FACTOR_COLUMNS = ["ticker", "date", "split_factor", "div_factor", "adj_factor"]


def _suffix_factor(dates: np.ndarray, ex_dates: np.ndarray, mults: np.ndarray) -> np.ndarray:
    """factor_t = product of mults for events with ex_date > t."""
    if len(ex_dates) == 0:
        return np.ones(len(dates))
    order = np.argsort(ex_dates)
    ex_sorted, m_sorted = ex_dates[order], mults[order]
    # suffix[i] = prod(m_sorted[i:]); suffix[len] = 1
    suffix = np.ones(len(m_sorted) + 1)
    suffix[:-1] = np.cumprod(m_sorted[::-1])[::-1]
    idx = np.searchsorted(ex_sorted, dates, side="right")
    return suffix[idx]


def compute_adjustment_factors(ohlcv: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Per-(ticker, date) split/dividend/total adjustment factors."""
    if ohlcv.empty:
        return pd.DataFrame(columns=FACTOR_COLUMNS)

    out: list[pd.DataFrame] = []
    for ticker, g in ohlcv.groupby("ticker"):
        g = g.sort_values("date")
        dates = pd.to_datetime(g["date"]).to_numpy()
        closes = g["close"].to_numpy()

        if actions.empty:
            acts = actions
        else:
            acts = actions[actions["ticker"] == ticker]

        split_mults = np.array([])
        split_ex = np.array([], dtype="datetime64[ns]")
        div_mults_list: list[float] = []
        div_ex_list: list[np.datetime64] = []

        if not acts.empty:
            splits = acts[acts["action_type"] == ActionType.SPLIT.value]
            if not splits.empty:
                split_ex = pd.to_datetime(splits["ex_date"]).to_numpy()
                split_mults = 1.0 / splits["value"].to_numpy(dtype=float)

            divs = acts[acts["action_type"] == ActionType.DIVIDEND.value]
            for _, d in divs.iterrows():
                ex = np.datetime64(pd.Timestamp(d["ex_date"]))
                pos = int(np.searchsorted(dates, ex))
                if pos == 0:
                    continue  # no prior close in our history; factor unidentifiable
                prev_close = float(closes[pos - 1])
                mult = 1.0 - float(d["value"]) / prev_close
                if mult <= 0:
                    logger.warning(
                        "Dividend %.4f >= prev close %.4f for %s on %s; skipping factor",
                        d["value"], prev_close, ticker, d["ex_date"],
                    )
                    continue
                div_ex_list.append(ex)
                div_mults_list.append(mult)

        split_factor = _suffix_factor(dates, split_ex, split_mults)
        div_factor = _suffix_factor(
            dates, np.array(div_ex_list, dtype="datetime64[ns]"), np.array(div_mults_list)
        )
        out.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": dates,
                    "split_factor": split_factor,
                    "div_factor": div_factor,
                    "adj_factor": split_factor * div_factor,
                }
            )
        )
    return pd.concat(out, ignore_index=True)
