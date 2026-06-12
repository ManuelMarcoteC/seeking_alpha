"""Statistical anomaly detectors — flag, never mutate.

Every detector is a pure function DataFrame -> DataFrame of flags
(ticker, date, flag_type, severity, detail). Prices are never altered:
a -12% COVID day gets flagged and STAYS in the data.

The return-outlier detector uses a TRAILING rolling median/MAD shifted by one
day, so the statistic for day t only sees data through t-1 — appending a new
day never changes historical flags (no look-ahead).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from qtdata import calendars
from qtdata.config import Settings
from qtdata.models import Severity

FLAG_COLUMNS = ["ticker", "date", "flag_type", "severity", "detail"]


def _empty_flags() -> pd.DataFrame:
    return pd.DataFrame(columns=FLAG_COLUMNS)


def _flags(ticker: str, dates: pd.Series, flag_type: str, severity, detail) -> pd.DataFrame:
    if len(dates) == 0:
        return _empty_flags()
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": pd.to_datetime(dates.to_numpy()),
            "flag_type": flag_type,
            "severity": severity if isinstance(severity, str) else list(severity),
            "detail": detail if isinstance(detail, str) else list(detail),
        }
    )


def _rolling_mad(x: pd.Series, window: int) -> pd.Series:
    def mad(values: np.ndarray) -> float:
        med = np.median(values)
        return float(np.median(np.abs(values - med)))

    return x.rolling(window, min_periods=window).apply(mad, raw=True)


def flag_return_outliers_mad(
    df: pd.DataFrame, window: int = 63, threshold: float = 8.0
) -> pd.DataFrame:
    """Robust |z| of log returns vs trailing rolling median/MAD (shifted: no look-ahead)."""
    out: list[pd.DataFrame] = []
    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("date")
        r = np.log(g["close"]).diff()
        med = r.rolling(window, min_periods=window).median().shift(1)
        mad = _rolling_mad(r, window).shift(1)
        mad = mad.where(mad > 0)
        rz = 0.6745 * (r - med) / mad
        hit = rz.abs() > threshold
        hit = hit.fillna(False)
        if hit.any():
            details = [
                json.dumps({"robust_z": round(float(z), 2), "log_return": round(float(lr), 4)})
                for z, lr in zip(rz[hit], r[hit], strict=True)
            ]
            out.append(
                _flags(ticker, g.loc[hit, "date"], "return_outlier_mad", Severity.WARN, details)
            )
    return pd.concat(out, ignore_index=True) if out else _empty_flags()


def _run_lengths(mask: pd.Series) -> pd.Series:
    """Length of the run of consecutive True values each row belongs to (0 where False)."""
    grp = (mask != mask.shift()).cumsum()
    sizes = mask.groupby(grp).transform("size")
    return sizes.where(mask, 0)


def flag_stale_prices(df: pd.DataFrame, min_run: int = 5) -> pd.DataFrame:
    out: list[pd.DataFrame] = []
    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("date")
        run_id = (g["close"] != g["close"].shift()).cumsum()
        lengths = g.groupby(run_id)["close"].transform("size")
        hit = lengths >= min_run
        if hit.any():
            sev = np.where(g.loc[hit, "volume"] > 0, Severity.WARN.value, Severity.INFO.value)
            details = [json.dumps({"run_length": int(n)}) for n in lengths[hit]]
            out.append(_flags(ticker, g.loc[hit, "date"], "stale_price", sev, details))
    return pd.concat(out, ignore_index=True) if out else _empty_flags()


def flag_zero_volume_runs(df: pd.DataFrame, min_run: int = 3) -> pd.DataFrame:
    out: list[pd.DataFrame] = []
    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("date")
        lengths = _run_lengths(g["volume"] == 0)
        hit = lengths >= min_run
        if hit.any():
            details = [json.dumps({"run_length": int(n)}) for n in lengths[hit]]
            out.append(
                _flags(ticker, g.loc[hit, "date"], "zero_volume_run", Severity.INFO, details)
            )
    return pd.concat(out, ignore_index=True) if out else _empty_flags()


def flag_unexplained_gaps(
    df: pd.DataFrame, actions: pd.DataFrame, gap_threshold: float = 0.30
) -> pd.DataFrame:
    """|close-to-close return| beyond threshold with NO corporate action on that date.

    The classic signature of an unadjusted split leaking through, or a bad print.
    Severity 'error' — but still only a flag.
    """
    out: list[pd.DataFrame] = []
    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("date")
        ret = g["close"].pct_change()
        hit = ret.abs() > gap_threshold
        if not hit.any():
            continue
        if not actions.empty:
            act_dates = set(
                pd.to_datetime(actions.loc[actions["ticker"] == ticker, "ex_date"]).dt.normalize()
            )
            hit &= ~g["date"].isin(act_dates)
        if hit.any():
            details = [json.dumps({"return": round(float(r), 4)}) for r in ret[hit]]
            out.append(
                _flags(ticker, g.loc[hit, "date"], "unexplained_gap", Severity.ERROR, details)
            )
    return pd.concat(out, ignore_index=True) if out else _empty_flags()


def flag_missing_sessions(df: pd.DataFrame, calendar: str = "XNYS") -> pd.DataFrame:
    """Exchange sessions inside the ticker's own [first, last] window with no bar.

    Bounded per ticker, so short histories (late listings, delistings) don't
    produce false positives.
    """
    out: list[pd.DataFrame] = []
    for ticker, g in df.groupby("ticker"):
        observed = pd.DatetimeIndex(g["date"]).normalize()
        expected = calendars.sessions_between(observed.min(), observed.max(), calendar)
        missing = expected.difference(observed)
        if len(missing) > 0:
            out.append(
                _flags(
                    ticker,
                    pd.Series(missing),
                    "missing_session",
                    Severity.WARN,
                    json.dumps({"calendar": calendar}),
                )
            )
    return pd.concat(out, ignore_index=True) if out else _empty_flags()


def run_detectors(
    ohlcv: pd.DataFrame, actions: pd.DataFrame, settings: Settings
) -> pd.DataFrame:
    """Run the full detector battery over curated-shape OHLCV data."""
    if ohlcv.empty:
        return _empty_flags()
    parts = [
        flag_return_outliers_mad(ohlcv, settings.mad_window, settings.mad_threshold),
        flag_stale_prices(ohlcv, settings.stale_price_min_run),
        flag_zero_volume_runs(ohlcv, settings.zero_volume_min_run),
        flag_unexplained_gaps(ohlcv, actions, settings.gap_threshold),
        flag_missing_sessions(ohlcv, settings.default_calendar),
    ]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return _empty_flags()
    flags = pd.concat(parts, ignore_index=True)
    if not flags.empty:
        flags["date"] = pd.to_datetime(flags["date"]).dt.normalize()
    return flags
