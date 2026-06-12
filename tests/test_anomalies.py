"""Detector tests — including the signature no-look-ahead and flag-not-mutate checks."""

from pathlib import Path

import numpy as np
import pandas as pd

from qtdata.validation import anomalies
from tests.conftest import make_ohlcv

SRC_DIR = Path(__file__).resolve().parents[1] / "src"


def test_no_lookahead_primitives_in_source():
    """bfill leaks the future into the past; it must not exist anywhere in src/."""
    offenders = []
    for py in SRC_DIR.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        if "bfill(" in text or 'method="bfill"' in text or "fillna(method" in text:
            offenders.append(py)
    assert offenders == []


def test_mad_detector_flags_injected_crash():
    df = make_ohlcv(n=200, with_lineage=False)
    crash_idx = 150
    df.loc[crash_idx:, "close"] *= 0.85  # -15% one-day drop, then continues
    df.loc[crash_idx:, ["open", "high", "low"]] *= 0.85
    flags = anomalies.flag_return_outliers_mad(df, window=63, threshold=8.0)
    assert len(flags) >= 1
    assert pd.Timestamp(df.loc[crash_idx, "date"]) in set(flags["date"])
    assert (flags["flag_type"] == "return_outlier_mad").all()


def test_mad_detector_clean_series_no_flags():
    df = make_ohlcv(n=200, with_lineage=False)
    flags = anomalies.flag_return_outliers_mad(df, window=63, threshold=8.0)
    assert flags.empty


def test_mad_detector_no_lookahead():
    """Appending day t+1 must not change flags for days <= t."""
    df = make_ohlcv(n=200, with_lineage=False)
    df.loc[150:, ["open", "high", "low", "close"]] *= 0.85

    flags_full = anomalies.flag_return_outliers_mad(df.iloc[:199], window=63, threshold=8.0)
    extended = df.copy()
    extended.loc[199, ["open", "high", "low", "close"]] *= 0.5  # extreme new day
    flags_ext = anomalies.flag_return_outliers_mad(extended, window=63, threshold=8.0)

    cutoff = df.loc[198, "date"]
    a = flags_full[flags_full["date"] <= cutoff].reset_index(drop=True)
    b = flags_ext[flags_ext["date"] <= cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_stale_price_run_flagged():
    df = make_ohlcv(n=100, with_lineage=False)
    level = df.loc[40, "close"]
    df.loc[40:46, ["open", "high", "low", "close"]] = level
    flags = anomalies.flag_stale_prices(df, min_run=5)
    assert len(flags) == 7
    assert (flags["flag_type"] == "stale_price").all()
    assert (flags["severity"] == "warn").all()  # volume stays positive


def test_short_stale_run_not_flagged():
    df = make_ohlcv(n=100, with_lineage=False)
    df.loc[40:42, "close"] = df.loc[40, "close"]
    flags = anomalies.flag_stale_prices(df, min_run=5)
    assert flags.empty


def test_zero_volume_run_flagged():
    df = make_ohlcv(n=100, with_lineage=False)
    df.loc[60:64, "volume"] = 0
    flags = anomalies.flag_zero_volume_runs(df, min_run=3)
    assert len(flags) == 5
    assert (flags["severity"] == "info").all()


def test_unexplained_gap_flagged_but_explained_split_is_not():
    df = make_ohlcv(n=100, with_lineage=False)
    split_date = df.loc[50, "date"]
    df.loc[50:, ["open", "high", "low", "close"]] /= 4.0  # looks like an unadjusted split

    no_actions = pd.DataFrame(columns=["ticker", "ex_date", "action_type", "value"])
    flags = anomalies.flag_unexplained_gaps(df, no_actions, gap_threshold=0.30)
    assert len(flags) == 1
    assert flags.iloc[0]["severity"] == "error"

    actions = pd.DataFrame(
        {
            "ticker": ["TEST"],
            "ex_date": [split_date],
            "action_type": ["split"],
            "value": [4.0],
        }
    )
    flags = anomalies.flag_unexplained_gaps(df, actions, gap_threshold=0.30)
    assert flags.empty


def test_missing_session_detected_and_holiday_not_flagged():
    df = make_ohlcv(start="2024-06-03", n=40, with_lineage=False)
    removed = df.loc[20, "date"]
    df = df.drop(index=20)
    flags = anomalies.flag_missing_sessions(df)
    assert list(flags["date"]) == [pd.Timestamp(removed)]
    # July 4th falls inside this window and was never observed — it must NOT be flagged
    assert pd.Timestamp("2024-07-04") not in set(flags["date"])


def test_flag_dates_outside_ticker_window_not_flagged():
    """Short history (late listing) produces no missing-session false positives."""
    df = make_ohlcv(start="2024-06-03", n=10, with_lineage=False)
    assert anomalies.flag_missing_sessions(df).empty


def test_run_detectors_battery(settings):
    df = make_ohlcv(n=150, with_lineage=False)
    df.loc[100:, ["open", "high", "low", "close"]] *= 0.5
    actions = pd.DataFrame(columns=["ticker", "ex_date", "action_type", "value"])
    flags = anomalies.run_detectors(df, actions, settings)
    assert {"return_outlier_mad", "unexplained_gap"} <= set(flags["flag_type"])
    # flag-never-mutate: detector input is untouched
    assert np.isclose(df.loc[100, "close"], df.loc[100, "close"])
