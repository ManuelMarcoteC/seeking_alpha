"""Tests for the AI-Asia basket analytics — focus on the short-history guard."""

import numpy as np
import pandas as pd

from qtdata.research.aiasia import (
    DEFAULT_BASKET,
    _relative_strength,
    _rsi,
    build_basket_report,
    compute_ticker_metrics,
    render_basket_table,
)
from qtdata.research.returns import load_adjusted_closes  # noqa: F401 - sibling sanity


def test_short_history_returns_na_not_garbage():
    # 5 sessions: ret_20d, vs_sma20/50, rsi_14 must be None (not computed on too few bars)
    closes = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    m = compute_ticker_metrics("0100.HK", closes, sentiment=0.5, n_news=3, rel_strength=None)
    assert m.sessions == 5
    assert m.ret_5d is None  # needs >5 sessions
    assert m.ret_20d is None
    assert m.vs_sma20 is None
    assert m.vs_sma50 is None
    assert m.rsi_14 is None
    # but price and sentiment ARE available even with 1 bar
    assert m.last_close == 104.0
    assert m.sentiment == 0.5


def test_single_bar_ipo_all_technical_na():
    m = compute_ticker_metrics("0100.HK", np.array([497.6]), None, 0, None)
    assert m.sessions == 1
    assert m.last_close == 497.6
    assert all(v is None for v in (m.ret_5d, m.ret_20d, m.vs_sma20, m.vs_sma50, m.rsi_14))


def test_long_history_computes_all_metrics():
    rng = np.random.default_rng(0)
    closes = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, size=120))
    m = compute_ticker_metrics("005930.KS", closes, 0.3, 50, 0.1)
    assert m.sessions == 120
    assert m.ret_5d is not None
    assert m.ret_20d is not None
    assert m.vs_sma20 is not None
    assert m.vs_sma50 is not None
    assert m.rsi_14 is not None and 0.0 <= m.rsi_14 <= 100.0


def test_rsi_all_up_is_100():
    closes = np.arange(1, 30, dtype=float)  # strictly increasing -> RSI 100
    assert _rsi(closes) == 100.0


def test_rsi_too_short_is_none():
    assert _rsi(np.array([1.0, 2.0, 3.0])) is None


def test_relative_strength_is_return_minus_median():
    rets = {"A": 0.10, "B": 0.00, "C": -0.10, "D": None}
    rel = _relative_strength(rets)
    # median of [0.10, 0.00, -0.10] = 0.00
    assert rel["A"] == 0.10
    assert rel["B"] == 0.00
    assert rel["C"] == -0.10
    assert rel["D"] is None  # undefined return stays None


def test_relative_strength_all_none():
    rel = _relative_strength({"A": None, "B": None})
    assert all(v is None for v in rel.values())


def test_build_basket_report_integration(settings, catalog):
    """End-to-end against a tiny synthetic lake: short and long names coexist."""
    # long name: 60 sessions; short name: 3 sessions
    long_dates = pd.bdate_range("2026-01-01", periods=60)
    short_dates = pd.bdate_range("2026-06-01", periods=3)
    rows = []
    for d in long_dates:
        rows.append(("LONG.KS", d, 1000.0 + (d - long_dates[0]).days))
    for d in short_dates:
        rows.append(("SHORT.HK", d, 500.0))
    df = pd.DataFrame(rows, columns=["ticker", "date", "close_raw"])
    df["close"] = df["close_raw"]
    df["adj_factor"] = 1.0
    df["volume"] = 1000
    df["open"] = df["close_raw"]
    df["high"] = df["close_raw"]
    df["low"] = df["close_raw"]
    df["source"] = "test"
    catalog.conn.register("seed_df", df)
    catalog.conn.execute("CREATE OR REPLACE VIEW ohlcv_daily_adj AS SELECT * FROM seed_df")

    report = build_basket_report(catalog, basket=("LONG.KS", "SHORT.HK"))
    by = {m.ticker: m for m in report.metrics}
    assert by["LONG.KS"].ret_20d is not None
    assert by["SHORT.HK"].ret_20d is None  # 3 sessions -> N/A
    assert by["SHORT.HK"].last_close == 500.0
    # render must not raise and must contain the N/A marker
    txt = render_basket_table(report)
    assert "AI-Asia basket" in txt
    assert "N/A" in txt


def test_default_basket_has_expected_names():
    assert "2513.HK" in DEFAULT_BASKET
    assert "0100.HK" in DEFAULT_BASKET
    assert "005930.KS" in DEFAULT_BASKET
    assert len(DEFAULT_BASKET) == 6
