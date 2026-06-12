import math

import pandas as pd
import pytest

from qtdata.research.ic import daily_ic, summarize_ic
from qtdata.research.returns import forward_returns
from tests.conftest import make_ohlcv


def _panel(n_tickers: int, n: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [
        make_ohlcv(f"T{i:02d}", n=n, seed=i, base_price=50 + i)[["ticker", "date", "close"]]
        for i in range(n_tickers)
    ]
    closes = pd.concat(frames, ignore_index=True)
    fwd = forward_returns(closes, horizons=(1,))
    return closes, fwd


def _factor_from(fwd: pd.DataFrame, flip: bool = False) -> pd.DataFrame:
    factor = fwd.rename(columns={"fwd_1d": "score"}).dropna(subset=["score"]).copy()
    if flip:
        factor["score"] = -factor["score"]
    factor["n_articles"] = 5
    return factor


def test_perfect_foresight_ic_is_one():
    _, fwd = _panel(12)
    factor = _factor_from(fwd)
    daily = daily_ic(factor, fwd, score_col="score", fwd_col="fwd_1d", min_breadth=10)
    assert not daily.empty
    assert daily["ic"].to_numpy() == pytest.approx(1.0)
    assert (daily["n"] == 12).all()


def test_inverted_foresight_ic_is_minus_one():
    _, fwd = _panel(12)
    factor = _factor_from(fwd, flip=True)
    daily = daily_ic(factor, fwd, score_col="score", fwd_col="fwd_1d", min_breadth=10)
    assert daily["ic"].to_numpy() == pytest.approx(-1.0)


def test_breadth_below_floor_yields_no_rows():
    _, fwd = _panel(5)
    factor = _factor_from(fwd)
    daily = daily_ic(factor, fwd, score_col="score", fwd_col="fwd_1d", min_breadth=10)
    assert daily.empty


def test_summarize_ic_arithmetic():
    daily = pd.DataFrame(
        {"date": pd.date_range("2026-01-01", periods=3), "n": 12, "ic": [0.1, 0.2, 0.3]}
    )
    s = summarize_ic(daily, horizon=1)
    assert s.n_days == 3
    assert s.mean_ic == pytest.approx(0.2)
    assert s.std_ic == pytest.approx(0.1)
    assert s.t_stat == pytest.approx(0.2 / 0.1 * math.sqrt(3))
    assert s.icir == pytest.approx(2.0)
    assert s.hit_rate == pytest.approx(1.0)


def test_summarize_ic_empty_is_nan():
    s = summarize_ic(pd.DataFrame(columns=["date", "n", "ic"]), horizon=5)
    assert s.n_days == 0
    assert math.isnan(s.mean_ic)
    assert math.isnan(s.t_stat)
