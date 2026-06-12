import pandas as pd
import pytest

from qtdata.research.returns import forward_returns
from tests.conftest import make_ohlcv


def _closes(*frames) -> pd.DataFrame:
    df = pd.concat(frames, ignore_index=True)
    return df[["ticker", "date", "close"]].copy()


def test_fwd_1d_is_next_session_return():
    closes = _closes(make_ohlcv("AAA", n=10))
    fwd = forward_returns(closes, horizons=(1,))
    expected = closes["close"].iloc[3] / closes["close"].iloc[2] - 1.0
    assert fwd["fwd_1d"].iloc[2] == pytest.approx(expected)
    assert pd.isna(fwd["fwd_1d"].iloc[-1])  # tail has no future session


def test_shift_stays_within_ticker():
    a = make_ohlcv("AAA", n=10, seed=1)
    b = make_ohlcv("BBB", n=10, seed=2)
    fwd = forward_returns(_closes(a, b), horizons=(1,))
    # last session of AAA must NOT borrow BBB's first close
    last_a = fwd[fwd["ticker"] == "AAA"].iloc[-1]
    assert pd.isna(last_a["fwd_1d"])


def test_gap_uses_next_observed_session():
    a = make_ohlcv("AAA", n=10)
    gapped = a.drop(index=3).reset_index(drop=True)  # session 3 missing (halt)
    fwd = forward_returns(_closes(gapped), horizons=(1,))
    # at session 2 the next OBSERVED close is session 4's
    expected = a["close"].iloc[4] / a["close"].iloc[2] - 1.0
    assert fwd["fwd_1d"].iloc[2] == pytest.approx(expected)


def test_no_lookahead_truncation_invariance():
    closes = _closes(make_ohlcv("AAA", n=20))
    full = forward_returns(closes, horizons=(5,))
    truncated = forward_returns(closes.iloc[:15].copy(), horizons=(5,))
    # every date with a complete window in the truncated set matches the full run
    merged = truncated.merge(full, on=["ticker", "date"], suffixes=("_t", "_f"))
    complete = merged.dropna(subset=["fwd_5d_t"])
    assert len(complete) == 10
    pd.testing.assert_series_equal(
        complete["fwd_5d_t"], complete["fwd_5d_f"], check_names=False
    )
