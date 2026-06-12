import numpy as np
import pandas as pd

from qtdata.curation.adjustments import compute_adjustment_factors
from tests.conftest import make_ohlcv


def _actions(rows):
    return pd.DataFrame(rows, columns=["ticker", "ex_date", "action_type", "value"])


def test_no_actions_means_unit_factor():
    df = make_ohlcv(n=20, with_lineage=False)
    factors = compute_adjustment_factors(df, _actions([]))
    assert np.allclose(factors["adj_factor"], 1.0)


def test_four_to_one_split_factor():
    df = make_ohlcv(n=40, with_lineage=False)
    ex = df.loc[20, "date"]
    factors = compute_adjustment_factors(
        df, _actions([("TEST", ex, "split", 4.0)])
    ).set_index("date")
    assert np.allclose(factors.loc[: ex - pd.Timedelta(days=1), "split_factor"], 0.25)
    assert np.allclose(factors.loc[ex:, "split_factor"], 1.0)


def test_dividend_factor_matches_hand_calculation():
    df = make_ohlcv(n=40, with_lineage=False)
    ex = df.loc[15, "date"]
    prev_close = df.loc[14, "close"]
    amount = 2.5
    factors = compute_adjustment_factors(
        df, _actions([("TEST", ex, "dividend", amount)])
    ).set_index("date")
    expected = 1.0 - amount / prev_close
    assert np.allclose(factors.loc[: ex - pd.Timedelta(days=1), "div_factor"], expected)
    assert np.allclose(factors.loc[ex:, "div_factor"], 1.0)


def test_combined_split_and_dividend():
    df = make_ohlcv(n=60, with_lineage=False)
    split_ex = df.loc[40, "date"]
    div_ex = df.loc[20, "date"]
    prev_close = df.loc[19, "close"]
    amount = 1.0
    factors = compute_adjustment_factors(
        df,
        _actions([("TEST", split_ex, "split", 2.0), ("TEST", div_ex, "dividend", amount)]),
    ).set_index("date")
    div_mult = 1.0 - amount / prev_close
    first = factors.iloc[0]
    assert np.isclose(first["adj_factor"], 0.5 * div_mult)
    between = factors.loc[div_ex : split_ex - pd.Timedelta(days=1)]
    assert np.allclose(between["adj_factor"], 0.5)
    assert np.allclose(factors.loc[split_ex:, "adj_factor"], 1.0)


def test_restated_action_flows_through():
    """Vendor restates the split ratio — recomputation reflects it, nothing stored breaks."""
    df = make_ohlcv(n=30, with_lineage=False)
    ex = df.loc[10, "date"]
    f1 = compute_adjustment_factors(df, _actions([("TEST", ex, "split", 2.0)]))
    f2 = compute_adjustment_factors(df, _actions([("TEST", ex, "split", 3.0)]))
    assert np.isclose(f1.iloc[0]["split_factor"], 0.5)
    assert np.isclose(f2.iloc[0]["split_factor"], 1.0 / 3.0)


def test_dividend_with_no_prior_close_is_skipped():
    df = make_ohlcv(n=20, with_lineage=False)
    before_history = df.loc[0, "date"] - pd.Timedelta(days=10)
    factors = compute_adjustment_factors(
        df, _actions([("TEST", before_history, "dividend", 1.0)])
    )
    assert np.allclose(factors["div_factor"], 1.0)
