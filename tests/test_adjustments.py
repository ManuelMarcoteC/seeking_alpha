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


def test_duplicate_split_same_ratio_is_deduped():
    """Vendor (yfinance) emits the same split twice days apart (Samsung 2018 50:1).

    Without dedup the factor compounds 50*50 = 2500x and corrupts all pre-split
    adjusted prices. The two ex-dates within the window must collapse to a single
    50x split anchored at the earliest ex-date.
    """
    df = make_ohlcv(n=60, with_lineage=False)
    ex1 = df.loc[30, "date"]
    ex2 = df.loc[38, "date"]  # ~8 sessions later, identical ratio
    factors = compute_adjustment_factors(
        df, _actions([("TEST", ex1, "split", 50.0), ("TEST", ex2, "split", 50.0)])
    ).set_index("date")
    # before the (single, earliest) split: exactly 1/50, never 1/2500
    assert np.allclose(factors.loc[: ex1 - pd.Timedelta(days=1), "split_factor"], 1.0 / 50.0)
    # from the earliest ex-date onward the split is already in the raw price
    assert np.allclose(factors.loc[ex1:, "split_factor"], 1.0)


def test_distinct_splits_same_ratio_far_apart_both_kept():
    """Two legitimate identical-ratio splits outside the window must BOTH apply."""
    df = make_ohlcv(n=120, with_lineage=False)
    ex1 = df.loc[30, "date"]
    ex2 = df.loc[90, "date"]  # months apart -> genuinely distinct events
    factors = compute_adjustment_factors(
        df, _actions([("TEST", ex1, "split", 2.0), ("TEST", ex2, "split", 2.0)])
    ).set_index("date")
    assert np.allclose(factors.loc[: ex1 - pd.Timedelta(days=1), "split_factor"], 0.25)
    assert np.allclose(
        factors.loc[ex1 : ex2 - pd.Timedelta(days=1), "split_factor"], 0.5
    )
    assert np.allclose(factors.loc[ex2:, "split_factor"], 1.0)
