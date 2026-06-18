import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qtdata.config import Settings
from qtdata.curation.adjustments import compute_adjustment_factors
from qtdata.models import Dataset
from qtdata.providers.yfinance_provider import (
    YFinanceProvider,
    normalize_history_actions,
    normalize_history_ohlcv,
    reconstruct_as_traded,
)
from tests.conftest import make_ohlcv

FIXTURE = Path(__file__).parent / "fixtures" / "yfinance_aapl_5d.json"


def _actions(rows):
    return pd.DataFrame(rows, columns=["ticker", "ex_date", "action_type", "value"])


def _raw_ohlcv(dates, closes, ticker="TEST", volume=1_000_000):
    """Minimal raw-shape OHLCV frame with OHLC all equal to close (easy asserts)."""
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": pd.to_datetime(dates),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": np.full(len(closes), volume, dtype="int64"),
        }
    )


def _vendor_frame() -> pd.DataFrame:
    """Rebuild the vendor-shaped frame (tz-aware index, capitalized columns)."""
    payload = json.loads(FIXTURE.read_text())
    df = pd.DataFrame(payload["data"])
    df.index = pd.DatetimeIndex(payload["index"]).tz_localize("America/New_York")
    return df


def test_normalize_ohlcv_from_recorded_fixture():
    df = normalize_history_ohlcv(_vendor_frame(), "aapl")
    assert list(df.columns) == ["ticker", "date", "open", "high", "low", "close", "volume"]
    assert (df["ticker"] == "AAPL").all()
    assert df["date"].dt.tz is None  # naive session dates
    assert df["volume"].dtype == "int64"
    assert len(df) == 5
    assert (df["high"] >= df["low"]).all()


def test_normalize_actions_from_recorded_fixture():
    df = normalize_history_actions(_vendor_frame(), "aapl")
    # the fixture contains one dividend and one split row
    assert set(df["action_type"]) == {"dividend", "split"}
    div = df[df["action_type"] == "dividend"].iloc[0]
    assert div["value"] == 0.25
    spl = df[df["action_type"] == "split"].iloc[0]
    assert spl["value"] == 4.0


def test_normalize_empty_frame():
    assert normalize_history_ohlcv(pd.DataFrame(), "AAPL").empty
    assert normalize_history_actions(pd.DataFrame(), "AAPL").empty


def test_provider_uses_unadjusted_history(monkeypatch, tmp_path):
    """auto_adjust must be False and end must be made inclusive."""
    captured = {}

    class FakeTicker:
        def __init__(self, symbol):
            captured["symbol"] = symbol

        def history(self, **kwargs):
            captured.update(kwargs)
            return _vendor_frame()

    monkeypatch.setattr("qtdata.providers.yfinance_provider.yf.Ticker", FakeTicker)
    provider = YFinanceProvider(Settings(data_dir=tmp_path, _env_file=None))
    result = provider.fetch_ohlcv("AAPL", date(2024, 1, 2), date(2024, 1, 8))
    assert captured["auto_adjust"] is False
    assert captured["actions"] is True
    assert captured["end"] == date(2024, 1, 9)  # exclusive end shifted by one day
    assert result.dataset is Dataset.OHLCV_DAILY
    assert len(result.df) == 5


def test_fetch_batch_slices_multiindex_download(monkeypatch, tmp_path):
    """yf.download multi-ticker frame -> per-ticker FetchResults for both datasets."""
    vendor = _vendor_frame()
    multi = pd.concat({"AAPL": vendor, "MSFT": vendor * 1.01}, axis=1)
    captured = {}

    def fake_download(tickers, **kwargs):
        captured["tickers"] = list(tickers)
        captured.update(kwargs)
        return multi

    monkeypatch.setattr("qtdata.providers.yfinance_provider.yf.download", fake_download)
    provider = YFinanceProvider(Settings(data_dir=tmp_path, _env_file=None))
    out = provider.fetch_batch(["AAPL", "MSFT"], date(2024, 1, 2), date(2024, 1, 8))

    assert captured["auto_adjust"] is False
    assert captured["actions"] is True
    assert captured["group_by"] == "ticker"
    assert set(out) == {"AAPL", "MSFT"}
    ohlcv = out["AAPL"][Dataset.OHLCV_DAILY].df
    assert len(ohlcv) == 5
    assert (ohlcv["ticker"] == "AAPL").all()
    actions = out["AAPL"][Dataset.CORPORATE_ACTIONS].df
    assert set(actions["action_type"]) == {"dividend", "split"}


def test_fetch_batch_skips_all_nan_ticker(monkeypatch, tmp_path):
    vendor = _vendor_frame()
    dead = vendor.astype("float64") * float("nan")
    multi = pd.concat({"AAPL": vendor, "DEAD": dead}, axis=1)
    monkeypatch.setattr(
        "qtdata.providers.yfinance_provider.yf.download", lambda tickers, **kw: multi
    )
    provider = YFinanceProvider(Settings(data_dir=tmp_path, _env_file=None))
    out = provider.fetch_batch(["AAPL", "DEAD"], date(2024, 1, 2), date(2024, 1, 8))
    assert "AAPL" in out
    assert "DEAD" not in out  # NaN columns = yf.download's silent failure signal


def test_fetch_batch_chunks_by_batch_size(monkeypatch, tmp_path):
    calls = []
    vendor = _vendor_frame()

    def fake_download(tickers, **kwargs):
        calls.append(list(tickers))
        return pd.concat({t: vendor for t in tickers}, axis=1)

    monkeypatch.setattr("qtdata.providers.yfinance_provider.yf.download", fake_download)
    settings = Settings(data_dir=tmp_path, yfinance_batch_size=2, _env_file=None)
    provider = YFinanceProvider(settings)
    out = provider.fetch_batch(["A", "B", "C", "D", "E"], date(2024, 1, 2), date(2024, 1, 8))
    assert [len(c) for c in calls] == [2, 2, 1]
    assert len(out) == 5


@pytest.mark.live
def test_live_smoke_aapl(tmp_path):
    provider = YFinanceProvider(Settings(data_dir=tmp_path, _env_file=None))
    result = provider.fetch_ohlcv("AAPL", date(2024, 1, 2), date(2024, 1, 12))
    assert not result.df.empty
    assert (result.df["close"] > 0).all()


# --- reconstruct_as_traded: invert yfinance's split back-adjustment ----------


def test_reconstruct_no_splits_is_noop():
    """A ticker with no splits (or only dividends) must pass through unchanged."""
    dates = make_ohlcv(n=10, with_lineage=False)["date"]
    closes = np.linspace(100.0, 110.0, 10)
    ohlcv = _raw_ohlcv(dates, closes)
    # only a dividend action -> reconstruct must NOT touch prices (splits only)
    actions = _actions([("TEST", dates.iloc[3], "dividend", 1.5)])
    out = reconstruct_as_traded(ohlcv, actions)
    pd.testing.assert_frame_equal(out, ohlcv)


def test_reconstruct_empty_inputs_pass_through():
    dates = make_ohlcv(n=5, with_lineage=False)["date"]
    ohlcv = _raw_ohlcv(dates, np.full(5, 100.0))
    # no actions at all
    pd.testing.assert_frame_equal(reconstruct_as_traded(ohlcv, _actions([])), ohlcv)
    # empty ohlcv stays empty
    assert reconstruct_as_traded(ohlcv.iloc[0:0], _actions([])).empty


def test_reconstruct_four_to_one_split_recovers_as_traded():
    """AAPL-style 4:1 split: pre-split prices x4, post-split untouched, ex-date untouched."""
    df = make_ohlcv(n=40, with_lineage=False)
    dates = df["date"]
    ex = dates.iloc[20]
    vendor_close = df["close"].to_numpy()  # yfinance already split-adjusted
    ohlcv = _raw_ohlcv(dates, vendor_close, volume=4_000_000)
    out = reconstruct_as_traded(ohlcv, _actions([("TEST", ex, "split", 4.0)]))

    pre = out["date"] < ex
    post = out["date"] >= ex  # ex-date row is already post-split -> NOT un-adjusted
    # pre-split close multiplied back by 4 (recover as-traded tape price)
    assert np.allclose(out.loc[pre, "close"], vendor_close[pre.to_numpy()] * 4.0)
    # post-split (incl. ex-date) unchanged
    assert np.allclose(out.loc[post, "close"], vendor_close[post.to_numpy()])
    # volume divided back by 4 pre-split, unchanged post
    assert np.allclose(out.loc[pre, "volume"], 4_000_000 / 4.0)
    assert np.allclose(out.loc[post, "volume"], 4_000_000)


def test_reconstruct_is_exact_inverse_of_split_factor():
    """The core invariant: future_mult * downstream split_factor == 1 everywhere.

    reconstruct_as_traded (provider) and compute_adjustment_factors (curation)
    must use the SAME dedup and the SAME side='right' convention, so that
    ohlcv_daily_adj re-derives the continuous vendor series with no double-adjust.
    """
    df = make_ohlcv(n=60, with_lineage=False)
    dates = df["date"]
    vendor_close = df["close"].to_numpy()
    ohlcv = _raw_ohlcv(dates, vendor_close)
    actions = _actions([("TEST", dates.iloc[30], "split", 4.0)])

    as_traded = reconstruct_as_traded(ohlcv, actions)
    factors = compute_adjustment_factors(as_traded, actions).set_index("date")
    # applying split_factor to the reconstructed as-traded close must give back
    # exactly the vendor's split-adjusted close we started from
    readjusted = as_traded["close"].to_numpy() * factors["split_factor"].to_numpy()
    assert np.allclose(readjusted, vendor_close)


def test_reconstruct_dedups_vendor_duplicated_split():
    """Samsung-style 50:1 emitted twice within the window collapses to a single 50x.

    A naive implementation would multiply pre-split prices by 50*50 = 2500.
    reconstruct_as_traded must reuse _dedup_splits and apply a single 50x.
    """
    df = make_ohlcv(n=60, with_lineage=False)
    dates = df["date"]
    ex1 = dates.iloc[30]
    ex2 = dates.iloc[35]  # ~5 sessions later, identical ratio -> duplicate
    vendor_close = df["close"].to_numpy()
    ohlcv = _raw_ohlcv(dates, vendor_close)
    out = reconstruct_as_traded(
        ohlcv, _actions([("TEST", ex1, "split", 50.0), ("TEST", ex2, "split", 50.0)])
    )
    pre = out["date"] < ex1
    # exactly 50x, never 2500x
    assert np.allclose(out.loc[pre, "close"], vendor_close[pre.to_numpy()] * 50.0)
    assert np.allclose(out.loc[~pre.to_numpy(), "close"], vendor_close[(~pre).to_numpy()])
