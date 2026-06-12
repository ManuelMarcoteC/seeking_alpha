import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from qtdata.config import Settings
from qtdata.models import Dataset
from qtdata.providers.yfinance_provider import (
    YFinanceProvider,
    normalize_history_actions,
    normalize_history_ohlcv,
)

FIXTURE = Path(__file__).parent / "fixtures" / "yfinance_aapl_5d.json"


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


@pytest.mark.live
def test_live_smoke_aapl(tmp_path):
    provider = YFinanceProvider(Settings(data_dir=tmp_path, _env_file=None))
    result = provider.fetch_ohlcv("AAPL", date(2024, 1, 2), date(2024, 1, 12))
    assert not result.df.empty
    assert (result.df["close"] > 0).all()
