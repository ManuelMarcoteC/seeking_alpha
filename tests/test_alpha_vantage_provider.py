from datetime import date

import pytest

from qtdata.config import Settings
from qtdata.models import ProviderNotConfiguredError
from qtdata.providers.alpha_vantage_provider import AlphaVantageProvider


def test_raises_clear_error_without_key(tmp_path):
    provider = AlphaVantageProvider(Settings(data_dir=tmp_path, _env_file=None))
    with pytest.raises(ProviderNotConfiguredError, match="QT_ALPHA_VANTAGE_API_KEY"):
        provider.fetch_ohlcv("AAPL", date(2024, 1, 2), date(2024, 1, 31))
    with pytest.raises(ProviderNotConfiguredError):
        provider.fetch_corporate_actions("AAPL", date(2024, 1, 2), date(2024, 1, 31))


def test_parses_time_series_payload(tmp_path, monkeypatch):
    payload = {
        "Time Series (Daily)": {
            "2024-01-03": {
                "1. open": "184.22", "2. high": "185.88", "3. low": "183.43",
                "4. close": "184.25", "5. volume": "58414500",
            },
            "2024-01-02": {
                "1. open": "187.15", "2. high": "188.44", "3. low": "183.89",
                "4. close": "185.64", "5. volume": "82488700",
            },
        }
    }
    monkeypatch.setenv("QT_ALPHA_VANTAGE_API_KEY", "k")
    provider = AlphaVantageProvider(Settings(data_dir=tmp_path, _env_file=None))
    monkeypatch.setattr(provider, "_get", lambda **kw: payload)
    result = provider.fetch_ohlcv("AAPL", date(2024, 1, 1), date(2024, 1, 31))
    df = result.df
    assert list(df["close"]) == [185.64, 184.25]  # sorted ascending by date
    assert df["volume"].dtype == "int64"
