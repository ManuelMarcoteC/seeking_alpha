"""Provider registry. Providers are interchangeable behind DataProviderProtocol."""

from __future__ import annotations

from qtdata.config import Settings
from qtdata.providers.base import DataProviderProtocol

PROVIDER_NAMES = ("yfinance", "synthetic", "alpha_vantage")


def get_provider(name: str, settings: Settings) -> DataProviderProtocol:
    key = name.lower().strip()
    if key == "yfinance":
        from qtdata.providers.yfinance_provider import YFinanceProvider

        return YFinanceProvider(settings)
    if key == "synthetic":
        from qtdata.providers.synthetic_provider import SyntheticProvider

        return SyntheticProvider(calendar=settings.default_calendar)
    if key == "alpha_vantage":
        from qtdata.providers.alpha_vantage_provider import AlphaVantageProvider

        return AlphaVantageProvider(settings)
    raise ValueError(f"Unknown provider {name!r}; expected one of {PROVIDER_NAMES}")
