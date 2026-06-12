"""Application settings via pydantic-settings (env prefix QT_, optional .env)."""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QT_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    default_provider: str = "yfinance"
    default_calendar: str = "XNYS"
    default_start_date: date = date(2015, 1, 2)

    alpha_vantage_api_key: SecretStr | None = None
    alpha_vantage_rate_limit_per_min: int = 5
    yfinance_rate_limit_per_min: int = 60

    reconcile_price_rel_tol: float = 0.005
    reconcile_volume_rel_tol: float = 0.10

    stale_price_min_run: int = 5
    zero_volume_min_run: int = 3
    mad_window: int = 63
    mad_threshold: float = 8.0
    gap_threshold: float = 0.30

    @field_validator("alpha_vantage_api_key", mode="before")
    @classmethod
    def _empty_key_is_none(cls, v: object) -> object:
        return None if v in ("", None) else v

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def curated_dir(self) -> Path:
        return self.data_dir / "curated"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def catalog_path(self) -> Path:
        return self.data_dir / "catalog.duckdb"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
