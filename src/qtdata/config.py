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
    default_universe: str = "NASDAQ"

    nasdaq_listed_url: str = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"

    alpha_vantage_api_key: SecretStr | None = None
    alpha_vantage_rate_limit_per_min: int = 5
    yfinance_rate_limit_per_min: int = 60
    yfinance_batch_size: int = 200
    yfinance_batch_threads: int = 8

    # agent layer
    anthropic_api_key: SecretStr | None = None  # falls back to ANTHROPIC_API_KEY
    # Bill the agent against a Claude Pro/Max subscription (OAuth) instead of a
    # pay-per-token API key. Reuses the token the official Claude Code CLI stores.
    agent_use_subscription: bool = False
    agent_credentials_path: str = "~/.claude/.credentials.json"
    agent_model: str = "claude-opus-4-8"
    agent_max_rounds: int = 6
    agent_sql_row_cap: int = 50
    agent_sql_col_cap: int = 20
    agent_sql_timeout_s: int = 15
    agent_max_sector_pct: float = 0.40

    # news / sentiment
    news_relevance_floor: float = 0.25
    news_cutoff_local: str = "15:30"
    news_decay_tau_days: float = 7.0
    av_news_page_limit: int = 23
    finbert_revision: str = "4556d13015211d73dccd3fdd39d39232506f3e43"

    reconcile_price_rel_tol: float = 0.005
    reconcile_volume_rel_tol: float = 0.10

    stale_price_min_run: int = 5
    zero_volume_min_run: int = 3
    mad_window: int = 63
    mad_threshold: float = 8.0
    gap_threshold: float = 0.30

    @field_validator("alpha_vantage_api_key", "anthropic_api_key", mode="before")
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
