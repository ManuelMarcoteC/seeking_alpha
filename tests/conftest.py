from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qtdata import calendars
from qtdata.config import Settings
from qtdata.storage.catalog import Catalog


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path / "data", _env_file=None)


@pytest.fixture
def catalog(settings):
    with Catalog(settings) as cat:
        cat.init_schema()
        yield cat


def make_ohlcv(
    ticker: str = "TEST",
    start: str = "2024-01-02",
    n: int = 120,
    seed: int = 7,
    base_price: float = 100.0,
    with_lineage: bool = True,
) -> pd.DataFrame:
    """Synthetic curated-shape OHLCV frame over real XNYS sessions."""
    sessions = calendars.sessions_between(start, "2030-12-31")[:n]
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.012, size=n)
    close = base_price * np.exp(np.cumsum(rets))
    open_ = np.empty(n)
    open_[0] = base_price
    open_[1:] = close[:-1]
    spread = np.abs(rng.normal(0, 0.005, size=n))
    df = pd.DataFrame(
        {
            "ticker": ticker,
            "date": sessions,
            "open": open_,
            "high": np.maximum(open_, close) * (1 + spread),
            "low": np.minimum(open_, close) * (1 - spread),
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, size=n),
        }
    )
    if with_lineage:
        df["source"] = "test"
        df["run_id"] = "testrun"
        df["ingested_at"] = pd.Timestamp.now(tz="UTC")
    return df


@pytest.fixture
def ohlcv_factory():
    return make_ohlcv
