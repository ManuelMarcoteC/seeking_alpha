"""Full offline pipeline: synthetic ingest -> curate -> DuckDB research queries."""

import hashlib
from datetime import date

import numpy as np
import pandas as pd

from qtdata.curation.curate import curate_all
from qtdata.ingestion.ingest import ingest
from qtdata.providers.synthetic_provider import (
    DividendEvent,
    Gap,
    SplitEvent,
    SyntheticProvider,
    ZeroVolumeRun,
)
from qtdata.storage import parquet_store

TICKERS = ["ALPHA", "BETA", "GAMMA"]
START, END = date(2024, 1, 2), date(2024, 12, 30)

EVENTS = {
    "ALPHA": [SplitEvent(ex_date=date(2024, 6, 10), ratio=4.0),
              DividendEvent(ex_date=date(2024, 3, 15), amount=1.2)],
    "BETA": [Gap(on=date(2024, 8, 5), pct=-0.18)],
    "GAMMA": [ZeroVolumeRun(start=date(2024, 5, 6), length=4)],
}


def _curated_hash(settings) -> str:
    df = parquet_store.read(settings.curated_dir / "ohlcv_daily")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return hashlib.sha256(
        df[["ticker", "date", "open", "high", "low", "close", "volume"]]
        .to_csv(index=False)
        .encode()
    ).hexdigest()


def test_end_to_end_synthetic_pipeline(settings, catalog, monkeypatch):
    provider = SyntheticProvider(seed=99, events=EVENTS)
    monkeypatch.setattr("qtdata.ingestion.ingest.get_provider", lambda name, s: provider)

    summary = ingest(settings, catalog, TICKERS, start=START, end=END)
    assert summary.failed == 0
    curate_all(settings, catalog)
    h1 = _curated_hash(settings)

    # --- research queries over the views -----------------------------------
    catalog.refresh_views()
    counts = catalog.query("SELECT ticker, COUNT(*) AS n FROM ohlcv_daily GROUP BY ticker")
    assert set(counts["ticker"]) == set(TICKERS)

    adj = catalog.query(
        "SELECT date, close, close_raw FROM ohlcv_daily_adj WHERE ticker='ALPHA' ORDER BY date"
    )
    pre_split = adj[pd.to_datetime(adj["date"]) < pd.Timestamp(2024, 6, 10)]
    assert (pre_split["close"] < pre_split["close_raw"]).all()  # adjusted-on-read

    # split day must NOT be flagged as an unexplained gap (the action explains it)
    flags = catalog.query("SELECT * FROM validation_flags")
    alpha_gap = flags[
        (flags["ticker"] == "ALPHA") & (flags["flag_type"] == "unexplained_gap")
    ]
    assert alpha_gap.empty
    # BETA's -18% day IS flagged, and its curated price is untouched
    beta_outliers = flags[
        (flags["ticker"] == "BETA") & (flags["flag_type"] == "return_outlier_mad")
    ]
    assert len(beta_outliers) >= 1
    # GAMMA's zero-volume run is flagged
    assert len(flags[(flags["ticker"] == "GAMMA") & (flags["flag_type"] == "zero_volume_run")]) == 4

    # --- idempotency: full re-ingest + curate leaves curated content identical ---
    ingest(settings, catalog, TICKERS, start=START, end=END, full_refresh=True)
    curate_all(settings, catalog)
    assert _curated_hash(settings) == h1

    # --- incremental continuation: extend the window, content stays consistent ---
    summary3 = ingest(settings, catalog, TICKERS, end=date(2025, 3, 31))
    assert summary3.ok > 0
    curate_all(settings, catalog)
    df = parquet_store.read(settings.curated_dir / "ohlcv_daily")
    alpha = df[df["ticker"] == "ALPHA"].sort_values("date")
    # no duplicate sessions, no holes introduced at the seam
    assert not alpha.duplicated(subset=["date"]).any()
    returns = np.log(alpha["close"]).diff().dropna()
    assert np.isfinite(returns).all()
