from datetime import date

import numpy as np
import pandas as pd

from qtdata.curation.curate import curate_all
from qtdata.ingestion.ingest import ingest
from qtdata.providers.synthetic_provider import DividendEvent, SplitEvent, SyntheticProvider


def _pipeline(settings, catalog, monkeypatch, events=None):
    provider = SyntheticProvider(seed=11, events=events or {})
    monkeypatch.setattr("qtdata.ingestion.ingest.get_provider", lambda name, s: provider)
    ingest(settings, catalog, ["AAA"], start=date(2024, 1, 2), end=date(2024, 6, 28))
    curate_all(settings, catalog)


def test_views_created_and_queryable(settings, catalog, monkeypatch):
    _pipeline(settings, catalog, monkeypatch)
    created = catalog.refresh_views()
    assert "ohlcv_daily" in created
    df = catalog.query("SELECT ticker, COUNT(*) AS n FROM ohlcv_daily GROUP BY ticker")
    assert df.iloc[0]["ticker"] == "AAA"
    assert df.iloc[0]["n"] > 100


def test_adjusted_view_matches_python_factors(settings, catalog, monkeypatch):
    split_day = date(2024, 4, 15)
    div_day = date(2024, 2, 15)
    _pipeline(
        settings, catalog, monkeypatch,
        events={"AAA": [SplitEvent(ex_date=split_day, ratio=4.0),
                        DividendEvent(ex_date=div_day, amount=1.5)]},
    )
    catalog.refresh_views()
    df = catalog.query(
        "SELECT date, close, close_raw, adj_factor, volume FROM ohlcv_daily_adj ORDER BY date"
    )
    # adjusted close = raw close * factor, row by row
    assert np.allclose(df["close"], df["close_raw"] * df["adj_factor"])
    # pre-split adjusted close ~ raw/4 modulo the dividend factor (< 2% effect)
    pre = df[pd.to_datetime(df["date"]) < pd.Timestamp(split_day)]
    ratio = (pre["close"] / pre["close_raw"]).to_numpy()
    assert (ratio < 0.25 + 1e-9).all()
    assert (ratio > 0.25 * 0.95).all()
    post = df[pd.to_datetime(df["date"]) >= pd.Timestamp(split_day)]
    assert np.allclose(post["close"], post["close_raw"])


def test_clean_view_joins_flags(settings, catalog, monkeypatch):
    from qtdata.providers.synthetic_provider import Gap

    _pipeline(
        settings, catalog, monkeypatch,
        events={"AAA": [Gap(on=date(2024, 4, 15), pct=-0.2)]},
    )
    catalog.refresh_views()
    df = catalog.query(
        "SELECT * FROM ohlcv_daily_clean WHERE n_flags > 0 ORDER BY date"
    )
    assert len(df) >= 1
    assert "return_outlier_mad" in df.iloc[0]["flag_types"]


def test_catalog_query_roundtrip(settings, catalog):
    out = catalog.query("SELECT 41 + 1 AS answer")
    assert out.iloc[0]["answer"] == 42
