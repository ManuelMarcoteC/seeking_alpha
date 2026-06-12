from datetime import date

import numpy as np
import pandas as pd
import pytest

from qtdata.curation.curate import curate_all
from qtdata.ingestion.ingest import ingest
from qtdata.providers.synthetic_provider import Gap, SplitEvent, SyntheticProvider
from qtdata.storage import parquet_store

START, END = date(2024, 1, 2), date(2024, 6, 28)


def _route(monkeypatch, provider):
    monkeypatch.setattr("qtdata.ingestion.ingest.get_provider", lambda name, s: provider)


def _run_pipeline(settings, catalog, monkeypatch, provider, tickers):
    _route(monkeypatch, provider)
    ingest(settings, catalog, tickers, start=START, end=END)
    return curate_all(settings, catalog)


def test_curate_promotes_and_is_idempotent(settings, catalog, monkeypatch):
    provider = SyntheticProvider(seed=1)
    _, ohlcv_summary = _run_pipeline(settings, catalog, monkeypatch, provider, ["AAA"])
    assert ohlcv_summary.rows_upserted > 0

    curated = parquet_store.read(settings.curated_dir / "ohlcv_daily")
    n1 = len(curated)

    # curating again with no new raw files is a no-op
    _, again = curate_all(settings, catalog)
    assert again.files_processed == 0
    assert len(parquet_store.read(settings.curated_dir / "ohlcv_daily")) == n1


def test_flag_not_mutate_crash_day_survives(settings, catalog, monkeypatch):
    """A -15% day gets flagged but the curated price is EXACTLY what the vendor sent."""
    crash_day = date(2024, 4, 15)
    provider = SyntheticProvider(seed=2, events={"AAA": [Gap(on=crash_day, pct=-0.15)]})
    _run_pipeline(settings, catalog, monkeypatch, provider, ["AAA"])

    raw_files = list((settings.raw_dir).rglob("dataset=ohlcv_daily/**/*.parquet"))
    raw = pd.concat([pd.read_parquet(f) for f in raw_files])
    curated = parquet_store.read(settings.curated_dir / "ohlcv_daily")

    t = pd.Timestamp(crash_day)
    raw_close = raw.loc[raw["date"] == t, "close"].iloc[0]
    curated_close = curated.loc[curated["date"] == t, "close"].iloc[0]
    assert np.isclose(curated_close, raw_close)  # untouched

    flags = parquet_store.read(settings.curated_dir / "validation_flags")
    crash_flags = flags[(pd.to_datetime(flags["date"]) == t)]
    assert "return_outlier_mad" in set(crash_flags["flag_type"])


def test_split_produces_factors_and_no_gap_flag(settings, catalog, monkeypatch):
    split_day = date(2024, 4, 15)
    provider = SyntheticProvider(seed=3, events={"AAA": [SplitEvent(ex_date=split_day, ratio=4.0)]})
    _run_pipeline(settings, catalog, monkeypatch, provider, ["AAA"])

    factors = parquet_store.read(settings.curated_dir / "adjustment_factors")
    t = pd.Timestamp(split_day)
    pre = factors[pd.to_datetime(factors["date"]) < t]
    post = factors[pd.to_datetime(factors["date"]) >= t]
    assert np.allclose(pre["split_factor"], 0.25)
    assert np.allclose(post["split_factor"], 1.0)

    # the -75% day is explained by the split: no unexplained_gap flag
    flags = parquet_store.read(settings.curated_dir / "validation_flags")
    if not flags.empty:
        gap_flags = flags[
            (flags["flag_type"] == "unexplained_gap")
            & (pd.to_datetime(flags["date"]) == t)
        ]
        assert gap_flags.empty


def test_corrupt_raw_rows_are_quarantined(settings, catalog, monkeypatch):
    provider = SyntheticProvider(seed=4)
    _route(monkeypatch, provider)
    ingest(settings, catalog, ["AAA"], start=START, end=date(2024, 2, 28))

    # hand-craft a poisoned raw payload (negative close, high < low)
    bad = provider.fetch_ohlcv("BBB", START, date(2024, 1, 31)).df.assign(
        source="synthetic", run_id="badrun", ingested_at=pd.Timestamp.now(tz="UTC")
    )
    bad.loc[bad.index[3], "close"] = -5.0
    bad.loc[bad.index[5], "high"] = bad.loc[bad.index[5], "low"] - 1.0
    raw_path = (
        settings.raw_dir / "provider=synthetic" / "dataset=ohlcv_daily"
        / "ticker=BBB" / "badrun.parquet"
    )
    parquet_store.write_raw(bad, raw_path)

    _, summary = curate_all(settings, catalog)
    assert summary.rows_quarantined == 2
    curated = parquet_store.read(settings.curated_dir / "ohlcv_daily")
    assert (curated["close"] > 0).all()
    assert (curated["high"] >= curated["low"]).all()
    quarantine_files = list(settings.reports_dir.glob("quarantine_*.parquet"))
    assert len(quarantine_files) == 1


def test_reingest_dedupe_latest_wins(settings, catalog, monkeypatch):
    provider = SyntheticProvider(seed=5)
    _run_pipeline(settings, catalog, monkeypatch, provider, ["AAA"])
    n1 = len(parquet_store.read(settings.curated_dir / "ohlcv_daily"))

    # full-refresh re-ingest of the same window -> duplicate raw -> curated unchanged
    ingest(settings, catalog, ["AAA"], start=START, end=END, full_refresh=True)
    curate_all(settings, catalog)
    assert len(parquet_store.read(settings.curated_dir / "ohlcv_daily")) == n1


@pytest.mark.parametrize("missing_n", [2])
def test_missing_sessions_flagged_via_pipeline(settings, catalog, monkeypatch, missing_n):
    from qtdata import calendars
    from qtdata.providers.synthetic_provider import MissingSessions

    sessions = calendars.sessions_between(START, END)
    holes = tuple(s.date() for s in sessions[30 : 30 + missing_n])
    provider = SyntheticProvider(seed=6, events={"AAA": [MissingSessions(dates=holes)]})
    _run_pipeline(settings, catalog, monkeypatch, provider, ["AAA"])

    flags = parquet_store.read(settings.curated_dir / "validation_flags")
    missing = flags[flags["flag_type"] == "missing_session"]
    assert set(pd.to_datetime(missing["date"]).dt.date) == set(holes)
