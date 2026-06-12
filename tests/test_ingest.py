from datetime import date

import pandas as pd
import pytest

from qtdata import calendars
from qtdata.ingestion.ingest import ingest
from qtdata.ingestion.watermarks import get_watermark
from qtdata.models import Dataset
from qtdata.providers.synthetic_provider import SyntheticProvider


@pytest.fixture
def synthetic(monkeypatch):
    """Route provider resolution to a deterministic synthetic instance."""
    provider = SyntheticProvider(seed=42)
    monkeypatch.setattr("qtdata.ingestion.ingest.get_provider", lambda name, s: provider)
    return provider


def test_ingest_writes_raw_manifest_and_watermark(settings, catalog, synthetic):
    summary = ingest(
        settings, catalog, ["AAA", "BBB"],
        provider_name="synthetic",
        start=date(2024, 1, 2), end=date(2024, 3, 28),
        datasets=(Dataset.OHLCV_DAILY,),
    )
    assert summary.ok == 2
    assert summary.failed == 0
    files = list(settings.raw_dir.rglob("*.parquet"))
    assert len(files) == 2
    df = pd.read_parquet(files[0])
    assert {"ticker", "date", "open", "close", "volume", "source", "run_id", "ingested_at"} <= set(
        df.columns
    )
    wm = get_watermark(catalog.conn, "synthetic", Dataset.OHLCV_DAILY, "AAA")
    assert wm == date(2024, 3, 28)


def test_second_ingest_is_incremental(settings, catalog, synthetic):
    ingest(
        settings, catalog, ["AAA"], start=date(2024, 1, 2), end=date(2024, 3, 28),
        datasets=(Dataset.OHLCV_DAILY,),
    )
    summary2 = ingest(
        settings, catalog, ["AAA"], end=date(2024, 6, 28), datasets=(Dataset.OHLCV_DAILY,),
    )
    assert summary2.ok == 1
    files = sorted(settings.raw_dir.rglob("*.parquet"))
    assert len(files) == 2
    second = pd.read_parquet([f for f in files if summary2.run_id in f.name][0])
    expected_start = calendars.next_session(date(2024, 3, 28))
    assert second["date"].min() == expected_start
    # overlap-free and continuous on the synthetic path
    first = pd.read_parquet([f for f in files if summary2.run_id not in f.name][0])
    assert set(first["date"]).isdisjoint(set(second["date"]))


def test_up_to_date_ticker_is_skipped(settings, catalog, synthetic):
    ingest(
        settings, catalog, ["AAA"], start=date(2024, 1, 2), end=date(2024, 3, 28),
        datasets=(Dataset.OHLCV_DAILY,),
    )
    summary = ingest(
        settings, catalog, ["AAA"], end=date(2024, 3, 28), datasets=(Dataset.OHLCV_DAILY,),
    )
    assert summary.skipped == 1
    assert summary.ok == 0


def test_failure_is_isolated(settings, catalog, monkeypatch):
    provider = SyntheticProvider(seed=42)
    real_fetch = provider.fetch_ohlcv

    def flaky(ticker, start, end):
        if ticker == "BAD":
            raise RuntimeError("vendor exploded")
        return real_fetch(ticker, start, end)

    monkeypatch.setattr(provider, "fetch_ohlcv", flaky)
    monkeypatch.setattr("qtdata.ingestion.ingest.get_provider", lambda name, s: provider)

    summary = ingest(
        settings, catalog, ["AAA", "BAD", "CCC"],
        start=date(2024, 1, 2), end=date(2024, 2, 28), datasets=(Dataset.OHLCV_DAILY,),
    )
    assert summary.ok == 2
    assert summary.failed == 1
    assert summary.failures[0][0] == "BAD"
    # failed ticker has no watermark, so the next run retries it
    assert get_watermark(catalog.conn, "synthetic", Dataset.OHLCV_DAILY, "BAD") is None


def test_default_end_is_last_completed_session(settings, catalog, synthetic):
    """Without --end, ingestion must stop at yesterday's close, never today's partial bar."""
    from datetime import date as date_cls

    summary = ingest(
        settings, catalog, ["AAA"], start=date(2024, 1, 2), datasets=(Dataset.OHLCV_DAILY,),
    )
    assert summary.ok == 1
    wm = get_watermark(catalog.conn, "synthetic", Dataset.OHLCV_DAILY, "AAA")
    last_complete = calendars.last_completed_session(date_cls.today()).date()
    assert wm <= last_complete
    assert wm < date_cls.today()


def test_raw_layer_is_append_only(settings, catalog, synthetic):
    ingest(
        settings, catalog, ["AAA"], start=date(2024, 1, 2), end=date(2024, 2, 28),
        datasets=(Dataset.OHLCV_DAILY,), full_refresh=True,
    )
    ingest(
        settings, catalog, ["AAA"], start=date(2024, 1, 2), end=date(2024, 2, 28),
        datasets=(Dataset.OHLCV_DAILY,), full_refresh=True,
    )
    files = list(settings.raw_dir.rglob("*.parquet"))
    assert len(files) == 2  # re-runs append new payloads, never overwrite
