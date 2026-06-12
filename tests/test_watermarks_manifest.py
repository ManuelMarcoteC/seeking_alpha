from datetime import UTC, date, datetime

from qtdata.ingestion.manifest import ManifestEntry, failures_for_run, recent_runs, record_fetch
from qtdata.ingestion.watermarks import get_watermark, set_watermark
from qtdata.models import Dataset


def test_watermark_roundtrip(catalog):
    assert get_watermark(catalog.conn, "p", Dataset.OHLCV_DAILY, "AAPL") is None
    set_watermark(catalog.conn, "p", Dataset.OHLCV_DAILY, "AAPL", date(2024, 5, 1), "r1")
    assert get_watermark(catalog.conn, "p", Dataset.OHLCV_DAILY, "AAPL") == date(2024, 5, 1)
    set_watermark(catalog.conn, "p", Dataset.OHLCV_DAILY, "AAPL", date(2024, 6, 1), "r2")
    assert get_watermark(catalog.conn, "p", Dataset.OHLCV_DAILY, "AAPL") == date(2024, 6, 1)


def test_watermarks_are_keyed_per_dataset(catalog):
    set_watermark(catalog.conn, "p", Dataset.OHLCV_DAILY, "AAPL", date(2024, 5, 1), "r1")
    assert get_watermark(catalog.conn, "p", Dataset.CORPORATE_ACTIONS, "AAPL") is None


def test_manifest_records_and_aggregates(catalog):
    now = datetime.now(UTC)
    for ticker, status, err in (("AAPL", "success", None), ("MSFT", "failed", "boom")):
        record_fetch(
            catalog.conn,
            ManifestEntry(
                "run1", "p", Dataset.OHLCV_DAILY, ticker,
                date(2024, 1, 1), date(2024, 2, 1),
                10 if status == "success" else 0, "abc" if status == "success" else None,
                status, err, now, now,
            ),
        )
    runs = recent_runs(catalog.conn)
    assert len(runs) == 1
    assert runs.iloc[0]["ok"] == 1
    assert runs.iloc[0]["failed"] == 1
    failures = failures_for_run(catalog.conn, "run1")
    assert list(failures["ticker"]) == ["MSFT"]
    assert failures.iloc[0]["error"] == "boom"
