"""Batched ingestion path: grouping, per-ticker bookkeeping, fallback isolation."""

from datetime import date

import pandas as pd
import pytest

from qtdata.ingestion.ingest import ingest
from qtdata.ingestion.watermarks import get_watermark
from qtdata.models import Dataset
from qtdata.providers.base import BatchFetchProtocol
from qtdata.providers.synthetic_provider import SyntheticProvider

START, END = date(2024, 1, 2), date(2024, 2, 28)


class SpyBatchProvider(SyntheticProvider):
    """Synthetic provider that records every fetch_batch / per-ticker call."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_calls: list[tuple[tuple[str, ...], date, date]] = []
        self.single_calls: list[str] = []

    def fetch_batch(self, tickers, start, end):
        self.batch_calls.append((tuple(tickers), start, end))
        self._in_batch = True
        try:
            return super().fetch_batch(tickers, start, end)
        finally:
            self._in_batch = False

    def fetch_ohlcv(self, ticker, start, end):
        if not getattr(self, "_in_batch", False):
            self.single_calls.append(ticker)  # only count true per-ticker path
        return super().fetch_ohlcv(ticker, start, end)


@pytest.fixture
def spy(monkeypatch):
    provider = SpyBatchProvider(seed=42)
    monkeypatch.setattr("qtdata.ingestion.ingest.get_provider", lambda name, s: provider)
    return provider


def test_synthetic_implements_batch_protocol():
    assert isinstance(SyntheticProvider(), BatchFetchProtocol)


def test_multi_ticker_uses_single_batch_call(settings, catalog, spy):
    summary = ingest(settings, catalog, ["AAA", "BBB", "CCC"], start=START, end=END)
    assert summary.ok == 3    # 3 tickers x OHLCV
    assert summary.empty == 3  # eventless synthetics have no corporate actions
    assert len(spy.batch_calls) == 1
    assert spy.batch_calls[0][0] == ("AAA", "BBB", "CCC")
    assert spy.single_calls == []  # no per-ticker fallback needed


def test_batch_keeps_per_ticker_manifest_and_watermarks(settings, catalog, spy):
    ingest(settings, catalog, ["AAA", "BBB"], start=START, end=END,
           datasets=(Dataset.OHLCV_DAILY,))
    for t in ("AAA", "BBB"):
        assert get_watermark(catalog.conn, "synthetic", Dataset.OHLCV_DAILY, t) == END
    rows = catalog.conn.execute(
        "SELECT ticker, status FROM ingestion_manifest ORDER BY ticker"
    ).fetchall()
    assert [r[0] for r in rows] == ["AAA", "BBB"]
    assert all(r[1] == "success" for r in rows)


def test_batch_groups_by_effective_start(settings, catalog, spy):
    # AAA gets a head start; BBB starts from scratch -> different effective starts
    ingest(settings, catalog, ["AAA"], start=START, end=date(2024, 1, 31),
           datasets=(Dataset.OHLCV_DAILY,))
    spy.batch_calls.clear()

    ingest(settings, catalog, ["AAA", "BBB"], end=END, datasets=(Dataset.OHLCV_DAILY,))
    # two groups -> two batch calls (different signatures can't share one window)
    assert len(spy.batch_calls) == 2
    grouped = {call[0] for call in spy.batch_calls}
    assert grouped == {("AAA",), ("BBB",)}

    # AAA's second payload starts strictly after its watermark; no overlap
    raw = sorted((settings.raw_dir / "provider=synthetic").rglob("ticker=AAA/*.parquet"))
    assert len(raw) == 2
    first = pd.read_parquet(raw[0])
    second = pd.read_parquet(raw[1])
    assert set(first["date"]).isdisjoint(set(second["date"]))


def test_batch_failure_falls_back_to_per_ticker(settings, catalog, monkeypatch):
    provider = SpyBatchProvider(seed=7)

    def exploding_batch(tickers, start, end):
        provider.batch_calls.append((tuple(tickers), start, end))
        raise RuntimeError("vendor batch endpoint down")

    monkeypatch.setattr(provider, "fetch_batch", exploding_batch)
    monkeypatch.setattr("qtdata.ingestion.ingest.get_provider", lambda name, s: provider)

    summary = ingest(settings, catalog, ["AAA", "BBB"], start=START, end=END,
                     datasets=(Dataset.OHLCV_DAILY,))
    assert summary.ok == 2          # fallback recovered everything
    assert summary.failed == 0
    assert provider.single_calls == ["AAA", "BBB"]


def test_ticker_missing_from_batch_recorded_as_empty(settings, catalog, monkeypatch):
    provider = SpyBatchProvider(seed=7)
    real_batch = SyntheticProvider.fetch_batch

    def partial_batch(tickers, start, end):
        out = real_batch(provider, tickers, start, end)
        out.pop("GONE", None)
        return out

    monkeypatch.setattr(provider, "fetch_batch", partial_batch)
    monkeypatch.setattr("qtdata.ingestion.ingest.get_provider", lambda name, s: provider)

    summary = ingest(settings, catalog, ["AAA", "GONE"], start=START, end=END,
                     datasets=(Dataset.OHLCV_DAILY,))
    assert summary.ok == 1
    assert summary.empty == 1
    # empty does NOT advance the watermark -> retried next run
    assert get_watermark(catalog.conn, "synthetic", Dataset.OHLCV_DAILY, "GONE") is None


def test_single_ticker_skips_batch_path(settings, catalog, spy):
    ingest(settings, catalog, ["AAA"], start=START, end=END,
           datasets=(Dataset.OHLCV_DAILY,))
    assert spy.batch_calls == []
    assert spy.single_calls == ["AAA"]
