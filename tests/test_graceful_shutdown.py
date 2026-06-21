"""SIGTERM -> ordinary unwind, so finally/cleanup runs on container stop.

CPython default disposition for SIGTERM terminates without raising; Ctrl+C
(SIGINT) raises KeyboardInterrupt. Manual testing with Ctrl+C therefore
validates a path that never runs under docker/k8s. We convert SIGTERM into a
catchable SystemExit and verify cleanup fires.

Ported from the SOFA TIL on the SIGTERM/finally asymmetry.
"""
import os
import signal
import time

import pytest

from qtdata.ingestion import shutdown


def test_sigterm_raises_system_exit_and_runs_finally():
    cleaned = {"ran": False}
    # Sentinel: if the code under test has not installed its handler yet, a
    # stray SIGTERM becomes a test failure instead of killing the test runner.
    def _sentinel(signum, frame):
        raise AssertionError("SIGTERM not handled by terminable yet")

    prev = signal.signal(signal.SIGTERM, _sentinel)
    try:
        with pytest.raises(SystemExit) as exc:
            with shutdown.terminable():
                try:
                    os.kill(os.getpid(), signal.SIGTERM)
                    time.sleep(0.5)  # deterministic delivery; handler interrupts
                finally:
                    cleaned["ran"] = True
        assert exc.value.code == 143
        assert cleaned["ran"] is True
    finally:
        signal.signal(signal.SIGTERM, prev)


def test_terminable_restores_previous_handler():
    before = signal.getsignal(signal.SIGTERM)
    with shutdown.terminable():
        assert signal.getsignal(signal.SIGTERM) is not before
    assert signal.getsignal(signal.SIGTERM) is before  # not leaked


def test_should_stop_flag_flips_on_signal():
    # raise_on_signal=False isolates the flag-flip from the SystemExit, so we can
    # assert should_stop directly without the handler raising.
    with shutdown.terminable(raise_on_signal=False) as guard:
        assert guard.should_stop is False
        guard._request_stop(signal.SIGTERM, None)  # simulate delivery
        assert guard.should_stop is True


def test_raise_on_signal_false_only_flips_flag():
    # Cooperative-drain mode: the handler flips should_stop but does NOT raise,
    # so the caller decides when to break out of its loop.
    prev = signal.signal(signal.SIGTERM, lambda *a: None)
    try:
        with shutdown.terminable(raise_on_signal=False) as guard:
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(0.3)
            assert guard.should_stop is True  # flag set, no SystemExit raised
    finally:
        signal.signal(signal.SIGTERM, prev)


def test_ingest_drains_on_sigterm_between_tickers(settings, catalog, monkeypatch):
    # End-to-end cooperative drain: a SIGTERM mid-run flips the terminable()
    # guard, and ingest() finishes the in-flight unit then returns a partial
    # summary (interrupted=True) instead of starting the remaining work.
    catalog.init_schema()
    import qtdata.ingestion.ingest as ing

    orig = ing._record_result
    calls = {"n": 0}

    def _record_then_signal(*args, **kwargs):
        orig(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] == 1:
            # terminable(raise_on_signal=False) -> guard.should_stop flips True
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(0.2)  # deterministic delivery before the next unit

    monkeypatch.setattr(ing, "_record_result", _record_then_signal)

    summary = ing.ingest(
        settings, catalog, ["AAA", "BBB", "CCC"],
        provider_name="synthetic", full_refresh=True,
    )
    assert summary.interrupted is True
    assert summary.ok >= 1 and summary.ok < 3  # drained before finishing all 3


def test_ingest_drains_between_batch_groups(settings, catalog, monkeypatch):
    # Exercises the BETWEEN-GROUPS guard (not just the inner per-ticker one):
    # give AAA a head start so AAA and BBB get different effective-start
    # signatures -> two separate batch groups. A SIGTERM after the first group's
    # work must stop ingest() before it dispatches the second group.
    from datetime import date

    from qtdata.models import Dataset

    catalog.init_schema()
    import qtdata.ingestion.ingest as ing

    # AAA already watermarked partway; BBB from scratch -> distinct group signatures
    ing.ingest(settings, catalog, ["AAA"], provider_name="synthetic", start=date(2024, 1, 2),
               end=date(2024, 1, 31), datasets=(Dataset.OHLCV_DAILY,))

    orig = ing._record_result
    calls = {"n": 0}

    def _record_then_signal(*args, **kwargs):
        orig(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] == 1:
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(0.2)

    monkeypatch.setattr(ing, "_record_result", _record_then_signal)

    summary = ing.ingest(settings, catalog, ["AAA", "BBB"], provider_name="synthetic",
                         end=date(2024, 2, 28), datasets=(Dataset.OHLCV_DAILY,))
    assert summary.interrupted is True
    assert summary.ok == 1  # only the first group's single ticker completed


def test_real_sigterm_midrun_preserves_committed_watermarks(settings, catalog, monkeypatch):
    # The TIL's core invariant: work committed before the stop survives, in-flight
    # work is left to retry. Send a real SIGTERM after the first ticker's result is
    # recorded, then assert its watermark persisted (resumable) and the run drained.
    from datetime import date

    from qtdata.ingestion.watermarks import get_watermark
    from qtdata.models import Dataset

    catalog.init_schema()
    import qtdata.ingestion.ingest as ing

    # Two groups (AAA head-started, BBB fresh) so AAA fully commits before BBB.
    ing.ingest(settings, catalog, ["AAA"], provider_name="synthetic", start=date(2024, 1, 2),
               end=date(2024, 1, 31), datasets=(Dataset.OHLCV_DAILY,))
    aaa_wm_before = get_watermark(catalog.conn, "synthetic", Dataset.OHLCV_DAILY, "AAA")
    assert aaa_wm_before is not None  # head start committed before the interrupted run

    orig = ing._record_result
    calls = {"n": 0}

    def _record_then_signal(*args, **kwargs):
        orig(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] == 1:
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(0.2)

    monkeypatch.setattr(ing, "_record_result", _record_then_signal)

    summary = ing.ingest(settings, catalog, ["AAA", "BBB"], provider_name="synthetic",
                         end=date(2024, 2, 28), datasets=(Dataset.OHLCV_DAILY,))
    assert summary.interrupted is True

    # AAA's group ran first and committed: its watermark advanced past the head start.
    aaa_wm_after = get_watermark(catalog.conn, "synthetic", Dataset.OHLCV_DAILY, "AAA")
    assert aaa_wm_after is not None
    assert aaa_wm_after > aaa_wm_before  # committed work survived the stop

    # BBB never ran (drained between groups): no watermark -> retried on resume.
    assert get_watermark(catalog.conn, "synthetic", Dataset.OHLCV_DAILY, "BBB") is None


def test_catalog_close_persists_across_reopen(settings):
    # A clean Catalog.close() must flush so a fresh (read-only) open sees the data
    # and finds no dangling lock — the durability guarantee a SIGTERM-unwound run
    # relies on when its `with Catalog(...)` block exits.
    from qtdata.storage.catalog import Catalog

    cat = Catalog(settings)
    cat.init_schema()
    cat.conn.execute(
        "INSERT OR REPLACE INTO watermarks VALUES "
        "('synthetic','ohlcv_daily','AAA','2026-01-02','run1', current_timestamp)"
    )
    cat.close()

    cat2 = Catalog(settings, read_only=True)
    row = cat2.conn.execute(
        "SELECT high_water_date FROM watermarks WHERE ticker='AAA'"
    ).fetchone()
    cat2.close()
    assert row is not None
