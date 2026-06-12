from datetime import date

from qtdata.ingestion.ingest import ingest
from qtdata.providers.synthetic_provider import SyntheticProvider
from qtdata.reconciliation.reconcile import load_raw_frame, reconcile
from tests.conftest import make_ohlcv


def test_identical_frames_have_no_discrepancies():
    df = make_ohlcv(n=50, with_lineage=False)
    result = reconcile(df, df.copy(), "a", "b")
    assert result.discrepancies.empty
    assert result.summary.empty


def test_single_perturbation_beyond_tolerance_yields_one_discrepancy():
    df_a = make_ohlcv(n=50, with_lineage=False)
    df_b = df_a.copy()
    df_b.loc[10, "close"] *= 1.02  # 2% off, tolerance 0.5%
    result = reconcile(df_a, df_b, "a", "b", price_rel_tol=0.005)
    disc = result.discrepancies[result.discrepancies["classification"] == "discrepancy"]
    assert len(disc) == 1
    assert disc.iloc[0]["field"] == "close"
    assert disc.iloc[0]["date"] == df_a.loc[10, "date"]


def test_within_tolerance_is_classified_not_flagged_as_discrepancy():
    df_a = make_ohlcv(n=50, with_lineage=False)
    df_b = df_a.copy()
    df_b.loc[10, "close"] *= 1.002  # 0.2% off, within 0.5%
    result = reconcile(df_a, df_b, "a", "b", price_rel_tol=0.005)
    assert (result.discrepancies["classification"] == "within_tolerance").all()


def test_missing_rows_detected_on_both_sides():
    df_a = make_ohlcv(n=50, with_lineage=False)
    df_b = df_a.drop(index=5).copy()
    df_a2 = df_a.drop(index=7)
    result = reconcile(df_a2, df_b, "a", "b")
    classes = set(result.discrepancies["classification"])
    assert "missing_in_b" in classes
    assert "missing_in_a" in classes


def test_load_raw_frame_takes_latest_ingest(settings, catalog, monkeypatch):
    provider = SyntheticProvider(seed=9)
    monkeypatch.setattr("qtdata.ingestion.ingest.get_provider", lambda name, s: provider)
    ingest(settings, catalog, ["AAA"], start=date(2024, 1, 2), end=date(2024, 2, 28))
    ingest(
        settings, catalog, ["AAA"], start=date(2024, 1, 2), end=date(2024, 2, 28),
        full_refresh=True,
    )
    df = load_raw_frame(settings, "synthetic")
    assert not df.empty
    assert not df.duplicated(subset=["ticker", "date"]).any()


def test_end_to_end_cross_provider_reconciliation(settings, catalog, monkeypatch):
    """Two providers, one with a perturbed close: exactly that discrepancy surfaces."""
    base = SyntheticProvider(seed=10)
    perturbed = SyntheticProvider(seed=10)
    real_fetch = perturbed.fetch_ohlcv

    def tweak(ticker, start, end):
        res = real_fetch(ticker, start, end)
        res.df.loc[res.df.index[5], "close"] *= 1.05
        # keep OHLC invariants intact so the comparison isolates the close
        res.df.loc[res.df.index[5], "high"] = max(
            res.df.loc[res.df.index[5], "high"], res.df.loc[res.df.index[5], "close"]
        )
        return res

    perturbed.fetch_ohlcv = tweak
    perturbed.name = "synthetic_b"

    for prov in (base, perturbed):
        monkeypatch.setattr("qtdata.ingestion.ingest.get_provider", lambda name, s, p=prov: p)
        ingest(settings, catalog, ["AAA"], start=date(2024, 1, 2), end=date(2024, 2, 28))

    df_a = load_raw_frame(settings, "synthetic")
    df_b = load_raw_frame(settings, "synthetic_b")
    result = reconcile(df_a, df_b, "synthetic", "synthetic_b")
    disc = result.discrepancies[result.discrepancies["classification"] == "discrepancy"]
    assert set(disc["field"]) <= {"close", "high"}
    assert len(disc[disc["field"] == "close"]) == 1
