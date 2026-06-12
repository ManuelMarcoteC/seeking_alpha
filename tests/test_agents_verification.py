from datetime import date

import pandas as pd
import pytest

from qtdata.agents.screener import Candidate, MetricCitation, Proposal
from qtdata.agents.verification import verify_proposal
from qtdata.fundamentals import ingest_screener_csv


def _proposal(entries):
    return Proposal(
        candidates=[
            Candidate(
                ticker=t,
                thesis="x",
                metrics=[MetricCitation(column=c, source_view=v, value=val)],
            )
            for t, c, v, val in entries
        ],
        methodology="m",
        caveats=[],
    )


@pytest.fixture
def seeded(settings, catalog, tmp_path):
    csv = tmp_path / "mini.csv"
    pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "GOOG", "NVDA", "META", "XOM"],
            "sector": ["Technology"] * 5 + ["Energy"],
            "roe": ["1.4", "0.4", "0.3", "0.9", "0.35", "0.2"],
        }
    ).to_csv(csv, index=False)
    ingest_screener_csv(settings, catalog, csv, as_of=date(2026, 5, 22))
    return catalog


def test_hallucinated_ticker_detected(settings, seeded):
    p = _proposal([("AAPL", "roe", "fundamentals_snapshot", 1.4),
                   ("GHOST", "roe", "fundamentals_snapshot", 9.9)])
    report = verify_proposal(seeded, p)
    check = next(c for c in report.checks if c.name == "tickers existen")
    assert not check.passed
    assert "GHOST" in check.detail


def test_sector_concentration_flagged(settings, seeded):
    p = _proposal([("AAPL", "roe", "fundamentals_snapshot", 1.4),
                   ("MSFT", "roe", "fundamentals_snapshot", 0.4),
                   ("GOOG", "roe", "fundamentals_snapshot", 0.3),
                   ("NVDA", "roe", "fundamentals_snapshot", 0.9),
                   ("META", "roe", "fundamentals_snapshot", 0.35)])
    report = verify_proposal(seeded, p, max_sector_pct=0.40)
    check = next(c for c in report.checks if "concentración" in c.name)
    assert not check.passed  # 100% Technology


def test_sector_concentration_skipped_for_small_books(settings, seeded):
    p = _proposal([("AAPL", "roe", "fundamentals_snapshot", 1.4),
                   ("XOM", "roe", "fundamentals_snapshot", 0.2)])
    report = verify_proposal(seeded, p, max_sector_pct=0.40)
    assert not any("concentración" in c.name for c in report.checks)


def test_nonexistent_column_flagged(settings, seeded):
    p = _proposal([("AAPL", "made_up_metric", "fundamentals_snapshot", 1.0)])
    report = verify_proposal(seeded, p)
    check = next(c for c in report.checks if c.name == "columnas citadas existen")
    assert not check.passed
    assert "made_up_metric" in check.detail


def test_numeric_spot_check_catches_fabricated_value(settings, seeded):
    p = _proposal([("AAPL", "roe", "fundamentals_snapshot", 99.0)])  # real is 1.4
    report = verify_proposal(seeded, p)
    check = next(c for c in report.checks if "spot-check" in c.name)
    assert not check.passed
    assert "AAPL.roe" in check.detail


def test_clean_proposal_passes(settings, seeded):
    p = _proposal([("AAPL", "roe", "fundamentals_snapshot", 1.4),
                   ("XOM", "roe", "fundamentals_snapshot", 0.2)])
    report = verify_proposal(seeded, p, max_sector_pct=0.60)
    assert report.passed
    assert "✓" in report.render()
