from datetime import date

import numpy as np
import pandas as pd
import pytest

from qtdata.agents.llm import LLMClient
from qtdata.agents.report import CLOSING_LINE, generate_report
from qtdata.curation.curate import curate_all
from qtdata.fundamentals import ingest_screener_csv
from qtdata.ingestion.ingest import ingest
from qtdata.providers.synthetic_provider import SyntheticProvider
from qtdata.storage.catalog import Catalog
from tests.fake_anthropic import FakeAnthropic, response, text_block


@pytest.fixture
def seeded(settings, catalog, monkeypatch, tmp_path):
    # curated synthetic prices for AAPL + fundamentals snapshot
    provider = SyntheticProvider(seed=21)
    monkeypatch.setattr("qtdata.ingestion.ingest.get_provider", lambda name, s: provider)
    ingest(settings, catalog, ["AAPL"], start=date(2024, 1, 2), end=date(2024, 12, 30))
    curate_all(settings, catalog)

    csv = tmp_path / "mini.csv"
    pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "name": ["Apple Inc."],
            "sector": ["Technology"],
            "industry": ["Consumer Electronics"],
            "country": ["United States"],
            "exchange": ["NASDAQ"],
            "marketCapCategory": ["Mega-Cap"],
            "marketCap": ["3000000000000"],
            "peRatio": ["29.1"],
            "roe": ["1.47"],
            "debtEquity": ["1.45"],
            "dividendYield": ["0.0042"],
            "analystRatings": ["Buy"],
            "priceTarget": ["250.0"],
            "analystCount": ["41"],
        }
    ).to_csv(csv, index=False)
    ingest_screener_csv(settings, catalog, csv, as_of=date(2026, 5, 22))
    catalog.close()
    ro = Catalog(settings, read_only=True)
    yield ro
    ro.close()


def test_report_deterministic_sections(settings, seeded):
    out = generate_report(settings, "aapl", catalog_ro=seeded)
    text = out.read_text(encoding="utf-8")

    assert out.name == "report_AAPL.md"
    assert "Apple Inc. (AAPL)" in text
    # every figure cites its column
    assert "`fundamentals_snapshot.peRatio`" in text
    assert "29.1" in text
    # missing columns print n/d instead of inventing
    assert "n/d" in text
    # momentum computed from OUR adjusted prices
    assert "`ohlcv_daily_adj.close`" in text
    assert "Retorno 1M" in text
    # third-party labeling + closing line + no-LLM fallback
    assert "TERCEROS" in text
    assert CLOSING_LINE in text
    assert "sin API key" in text  # LLM section cleanly omitted offline


def test_report_momentum_matches_hand_computation(settings, seeded):
    generate_report(settings, "AAPL", catalog_ro=seeded)
    df = seeded.query(
        "SELECT close FROM ohlcv_daily_adj WHERE ticker='AAPL' ORDER BY date"
    )
    close = df["close"]
    expected_1m = close.iloc[-1] / close.iloc[-22] - 1.0
    text = (settings.reports_dir / "report_AAPL.md").read_text(encoding="utf-8")
    line = next(ln for ln in text.splitlines() if "Retorno 1M" in ln)
    shown = float(line.split(":")[-1].strip().rstrip("%").replace("+", "")) / 100
    assert np.isclose(shown, expected_1m, atol=5e-4)


def test_report_with_mocked_llm_judgment(settings, seeded):
    fake = FakeAnthropic()
    fake.messages.create_queue = [
        response([text_block("ROE excepcional con balance apalancado; tensión clásica.")],
                 stop_reason="end_turn"),
    ]
    llm = LLMClient(settings, client=fake)
    out = generate_report(settings, "AAPL", llm=llm, catalog_ro=seeded)
    text = out.read_text(encoding="utf-8")
    assert "tensión clásica" in text
    assert llm.meter.calls == 1
    # the LLM received the cited data block, not raw tables
    sent = fake.messages.create_calls[0]["messages"][0]["content"]
    assert "`fundamentals_snapshot.roe`" in sent


def test_report_unknown_ticker_raises(settings, seeded):
    with pytest.raises(ValueError, match="fundamentals_snapshot"):
        generate_report(settings, "NOPE", catalog_ro=seeded)
