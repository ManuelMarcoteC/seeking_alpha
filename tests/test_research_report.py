from datetime import date

import pandas as pd
import pytest
from typer.testing import CliRunner

import qtdata.cli as cli
from qtdata.config import Settings
from qtdata.models import FACTORS_KEY, OHLCV_KEY, SENTIMENT_DAILY_KEY
from qtdata.research.ic import ICSummary
from qtdata.research.report import SentimentValidationReport, persist_research_report
from qtdata.research.sentiment_validation import run_sentiment_validation
from qtdata.storage import parquet_store
from tests.conftest import make_ohlcv

runner = CliRunner()


def test_persist_report_structure_and_standing_caveats(settings):
    report = SentimentValidationReport(
        run_id="abc123",
        params={"score_col": "sent_av", "horizons": [1, 5]},
        ic=[ICSummary(1, 10, 0.03, 0.1, 0.95, 0.3, 0.6)],
        events=None,
        n_factor_days=10,
        n_tickers=4,
        caveats=["caveat propio"],
    )
    out = persist_research_report(report, settings)
    text = out.read_text(encoding="utf-8")
    assert "IC de Spearman" in text
    assert "| 1 | 10 | 0.0300 |" in text
    assert "Sin eventos" in text
    assert "## Caveats" in text
    assert "caveat propio" in text
    assert "Newey-West" in text  # standing caveats always present
    assert report.path == out


def _seed_market(settings, catalog, tickers: list[str], n: int = 40) -> pd.DataFrame:
    closes = []
    for i, t in enumerate(tickers):
        df = make_ohlcv(t, n=n, seed=i)
        df["year"] = pd.to_datetime(df["date"]).dt.year
        parquet_store.upsert(
            df, settings.curated_dir / "ohlcv_daily", OHLCV_KEY, partition_col="year"
        )
        closes.append(df[["ticker", "date", "close"]])
    adj = pd.DataFrame(
        {
            "ticker": tickers,
            "date": pd.Timestamp("2024-01-02"),
            "adj_factor": 1.0,
            "split_factor": 1.0,
        }
    )
    adj["year"] = 2024
    parquet_store.upsert(
        adj, settings.curated_dir / "adjustment_factors", FACTORS_KEY, partition_col="year"
    )
    return pd.concat(closes, ignore_index=True)


def _seed_sentiment(settings, closes: pd.DataFrame) -> None:
    rows = closes.groupby("ticker").head(20).copy()
    rows = rows.rename(columns={"close": "_c"}).drop(columns=["_c"])
    rows["sent_av"] = 0.4
    rows["sent_finbert"] = None
    rows["n_articles"] = 4
    rows["log_n_articles"] = 1.6
    rows["rel_sum"] = 2.0
    rows["built_at"] = pd.Timestamp.now(tz="UTC")
    rows["year"] = pd.to_datetime(rows["date"]).dt.year
    parquet_store.upsert(
        rows, settings.curated_dir / "sentiment_daily", SENTIMENT_DAILY_KEY,
        partition_col="year",
    )


def test_run_sentiment_validation_end_to_end(settings, catalog):
    tickers = [f"T{i:02d}" for i in range(12)]
    closes = _seed_market(settings, catalog, tickers)
    _seed_sentiment(settings, closes)
    catalog.refresh_views()

    report = run_sentiment_validation(
        settings, catalog, horizons=(1, 5), min_breadth=10, start=date(2024, 1, 2)
    )
    assert report.path is not None and report.path.exists()
    assert [s.horizon for s in report.ic] == [1, 5]
    # constant score -> Spearman undefined -> every day skipped, reported as n/d
    assert report.ic[0].n_days == 0
    # sent_finbert all-null degrades to sent_av with a recorded caveat
    assert any("sent_av" in c for c in report.caveats)
    assert report.n_tickers == 12


def test_cli_sentiment_ic_without_data_exits_cleanly(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path / "data", _env_file=None)
    monkeypatch.setattr(cli, "get_settings", lambda: s)
    runner.invoke(cli.app, ["init"])
    result = runner.invoke(cli.app, ["research", "sentiment-ic"])
    assert result.exit_code == 1
    assert "sentiment_daily" in result.output


def test_run_validation_requires_factor(settings, catalog):
    with pytest.raises(ValueError, match="sentiment_daily"):
        run_sentiment_validation(settings, catalog)
