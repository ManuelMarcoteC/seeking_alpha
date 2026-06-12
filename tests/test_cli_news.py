import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import qtdata.cli as cli
from qtdata.config import Settings
from qtdata.providers.alpha_vantage_news import parse_feed
from qtdata.storage import parquet_store

runner = CliRunner()

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "av_news_sample.json").read_text(encoding="utf-8")
)


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path / "data", alpha_vantage_api_key="k", _env_file=None)
    monkeypatch.setattr(cli, "get_settings", lambda: s)
    monkeypatch.setattr(
        "qtdata.providers.alpha_vantage_news.AlphaVantageNewsProvider.fetch_news_day",
        lambda self, day, page_limit: (parse_feed(FIXTURE["feed"]), 1),
    )
    return s


def test_news_ingest_requires_av_key(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path / "data", _env_file=None)
    monkeypatch.setattr(cli, "get_settings", lambda: s)
    runner.invoke(cli.app, ["init"])
    result = runner.invoke(cli.app, ["news", "ingest"])
    assert result.exit_code == 1
    assert "QT_ALPHA_VANTAGE_API_KEY" in result.output


def test_news_ingest_and_curate_via_cli(isolated_settings):
    runner.invoke(cli.app, ["init"])
    result = runner.invoke(
        cli.app, ["news", "ingest", "--from", "2026-06-10", "--to", "2026-06-10"]
    )
    assert result.exit_code == 0, result.output
    assert "ok=1" in result.output

    result = runner.invoke(cli.app, ["news", "curate"])
    assert result.exit_code == 0, result.output
    assert "files=1" in result.output

    articles = parquet_store.read(isolated_settings.curated_dir / "news_articles")
    assert len(articles) == 3
    rows = parquet_store.read(isolated_settings.curated_dir / "news_ticker_sentiment")
    assert {"AAPL", "MSFT"} <= set(rows["ticker"].dropna())


def test_news_curate_with_nothing_pending(isolated_settings):
    runner.invoke(cli.app, ["init"])
    result = runner.invoke(cli.app, ["news", "curate"])
    assert result.exit_code == 0, result.output
    assert "files=0" in result.output
