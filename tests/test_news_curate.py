import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from qtdata.news.curate import curate_news
from qtdata.news.ingest import ingest_news
from qtdata.providers.alpha_vantage_news import parse_feed
from qtdata.storage import parquet_store

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "av_news_sample.json").read_text(encoding="utf-8")
)


@pytest.fixture
def ingested(settings, catalog, monkeypatch):
    monkeypatch.setattr(
        "qtdata.providers.alpha_vantage_news.AlphaVantageNewsProvider.fetch_news_day",
        lambda self, day, page_limit: (parse_feed(FIXTURE["feed"]), 1),
    )
    ingest_news(settings, catalog, date_from=date(2026, 6, 10), date_to=date(2026, 6, 10))
    return settings, catalog


def test_curate_splits_articles_and_ticker_rows(ingested):
    settings, catalog = ingested
    summary = curate_news(settings, catalog)
    assert summary.files_processed == 1

    articles = parquet_store.read(settings.curated_dir / "news_articles")
    assert len(articles) == 3  # incl. the macro article without tickers
    assert articles["article_id"].is_unique
    assert {"published_at", "ingested_at"} <= set(articles.columns)  # dual PIT stamps

    tickers = parquet_store.read(settings.curated_dir / "news_ticker_sentiment")
    assert len(tickers) == 3  # AAPL, MSFT, NVDA — null-ticker row excluded
    assert set(tickers["ticker"]) == {"AAPL", "MSFT", "NVDA"}
    assert tickers["score_finbert"].isna().all()  # scoring comes later
    # published_at duplicated onto ticker rows for cutoff math without a join
    assert "published_at" in tickers.columns


def test_dedupe_keeps_first_capture(settings, catalog, monkeypatch):
    base = parse_feed(FIXTURE["feed"])

    edited = base.copy()
    edited["title"] = edited["title"] + " (EDITADO)"

    calls = {"n": 0}

    def fetch(self, day, page_limit):
        calls["n"] += 1
        return (base if calls["n"] == 1 else edited), 1

    monkeypatch.setattr(
        "qtdata.providers.alpha_vantage_news.AlphaVantageNewsProvider.fetch_news_day",
        fetch,
    )
    ingest_news(settings, catalog, date_from=date(2026, 6, 10), date_to=date(2026, 6, 10))
    curate_news(settings, catalog)
    # vendor re-serves the same urls with edited titles next day
    ingest_news(settings, catalog, date_from=date(2026, 6, 11), date_to=date(2026, 6, 11))
    curate_news(settings, catalog)

    articles = parquet_store.read(settings.curated_dir / "news_articles")
    assert len(articles) == 3  # no duplicates
    # FIRST capture wins: the edited re-serve must NOT rewrite history
    assert not articles["title"].str.contains("EDITADO").any()


def test_curate_is_idempotent(ingested):
    settings, catalog = ingested
    curate_news(settings, catalog)
    n1 = len(parquet_store.read(settings.curated_dir / "news_ticker_sentiment"))
    again = curate_news(settings, catalog)
    assert again.files_processed == 0
    assert len(parquet_store.read(settings.curated_dir / "news_ticker_sentiment")) == n1


def test_bad_rows_quarantined(settings, catalog, monkeypatch):
    poisoned = parse_feed(FIXTURE["feed"])
    poisoned.loc[poisoned["ticker"] == "NVDA", "score_av"] = 7.5  # out of [-1, 1]
    monkeypatch.setattr(
        "qtdata.providers.alpha_vantage_news.AlphaVantageNewsProvider.fetch_news_day",
        lambda self, day, page_limit: (poisoned, 1),
    )
    ingest_news(settings, catalog, date_from=date(2026, 6, 10), date_to=date(2026, 6, 10))
    summary = curate_news(settings, catalog)
    assert summary.rows_quarantined >= 1
    tickers = parquet_store.read(settings.curated_dir / "news_ticker_sentiment")
    assert "NVDA" not in set(tickers["ticker"])
    assert pd.Series(tickers["score_av"]).between(-1, 1).all()
