import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from qtdata.ingestion.watermarks import get_watermark
from qtdata.models import Dataset, ProviderNotConfiguredError
from qtdata.news.ingest import FIREHOSE_KEY, ingest_news
from qtdata.providers.alpha_vantage_news import (
    AlphaVantageNewsProvider,
    article_id_for,
    parse_feed,
)
from qtdata.providers.yfinance_news import parse_items

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "av_news_sample.json").read_text(encoding="utf-8")
)


def test_parse_feed_explodes_ticker_sentiment():
    df = parse_feed(FIXTURE["feed"])
    # 2 tickers (art 1) + 1 ticker (art 2) + 1 null-ticker row (art 3)
    assert len(df) == 4
    aapl = df[df["ticker"] == "AAPL"].iloc[0]
    assert aapl["relevance"] == pytest.approx(0.85)
    assert aapl["score_av"] == pytest.approx(0.42)
    assert aapl["published_at"] == pd.Timestamp("2026-06-10T14:30:00", tz="UTC")
    macro = df[df["ticker"].isna()]
    assert len(macro) == 1  # article without tickers preserved for the articles table


def test_article_id_is_stable_sha256_of_url():
    df = parse_feed(FIXTURE["feed"])
    url = "https://example.com/news/apple-ai-services-record"
    expected = article_id_for(url)
    assert (df[df["url"] == url]["article_id"] == expected).all()
    assert len(expected) == 64


def test_provider_requires_key(settings):
    provider = AlphaVantageNewsProvider(settings)
    with pytest.raises(ProviderNotConfiguredError):
        provider.fetch_news_day(date(2026, 6, 10), page_limit=2)


def test_firehose_pagination(settings, monkeypatch):
    monkeypatch.setenv("QT_ALPHA_VANTAGE_API_KEY", "k")
    from qtdata.config import Settings

    s = Settings(data_dir=settings.data_dir, _env_file=None)
    provider = AlphaVantageNewsProvider(s)

    # page 1: full page of 1000 articles -> paginate; page 2: short page -> stop
    def make_feed(n, start_minute):
        return [
            {
                "title": f"t{i}",
                "url": f"https://x.com/{start_minute}/{i}",
                "time_published": f"20260610T{8 + (start_minute + i) // 600:02d}"
                                  f"{(start_minute + i) % 60:02d}00",
                "source": "s",
                "summary": "",
                "overall_sentiment_score": 0.0,
                "ticker_sentiment": [],
            }
            for i in range(n)
        ]

    pages = [
        {"feed": make_feed(1000, 0)},
        {"feed": make_feed(137, 30)},
    ]
    calls = []

    def fake_get(**params):
        calls.append(params)
        return pages[len(calls) - 1]

    monkeypatch.setattr(provider, "_get", fake_get)
    frame, used = provider.fetch_news_day(date(2026, 6, 10), page_limit=23)
    assert used == 2
    assert calls[0]["sort"] == "EARLIEST"
    assert calls[0]["limit"] == "1000"
    assert calls[0]["time_from"] == "20260610T0000"
    # second page advances time_from past the last published article
    assert calls[1]["time_from"] > calls[0]["time_from"]
    assert len(frame) == 1000 + 137


def test_firehose_respects_page_limit(settings, monkeypatch):
    monkeypatch.setenv("QT_ALPHA_VANTAGE_API_KEY", "k")
    from qtdata.config import Settings

    s = Settings(data_dir=settings.data_dir, _env_file=None)
    provider = AlphaVantageNewsProvider(s)
    full = {
        "feed": [
            {
                "title": "t", "url": f"https://x.com/{i}",
                "time_published": "20260610T120000", "source": "s", "summary": "",
                "overall_sentiment_score": 0.0, "ticker_sentiment": [],
            }
            for i in range(1000)
        ]
    }
    monkeypatch.setattr(provider, "_get", lambda **kw: full)
    _, used = provider.fetch_news_day(date(2026, 6, 10), page_limit=3)
    assert used <= 3


def test_ingest_news_firehose_raw_layout_and_watermark(settings, catalog, monkeypatch):
    def fake_fetch(self, day, page_limit):
        return parse_feed(FIXTURE["feed"]), 1

    monkeypatch.setattr(
        "qtdata.providers.alpha_vantage_news.AlphaVantageNewsProvider.fetch_news_day",
        fake_fetch,
    )
    summary = ingest_news(
        settings, catalog, date_from=date(2026, 6, 10), date_to=date(2026, 6, 10)
    )
    assert summary.ok == 1
    raw = list(
        (settings.raw_dir / "provider=alpha_vantage_news" / "dataset=news").rglob("*.parquet")
    )
    assert len(raw) == 1
    assert raw[0].parent.name == "date=2026-06-10"
    wm = get_watermark(catalog.conn, "alpha_vantage_news", Dataset.NEWS, FIREHOSE_KEY)
    assert wm == date(2026, 6, 10)

    # second run resumes from watermark + 1 -> nothing to do for same date_to
    summary2 = ingest_news(settings, catalog, date_to=date(2026, 6, 10))
    assert summary2.skipped == 1


def test_yfinance_news_parse_new_format():
    items = [
        {
            "id": "abc",
            "content": {
                "title": "Apple ships new device",
                "summary": "Lots of demand.",
                "pubDate": "2026-06-10T14:30:00Z",
                "contentType": "STORY",
                "provider": {"displayName": "Reuters"},
                "canonicalUrl": {"url": "https://news.example/apple-device"},
            },
        },
        {"id": "ad", "content": {"title": "", "canonicalUrl": {"url": ""}}},  # dropped
    ]
    df = parse_items(items, "aapl")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["ticker"] == "AAPL"
    assert row["provider"] == "yfinance_news"
    assert row["source"] == "Reuters"
    assert pd.isna(row["relevance"]) and pd.isna(row["score_av"])
    assert row["published_at"] == pd.Timestamp("2026-06-10T14:30:00", tz="UTC")
