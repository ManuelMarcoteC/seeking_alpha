"""The PIT-critical tests: cutoff mapping and weighted aggregation math."""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qtdata.news.aggregate import build_sentiment_daily, trading_day_for
from qtdata.news.curate import curate_news
from qtdata.news.ingest import ingest_news
from qtdata.providers.alpha_vantage_news import parse_feed
from qtdata.storage import parquet_store

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "av_news_sample.json").read_text(encoding="utf-8")
)


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="America/New_York").tz_convert("UTC")


class TestTradingDayFor:
    def test_before_cutoff_maps_to_same_session(self):
        # Tuesday 2026-06-09 15:29 ET -> same day
        assert trading_day_for(_ts("2026-06-09 15:29")) == pd.Timestamp("2026-06-09")

    def test_after_cutoff_maps_to_next_session(self):
        # Tuesday 16:00 ET -> Wednesday
        assert trading_day_for(_ts("2026-06-09 16:00")) == pd.Timestamp("2026-06-10")

    def test_saturday_maps_to_monday(self):
        assert trading_day_for(_ts("2026-06-13 10:00")) == pd.Timestamp("2026-06-15")

    def test_friday_after_cutoff_maps_to_monday(self):
        assert trading_day_for(_ts("2026-06-12 17:00")) == pd.Timestamp("2026-06-15")

    def test_holiday_maps_to_next_session(self):
        # July 3 2026 is the observed Independence Day holiday (July 4 is Saturday)
        assert trading_day_for(_ts("2026-07-03 10:00")) == pd.Timestamp("2026-07-06")

    def test_ingested_at_dominates_published_at(self):
        # an article published years ago but ingested today is only tradable today:
        # the caller passes effective_ts = max(published, ingested)
        published = _ts("2020-01-02 10:00")
        ingested = _ts("2026-06-09 10:00")
        effective = max(published, ingested)
        assert trading_day_for(effective) == pd.Timestamp("2026-06-09")


def test_build_sentiment_daily_weighted_mean(settings, catalog, monkeypatch):
    monkeypatch.setattr(
        "qtdata.providers.alpha_vantage_news.AlphaVantageNewsProvider.fetch_news_day",
        lambda self, day, page_limit: (parse_feed(FIXTURE["feed"]), 1),
    )
    ingest_news(settings, catalog, date_from=date(2026, 6, 10), date_to=date(2026, 6, 10))
    curate_news(settings, catalog)
    n = build_sentiment_daily(settings, catalog)
    assert n >= 1

    daily = parquet_store.read(settings.curated_dir / "sentiment_daily")
    # MSFT row has relevance 0.12 < floor 0.25 -> dropped entirely
    assert "MSFT" not in set(daily["ticker"])

    aapl = daily[daily["ticker"] == "AAPL"].iloc[0]
    # single article: weighted mean == its score
    assert aapl["sent_av"] == pytest.approx(0.42)
    assert aapl["n_articles"] == 1
    assert aapl["log_n_articles"] == pytest.approx(np.log1p(1))
    assert pd.isna(aapl["sent_finbert"])  # not scored yet

    # article published 2026-06-10 14:30 UTC = 10:30 ET (before cutoff, session day)
    # but ingested_at (now, 2026) dominates -> attributed to a session >= ingestion
    eff_expected = max(
        pd.Timestamp("2026-06-10T14:30:00Z"),
        pd.Timestamp.now(tz="UTC"),
    )
    expected_day = trading_day_for(eff_expected)
    assert pd.Timestamp(aapl["date"]) == expected_day


def test_weighted_mean_multiple_articles(settings, catalog, monkeypatch):
    feed = [
        {
            "title": "a", "url": "https://x/a", "time_published": "20260610T120000",
            "source": "s", "summary": "", "overall_sentiment_score": 0.0,
            "ticker_sentiment": [
                {"ticker": "AAPL", "relevance_score": "0.8", "ticker_sentiment_score": "0.5"}
            ],
        },
        {
            "title": "b", "url": "https://x/b", "time_published": "20260610T130000",
            "source": "s", "summary": "", "overall_sentiment_score": 0.0,
            "ticker_sentiment": [
                {"ticker": "AAPL", "relevance_score": "0.4", "ticker_sentiment_score": "-0.25"}
            ],
        },
    ]
    monkeypatch.setattr(
        "qtdata.providers.alpha_vantage_news.AlphaVantageNewsProvider.fetch_news_day",
        lambda self, day, page_limit: (parse_feed(feed), 1),
    )
    ingest_news(settings, catalog, date_from=date(2026, 6, 10), date_to=date(2026, 6, 10))
    curate_news(settings, catalog)
    build_sentiment_daily(settings, catalog)

    daily = parquet_store.read(settings.curated_dir / "sentiment_daily")
    aapl = daily[daily["ticker"] == "AAPL"].iloc[0]
    expected = (0.8 * 0.5 + 0.4 * -0.25) / (0.8 + 0.4)
    assert aapl["sent_av"] == pytest.approx(expected)
    assert aapl["n_articles"] == 2
    assert aapl["rel_sum"] == pytest.approx(1.2)


def test_rebuild_is_idempotent(settings, catalog, monkeypatch):
    monkeypatch.setattr(
        "qtdata.providers.alpha_vantage_news.AlphaVantageNewsProvider.fetch_news_day",
        lambda self, day, page_limit: (parse_feed(FIXTURE["feed"]), 1),
    )
    ingest_news(settings, catalog, date_from=date(2026, 6, 10), date_to=date(2026, 6, 10))
    curate_news(settings, catalog)
    build_sentiment_daily(settings, catalog)
    first = parquet_store.read(settings.curated_dir / "sentiment_daily")
    build_sentiment_daily(settings, catalog)
    second = parquet_store.read(settings.curated_dir / "sentiment_daily")
    assert len(first) == len(second)
    pd.testing.assert_frame_equal(
        first.drop(columns=["built_at"]).sort_values(["ticker"]).reset_index(drop=True),
        second.drop(columns=["built_at"]).sort_values(["ticker"]).reset_index(drop=True),
    )
