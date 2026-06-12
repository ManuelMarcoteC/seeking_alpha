"""yfinance news harvester — redundancy source, FORWARD collection only.

Yahoo exposes only the ~recent stream per ticker (no archive, no date range):
this provider is useful exclusively as a daily harvester persisted with our own
ingest timestamp. Items use the post-v0.2.50 format: {'id', 'content': {...}}.
No vendor sentiment: relevance and score_av stay null (FinBERT fills the score).
"""

from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

from qtdata.config import Settings
from qtdata.providers.alpha_vantage_news import DENORMALIZED_COLUMNS, article_id_for
from qtdata.providers.base import RateLimiter, retry_transient

logger = logging.getLogger(__name__)


def parse_items(items: list[dict], ticker: str) -> pd.DataFrame:
    rows: list[dict] = []
    for item in items:
        content = item.get("content") or {}
        url = ((content.get("canonicalUrl") or {}).get("url")
               or (content.get("clickThroughUrl") or {}).get("url") or "")
        title = content.get("title", "")
        if not url or not title:
            continue
        provider_info = content.get("provider") or {}
        rows.append(
            {
                "article_id": article_id_for(url),
                "published_at": pd.to_datetime(content.get("pubDate"), utc=True,
                                               errors="coerce"),
                "source": provider_info.get("displayName", ""),
                "provider": "yfinance_news",
                "title": title,
                "summary": content.get("summary") or content.get("description") or "",
                "url": url,
                "overall_sentiment_score": None,
                "ticker": ticker.upper(),
                "relevance": None,
                "score_av": None,
            }
        )
    df = pd.DataFrame(rows, columns=DENORMALIZED_COLUMNS)
    return df.dropna(subset=["published_at"]).reset_index(drop=True)


class YFinanceNewsProvider:
    name = "yfinance_news"

    def __init__(self, settings: Settings):
        self._limiter = RateLimiter(settings.yfinance_rate_limit_per_min)

    @retry_transient
    def _news(self, ticker: str, count: int) -> list[dict]:
        self._limiter.acquire()
        return yf.Ticker(ticker).get_news(count=count) or []

    def fetch_news(self, ticker: str, count: int = 50) -> pd.DataFrame:
        try:
            items = self._news(ticker, count)
        except Exception:  # noqa: BLE001 — scraper source; isolate per ticker
            logger.warning("yfinance news failed for %s", ticker, exc_info=True)
            return pd.DataFrame(columns=DENORMALIZED_COLUMNS)
        return parse_items(items, ticker)
