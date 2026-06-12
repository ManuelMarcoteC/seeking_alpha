"""Alpha Vantage NEWS_SENTIMENT provider in FIREHOSE mode.

Free-tier strategy: 25 requests/day cannot query per ticker at NASDAQ scale, so
we omit the `tickers` param and page through each calendar day with
time_from/time_to windows (sort=EARLIEST, limit=1000) — up to ~23 pages/day ≈
23,000 articles, each carrying per-ticker sentiment and relevance that curation
explodes into rows. This is a FORWARD collector: historical pulls are scored by
the vendor's CURRENT model (not point-in-time) and belong in research only.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime, time, timedelta

import pandas as pd
import requests

from qtdata.config import Settings
from qtdata.models import ProviderNotConfiguredError
from qtdata.providers.base import RateLimiter, retry_transient

BASE_URL = "https://www.alphavantage.co/query"
_TIMEOUT = 30

DENORMALIZED_COLUMNS = [
    "article_id", "published_at", "source", "provider", "title", "summary", "url",
    "overall_sentiment_score", "ticker", "relevance", "score_av",
]


def article_id_for(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def parse_feed(feed: list[dict], provider: str = "alpha_vantage_news") -> pd.DataFrame:
    """NEWS_SENTIMENT feed[] -> denormalized rows (one per article x ticker).

    Articles with no ticker_sentiment keep a single row with null ticker.
    """
    rows: list[dict] = []
    for item in feed:
        url = item.get("url", "")
        if not url:
            continue
        base = {
            "article_id": article_id_for(url),
            "published_at": pd.to_datetime(
                item.get("time_published"), format="%Y%m%dT%H%M%S", utc=True,
                errors="coerce",
            ),
            "source": item.get("source", ""),
            "provider": provider,
            "title": item.get("title", ""),
            "summary": item.get("summary", ""),
            "url": url,
            "overall_sentiment_score": pd.to_numeric(
                item.get("overall_sentiment_score"), errors="coerce"
            ),
        }
        ticker_rows = item.get("ticker_sentiment") or []
        if not ticker_rows:
            rows.append({**base, "ticker": None, "relevance": None, "score_av": None})
            continue
        for ts in ticker_rows:
            rows.append(
                {
                    **base,
                    "ticker": str(ts.get("ticker", "")).upper() or None,
                    "relevance": pd.to_numeric(ts.get("relevance_score"), errors="coerce"),
                    "score_av": pd.to_numeric(
                        ts.get("ticker_sentiment_score"), errors="coerce"
                    ),
                }
            )
    df = pd.DataFrame(rows, columns=DENORMALIZED_COLUMNS)
    return df.dropna(subset=["published_at"]).reset_index(drop=True)


class AlphaVantageNewsProvider:
    name = "alpha_vantage_news"

    def __init__(self, settings: Settings):
        key = settings.alpha_vantage_api_key
        self._key = key.get_secret_value() if key else None
        self._limiter = RateLimiter(settings.alpha_vantage_rate_limit_per_min)

    def _require_key(self) -> None:
        if not self._key:
            raise ProviderNotConfiguredError(
                "Alpha Vantage news requires QT_ALPHA_VANTAGE_API_KEY (set it in .env)."
            )

    @retry_transient
    def _get(self, **params: str) -> dict:
        self._limiter.acquire()
        resp = requests.get(BASE_URL, params={**params, "apikey": self._key}, timeout=_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
        for key in ("Error Message", "Information", "Note"):
            if key in payload:
                raise RuntimeError(f"Alpha Vantage refused the request: {payload[key]}")
        return payload

    def fetch_news_day(self, day: date, page_limit: int) -> tuple[pd.DataFrame, int]:
        """Page through one UTC calendar day. Returns (denormalized frame, pages used)."""
        self._require_key()
        window_start = datetime.combine(day, time(0, 0))
        window_end = datetime.combine(day, time(23, 59))
        frames: list[pd.DataFrame] = []
        cursor = window_start
        pages = 0
        while pages < page_limit:
            payload = self._get(
                function="NEWS_SENTIMENT",
                time_from=cursor.strftime("%Y%m%dT%H%M"),
                time_to=window_end.strftime("%Y%m%dT%H%M"),
                sort="EARLIEST",
                limit="1000",
            )
            pages += 1
            feed = payload.get("feed", []) or []
            if not feed:
                break
            frames.append(parse_feed(feed, self.name))
            if len(feed) < 1000:
                break
            last_published = max(
                pd.to_datetime(
                    item.get("time_published"), format="%Y%m%dT%H%M%S", errors="coerce"
                )
                for item in feed
            )
            if pd.isna(last_published):
                break
            cursor = (last_published + timedelta(minutes=1)).to_pydatetime()
            if cursor > window_end:
                break

        if not frames:
            return pd.DataFrame(columns=DENORMALIZED_COLUMNS), pages
        out = pd.concat(frames, ignore_index=True)
        # pages overlap at minute granularity: first occurrence wins
        out = out.drop_duplicates(subset=["article_id", "ticker"], keep="first")
        return out.reset_index(drop=True), pages
