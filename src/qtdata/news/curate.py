"""Raw news -> curated `news_articles` + `news_ticker_sentiment`.

PIT discipline: every row carries BOTH published_at and ingested_at; the layer
is append-only with FIRST-capture-wins dedupe (the opposite of OHLCV's
latest-wins, deliberately: the first observation is the point-in-time fact —
later re-pulls of an edited article must not rewrite history). The only later
write is the FinBERT scoring upsert, which fills score columns on identical keys.
"""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4

import pandas as pd

from qtdata.config import Settings
from qtdata.curation.curate import CurationSummary
from qtdata.models import NEWS_ARTICLES_KEY, NEWS_TICKER_KEY
from qtdata.storage import parquet_store
from qtdata.storage.catalog import Catalog
from qtdata.validation.report import persist_quarantine
from qtdata.validation.schemas import NEWS_ARTICLES_SCHEMA, NEWS_TICKER_SCHEMA, validate_frame

logger = logging.getLogger(__name__)

ARTICLE_COLUMNS = [
    "article_id", "published_at", "ingested_at", "source", "provider",
    "title", "summary", "url", "overall_sentiment_score", "run_id",
]
TICKER_COLUMNS = [
    "article_id", "ticker", "published_at", "ingested_at", "relevance", "score_av",
    "score_finbert", "finbert_revision", "scored_at", "run_id",
]


def _uncurated_news_files(settings: Settings, catalog: Catalog) -> list[Path]:
    files = sorted(settings.raw_dir.glob("provider=*/dataset=news/date=*/*.parquet"))
    return [f for f in files if not catalog.is_file_curated(f)]


def _existing_keys(settings: Settings, table: str, cols: list[str]) -> pd.DataFrame:
    existing = parquet_store.read(settings.curated_dir / table, columns=cols)
    if existing.empty:
        return pd.DataFrame(columns=cols)
    return existing


def curate_news(settings: Settings, catalog: Catalog) -> CurationSummary:
    run_id = uuid4().hex[:12]
    summary = CurationSummary(run_id=run_id)
    files = _uncurated_news_files(settings, catalog)
    if not files:
        return summary

    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    raw["published_at"] = pd.to_datetime(raw["published_at"], utc=True)
    raw["ingested_at"] = pd.to_datetime(raw["ingested_at"], utc=True)

    # ---- articles: one row per article_id, first capture wins -------------
    articles = (
        raw.sort_values("ingested_at")
        .drop_duplicates(subset=NEWS_ARTICLES_KEY, keep="first")
        .loc[:, [c for c in ARTICLE_COLUMNS if c in raw.columns]]
        .reset_index(drop=True)
    )
    already = set(_existing_keys(settings, "news_articles", ["article_id"])["article_id"])
    articles = articles[~articles["article_id"].isin(already)].reset_index(drop=True)

    valid_articles, art_failures = validate_frame(articles, NEWS_ARTICLES_SCHEMA)
    persist_quarantine(art_failures, f"{run_id}_articles", settings)
    if not valid_articles.empty:
        out = valid_articles.copy()
        out["year"] = out["published_at"].dt.year
        parquet_store.upsert(
            out, settings.curated_dir / "news_articles", NEWS_ARTICLES_KEY,
            partition_col="year",
        )
        summary.rows_upserted += len(out)

    # ---- ticker sentiment rows: (article_id, ticker), first capture wins --
    tickers = raw.dropna(subset=["ticker"]).copy()
    tickers["ticker"] = tickers["ticker"].astype(str).str.upper()
    tickers = (
        tickers.sort_values("ingested_at")
        .drop_duplicates(subset=NEWS_TICKER_KEY, keep="first")
        .reset_index(drop=True)
    )
    tickers["score_finbert"] = pd.Series([None] * len(tickers), dtype="float64")
    tickers["finbert_revision"] = pd.Series([None] * len(tickers), dtype="object")
    tickers["scored_at"] = pd.Series([pd.NaT] * len(tickers), dtype="datetime64[ns, UTC]")
    tickers = tickers.loc[:, TICKER_COLUMNS]

    existing_pairs = _existing_keys(
        settings, "news_ticker_sentiment", ["article_id", "ticker"]
    )
    if not existing_pairs.empty:
        merged = tickers.merge(
            existing_pairs.assign(_seen=True), on=NEWS_TICKER_KEY, how="left"
        )
        tickers = tickers[merged["_seen"].isna().to_numpy()].reset_index(drop=True)

    valid_tickers, tick_failures = validate_frame(tickers, NEWS_TICKER_SCHEMA)
    persist_quarantine(tick_failures, f"{run_id}_tickers", settings)
    if not valid_tickers.empty:
        out = valid_tickers.copy()
        out["year"] = out["published_at"].dt.year
        parquet_store.upsert(
            out, settings.curated_dir / "news_ticker_sentiment", NEWS_TICKER_KEY,
            partition_col="year",
        )
        summary.rows_upserted += len(out)
        summary.tickers = sorted(set(out["ticker"]))

    for f in files:
        catalog.mark_file_curated(f)
    summary.files_processed = len(files)
    n_quarantined = 0
    for failures in (art_failures, tick_failures):
        if not failures.empty:
            n_quarantined += len(failures["index"].unique())
    summary.rows_quarantined = n_quarantined
    catalog.refresh_views()
    return summary
