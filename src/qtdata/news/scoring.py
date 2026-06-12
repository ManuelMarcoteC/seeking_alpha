"""FinBERT scoring of curated headlines — the frozen, reproducible sentiment engine.

The model revision is PINNED (QT_FINBERT_REVISION): a frozen scorer means the
factor series is reproducible and free of the LLM look-ahead trap (a current
model scoring historical headlines 'knows' the outcomes). torch/transformers
are an optional install (requirements-sentiment.txt); imports stay lazy.

score = P(positive) - P(negative), in [-1, 1].
"""

from __future__ import annotations

import logging

import pandas as pd

from qtdata.config import Settings
from qtdata.models import NEWS_TICKER_KEY
from qtdata.storage import parquet_store
from qtdata.storage.catalog import Catalog

logger = logging.getLogger(__name__)

MODEL_NAME = "ProsusAI/finbert"
_INSTALL_HINT = (
    "FinBERT scoring requires the optional sentiment extras:\n"
    "  pip install -r requirements-sentiment.txt\n"
    "(torch CPU + transformers; ~2GB)"
)


def _load_finbert(revision: str):
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised via clear-error test
        raise ImportError(_INSTALL_HINT) from exc

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, revision=revision)
    model.eval()
    return tokenizer, model


def score_texts(texts: list[str], revision: str, batch_size: int = 32) -> list[float]:
    """Batch-score texts on CPU; returns P(pos) - P(neg) per text."""
    import torch

    tokenizer, model = _load_finbert(revision)
    # ProsusAI/finbert label order: positive, negative, neutral
    labels = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
    pos_idx, neg_idx = labels.index("positive"), labels.index("negative")

    scores: list[float] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=64
            )
            probs = torch.softmax(model(**inputs).logits, dim=-1)
            scores.extend((probs[:, pos_idx] - probs[:, neg_idx]).tolist())
    return scores


def score_pending(
    settings: Settings,
    catalog: Catalog,
    *,
    batch_size: int = 32,
    limit: int | None = None,
) -> int:
    """Score curated ticker rows lacking score_finbert. Returns rows scored.

    Only the score columns change on the upserted rows — everything else is
    written back byte-identical (the news layer never mutates observations).
    """
    table_dir = settings.curated_dir / "news_ticker_sentiment"
    rows = parquet_store.read(table_dir)
    if rows.empty:
        return 0
    pending = rows[rows["score_finbert"].isna()].copy()
    if pending.empty:
        return 0
    if limit is not None:
        pending = pending.head(limit).copy()

    articles = parquet_store.read(
        settings.curated_dir / "news_articles", columns=["article_id", "title", "summary"]
    )
    texts_by_id = {
        r["article_id"]: f"{r['title']}. {r['summary'] or ''}".strip()
        for _, r in articles.iterrows()
    }
    texts = [texts_by_id.get(aid, "") for aid in pending["article_id"]]

    revision = settings.finbert_revision
    scores = score_texts(texts, revision, batch_size=batch_size)

    pending["score_finbert"] = scores
    pending["finbert_revision"] = revision
    pending["scored_at"] = pd.Timestamp.now(tz="UTC")
    if "year" not in pending.columns:
        pending["year"] = pd.to_datetime(pending["published_at"], utc=True).dt.year

    parquet_store.upsert(pending, table_dir, NEWS_TICKER_KEY, partition_col="year")
    catalog.refresh_views()
    logger.info("FinBERT (rev %s) scored %d rows", revision[:8], len(pending))
    return len(pending)
