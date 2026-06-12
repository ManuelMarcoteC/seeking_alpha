"""Orchestrator: sentiment_daily + ohlcv_daily_adj -> IC / decay / event study report."""

from __future__ import annotations

import logging
from datetime import date
from uuid import uuid4

import pandas as pd

from qtdata.config import Settings
from qtdata.research.event_study import find_events, run_event_study
from qtdata.research.ic import decay_profile
from qtdata.research.report import (
    SentimentValidationReport,
    persist_research_report,
)
from qtdata.research.returns import forward_returns, load_adjusted_closes
from qtdata.storage.catalog import Catalog

logger = logging.getLogger(__name__)


def run_sentiment_validation(
    settings: Settings,
    catalog: Catalog,
    *,
    horizons: tuple[int, ...] = (1, 5, 20),
    score_col: str = "sent_finbert",
    min_breadth: int = 10,
    event_threshold: float = 0.5,
    min_articles: int = 3,
    start: date | None = None,
    end: date | None = None,
) -> SentimentValidationReport:
    views = {
        r[0]
        for r in catalog.conn.execute(
            "SELECT table_name FROM information_schema.tables"
        ).fetchall()
    }
    if "sentiment_daily" not in views:
        raise ValueError("sentiment_daily no existe — ejecuta `qt news update` primero")
    if "ohlcv_daily_adj" not in views:
        raise ValueError("ohlcv_daily_adj no existe — ejecuta `qt ingest` + `qt curate` primero")

    factor = catalog.query(
        "SELECT ticker, date, sent_av, sent_finbert, n_articles FROM sentiment_daily"
    )
    if factor.empty:
        raise ValueError("sentiment_daily está vacío — ejecuta `qt news update` primero")
    factor["date"] = pd.to_datetime(factor["date"])
    if start is not None:
        factor = factor[factor["date"] >= pd.Timestamp(start)]
    if end is not None:
        factor = factor[factor["date"] <= pd.Timestamp(end)]

    caveats: list[str] = []
    if score_col == "sent_finbert" and factor["sent_finbert"].notna().sum() == 0:
        score_col = "sent_av"
        caveats.append(
            "sent_finbert sin puntuar — análisis degradado a sent_av (score del "
            "vendor, NO point-in-time del lado del scorer)."
        )

    # closes load from `start` but ignore `end`: forward returns need future sessions
    closes = load_adjusted_closes(catalog, start=start)
    if closes.empty:
        raise ValueError("ohlcv_daily_adj está vacío para ese rango")
    fwd = forward_returns(closes, horizons)

    ic_summaries = decay_profile(
        factor, fwd, score_col=score_col, horizons=horizons, min_breadth=min_breadth
    )
    events_df = find_events(
        factor, score_col=score_col, threshold=event_threshold, min_articles=min_articles
    )
    events = run_event_study(closes, events_df) if not events_df.empty else None

    report = SentimentValidationReport(
        run_id=uuid4().hex[:12],
        params={
            "score_col": score_col,
            "horizons": list(horizons),
            "min_breadth": min_breadth,
            "event_threshold": event_threshold,
            "min_articles": min_articles,
            "start": str(start) if start else "-",
            "end": str(end) if end else "-",
        },
        ic=ic_summaries,
        events=events,
        n_factor_days=int(factor["date"].nunique()),
        n_tickers=int(factor["ticker"].nunique()),
        caveats=caveats,
    )
    persist_research_report(report, settings)
    logger.info("Sentiment validation %s written to %s", report.run_id, report.path)
    return report
