"""Declarative structural invariants for curated tables (pandera).

Schema violations BLOCK promotion to curated (rows are quarantined).
Statistical anomalies (anomalies.py) never block — they only flag.
"""

from __future__ import annotations

import pandas as pd

try:  # pandera >= 0.24 namespaced the pandas API
    import pandera.pandas as pa
except ImportError:  # pragma: no cover
    import pandera as pa
from pandera.errors import SchemaErrors

OHLCV_SCHEMA = pa.DataFrameSchema(
    columns={
        "ticker": pa.Column(str),
        "date": pa.Column("datetime64[ns]"),
        "open": pa.Column(float, pa.Check.gt(0)),
        "high": pa.Column(float, pa.Check.gt(0)),
        "low": pa.Column(float, pa.Check.gt(0)),
        "close": pa.Column(float, pa.Check.gt(0)),
        "volume": pa.Column("Int64", pa.Check.ge(0)),
        "source": pa.Column(str),
        "run_id": pa.Column(str),
        "ingested_at": pa.Column(pd.DatetimeTZDtype(tz="UTC")),
    },
    checks=[
        pa.Check(lambda d: d["high"] >= d["low"], name="high_ge_low"),
        pa.Check(lambda d: d["high"] >= d["open"], name="high_ge_open"),
        pa.Check(lambda d: d["high"] >= d["close"], name="high_ge_close"),
        pa.Check(lambda d: d["low"] <= d["open"], name="low_le_open"),
        pa.Check(lambda d: d["low"] <= d["close"], name="low_le_close"),
    ],
    unique=["ticker", "date"],
    coerce=True,
)

NEWS_ARTICLES_SCHEMA = pa.DataFrameSchema(
    columns={
        "article_id": pa.Column(str, pa.Check.str_matches(r"^[0-9a-f]{64}$")),
        "published_at": pa.Column(pd.DatetimeTZDtype(tz="UTC")),
        "ingested_at": pa.Column(pd.DatetimeTZDtype(tz="UTC")),
        "source": pa.Column(str, nullable=True),
        "provider": pa.Column(str),
        "title": pa.Column(str),
        "summary": pa.Column(str, nullable=True),
        "url": pa.Column(str),
        "overall_sentiment_score": pa.Column(
            float, pa.Check.in_range(-1.0, 1.0), nullable=True
        ),
        "run_id": pa.Column(str),
    },
    unique=["article_id"],
    coerce=True,
)

NEWS_TICKER_SCHEMA = pa.DataFrameSchema(
    columns={
        "article_id": pa.Column(str, pa.Check.str_matches(r"^[0-9a-f]{64}$")),
        # str_length up to 12 to accommodate market suffixes (e.g. "005930.KS",
        # "247540.KQ", "2513.HK"). NASDAQ symbols are <=6; foreign tickers carry a
        # ".XX" / ".XXX" exchange suffix and would otherwise be wrongly quarantined.
        "ticker": pa.Column(str, pa.Check.str_length(1, 12)),
        "published_at": pa.Column(pd.DatetimeTZDtype(tz="UTC")),
        "ingested_at": pa.Column(pd.DatetimeTZDtype(tz="UTC")),
        "relevance": pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=True),
        "score_av": pa.Column(float, pa.Check.in_range(-1.0, 1.0), nullable=True),
        "score_finbert": pa.Column(float, pa.Check.in_range(-1.0, 1.0), nullable=True),
        "finbert_revision": pa.Column(str, nullable=True),
        "scored_at": pa.Column(pd.DatetimeTZDtype(tz="UTC"), nullable=True),
        "run_id": pa.Column(str),
    },
    unique=["article_id", "ticker"],
    coerce=True,
)

ACTIONS_SCHEMA = pa.DataFrameSchema(
    columns={
        "ticker": pa.Column(str),
        "ex_date": pa.Column("datetime64[ns]"),
        "action_type": pa.Column(str, pa.Check.isin(["split", "dividend"])),
        "value": pa.Column(float, pa.Check.gt(0)),
        "source": pa.Column(str),
        "run_id": pa.Column(str),
        "ingested_at": pa.Column(pd.DatetimeTZDtype(tz="UTC")),
    },
    unique=["ticker", "ex_date", "action_type"],
    coerce=True,
)


def validate_frame(
    df: pd.DataFrame, schema: pa.DataFrameSchema
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate; return (valid_rows, failure_cases).

    Row-level failures are dropped into the failure-case frame (quarantine).
    Schema-level failures that cannot be attributed to rows (e.g. a missing
    column) raise — that's a pipeline bug, not bad data.
    """
    try:
        return schema.validate(df, lazy=True), pd.DataFrame()
    except SchemaErrors as exc:
        cases = exc.failure_cases
        if "index" not in cases.columns or cases["index"].isna().all():
            raise ValueError(f"Structural schema failure (not row-level): {cases}") from exc
        bad_idx = pd.Index(cases["index"].dropna().unique())
        valid = schema.validate(df.drop(index=bad_idx), lazy=False)
        return valid, cases
