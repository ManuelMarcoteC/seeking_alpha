"""PIT-safe forward returns from adjusted closes.

Forward return for date D = close(D) -> close(D+h), within each ticker's OWN
observed sessions (a halt or gap never borrows another ticker's price). The
sentiment factor attributes an article to session D only when its effective_ts
is <= 15:30 ET of D (news/aggregate.trading_day_for), so a 16:00 close fill has
a 30-minute information buffer: close-to-close forward returns are
look-ahead-free by construction of the factor's date attribution. The more
conservative open(D+1) entry is a documented robustness check, not v1.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from qtdata.storage.catalog import Catalog


def load_adjusted_closes(
    catalog: Catalog, start: date | None = None, end: date | None = None
) -> pd.DataFrame:
    """[ticker, date, close] from ohlcv_daily_adj, sorted by (ticker, date)."""
    sql = "SELECT ticker, date, close FROM ohlcv_daily_adj"
    clauses = []
    if start is not None:
        clauses.append(f"date >= DATE '{start}'")
    if end is not None:
        clauses.append(f"date <= DATE '{end}'")
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY ticker, date"
    df = catalog.query(sql)
    df["date"] = pd.to_datetime(df["date"])
    return df


def forward_returns(
    closes: pd.DataFrame, horizons: tuple[int, ...] = (1, 5, 20)
) -> pd.DataFrame:
    """Long frame [ticker, date, fwd_{h}d ...]; the tail h sessions are NaN."""
    out = closes.sort_values(["ticker", "date"]).reset_index(drop=True).copy()
    grouped = out.groupby("ticker")["close"]
    for h in horizons:
        out[f"fwd_{h}d"] = grouped.shift(-h) / out["close"] - 1.0
    return out.drop(columns=["close"])
