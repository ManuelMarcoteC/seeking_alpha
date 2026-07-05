"""Other-listed symbol directory (NYSE/ARCA/AMEX/CBOE): download, filter, PIT membership diff.

B0 of the seeking-alpha plan (universe widened to the full US listed market, C13).
Source: NASDAQ Trader ``otherlisted.txt`` — the sibling of ``nasdaqlisted.txt``
covering non-NASDAQ listings, pipe-delimited with the same ``File Creation Time:``
footer. Verified live 2026-07-05: 7,420 rows; Exchange codes N=2,934 (NYSE),
P=2,674 (Arca), Z=1,492 (Cboe), A=316 (Amex); 3,204 non-ETF.

Each exchange keeps ITS OWN ``index_name`` roster (NYSE/ARCA/AMEX/CBOE) — rosters
are never mixed (Tier B pattern), so a backtest can ask for exactly the universe
it means. Forward-PIT from the seed date, identical to the NASDAQ roster:
pre-history unknown, survivorship boundary documented in the row note.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from datetime import date
from uuid import uuid4

import pandas as pd
import requests

from qtdata.config import Settings
from qtdata.models import LISTING_KEY, UNIVERSE_KEY
from qtdata.nasdaq_directory import (
    COMMON_STOCK_NAME_EXCLUDE,
    INITIAL_SNAPSHOT_NOTE,
    SYMBOL_RE,
)
from qtdata.providers.base import retry_transient
from qtdata.storage import parquet_store

logger = logging.getLogger(__name__)

# Exchange code -> roster (index_name). M (Chicago) and V (IEX) are ignored:
# a handful of listings, no analytical value for the discovery ring.
EXCHANGE_ROSTERS = {"N": "NYSE", "P": "ARCA", "A": "AMEX", "Z": "CBOE"}

_COLUMNS = {
    "ACT Symbol": "ticker",
    "Security Name": "security_name",
    "Exchange": "exchange",
    "CQS Symbol": "cqs_symbol",
    "ETF": "etf",
    "Round Lot Size": "round_lot_size",
    "Test Issue": "test_issue",
    "NASDAQ Symbol": "nasdaq_symbol",
}


@dataclass
class OtherRefreshSummary:
    as_of: date
    run_id: str
    directory_rows: int = 0
    per_roster: dict[str, dict[str, int]] = field(default_factory=dict)


@retry_transient
def download_directory(settings: Settings) -> str:
    resp = requests.get(settings.other_listed_url, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_other_listed(text: str) -> pd.DataFrame:
    """Parse the pipe-delimited directory, stripping the File Creation Time footer."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if lines and lines[-1].startswith("File Creation Time"):
        lines = lines[:-1]
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|", dtype=str)
    missing = set(_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"otherlisted.txt schema changed; missing columns: {missing}")
    df = df.rename(columns=_COLUMNS)[list(_COLUMNS.values())]
    for col in ("ticker", "exchange", "cqs_symbol", "etf", "test_issue", "nasdaq_symbol"):
        df[col] = df[col].astype(str).str.strip()
    df["round_lot_size"] = pd.to_numeric(df["round_lot_size"], errors="coerce")
    return df.reset_index(drop=True)


def filter_common_stocks(directory: pd.DataFrame, exchange_code: str) -> pd.DataFrame:
    """Common stocks of ONE exchange: no test issues, ETFs, derivative-like names.

    Unlike nasdaqlisted there is no Financial Status column here; the name-based
    exclusion regex plus the plain-symbol regex carry the filtering weight.
    """
    d = directory
    mask = (
        (d["exchange"] == exchange_code)
        & (d["test_issue"] == "N")
        & (d["etf"] == "N")
        & ~d["security_name"].fillna("").str.contains(COMMON_STOCK_NAME_EXCLUDE)
        & d["ticker"].str.fullmatch(SYMBOL_RE)
    )
    return d[mask].reset_index(drop=True)


def refresh_other(
    settings: Settings,
    as_of: date | None = None,
    raw_text: str | None = None,
    exchanges: tuple[str, ...] = tuple(EXCHANGE_ROSTERS),
) -> OtherRefreshSummary:
    """Snapshot the directory and diff each exchange roster (forward-PIT accrual)."""
    as_of = as_of or date.today()
    run_id = uuid4().hex[:12]
    text = raw_text if raw_text is not None else download_directory(settings)

    directory = parse_other_listed(text)
    summary = OtherRefreshSummary(as_of=as_of, run_id=run_id, directory_rows=len(directory))

    # 1. immutable raw snapshot (audit) — one snapshot for the whole file
    common_by_code = {
        code: filter_common_stocks(directory, code) for code in exchanges
    }
    all_common = (
        set().union(*(set(c["ticker"]) for c in common_by_code.values()))
        if common_by_code else set()
    )
    snapshot = directory.assign(
        as_of=pd.Timestamp(as_of),
        is_common_stock=directory["ticker"].isin(all_common),
        source="nasdaq_trader_other",
        run_id=run_id,
        ingested_at=pd.Timestamp.now(tz="UTC"),
    )
    raw_path = (
        settings.raw_dir / "provider=nasdaq_trader" / "dataset=other_listing_directory"
        / f"as_of={as_of}" / f"{run_id}.parquet"
    )
    parquet_store.write_raw(snapshot, raw_path)

    # 2. curated dated directory (PIT history of the listing file, shared table
    #    with the NASDAQ directory — rows are distinguishable by `source`)
    curated = snapshot.copy()
    curated["year"] = as_of.year
    parquet_store.upsert(
        curated, settings.curated_dir / "other_listing_directory", LISTING_KEY,
        partition_col="year",
    )

    # 3. diff each exchange roster vs its open membership — NEVER mixed
    members = parquet_store.read(settings.curated_dir / "universe_membership")
    table_dir = settings.curated_dir / "universe_membership"

    for code in exchanges:
        roster = EXCHANGE_ROSTERS[code]
        common = common_by_code[code]
        if members.empty:
            open_rows = pd.DataFrame(
                columns=["index_name", "ticker", "effective_from", "effective_to", "source", "note"]
            )
        else:
            open_rows = members[
                (members["index_name"] == roster) & members["effective_to"].isna()
            ]
        current = set(open_rows["ticker"]) if not open_rows.empty else set()
        target = set(common["ticker"])
        added = sorted(target - current)
        removed = sorted(current - target)
        summary.per_roster[roster] = {
            "common": len(target), "added": len(added), "removed": len(removed),
            "unchanged": len(target & current),
        }

        if added:
            note = INITIAL_SNAPSHOT_NOTE if not current else ""
            new_rows = pd.DataFrame(
                {
                    "index_name": roster,
                    "ticker": added,
                    "effective_from": pd.Timestamp(as_of),
                    "effective_to": pd.NaT,
                    "source": "nasdaq_trader_other",
                    "note": note,
                }
            )
            parquet_store.upsert(new_rows, table_dir, UNIVERSE_KEY, partition_col=None)
        if removed:
            closing = open_rows[open_rows["ticker"].isin(removed)].copy()
            closing["effective_to"] = pd.Timestamp(as_of)
            closing = closing[
                ["index_name", "ticker", "effective_from", "effective_to", "source", "note"]
            ]
            parquet_store.upsert(closing, table_dir, UNIVERSE_KEY, partition_col=None)

        if added or removed:
            logger.info(
                "Universe %s refreshed as of %s: +%d / -%d (unchanged %d)",
                roster, as_of, len(added), len(removed),
                summary.per_roster[roster]["unchanged"],
            )
    return summary
