"""NASDAQ symbol directory: download, filter to common stocks, PIT membership diff.

Source: NASDAQ Trader symbol directory (nasdaqlisted.txt), pipe-delimited with a
`File Creation Time:` footer line. Each refresh snapshots the parsed directory to
the raw layer (audit trail), upserts a dated `listing_directory` curated table,
and diffs the filtered common-stock set against the open NASDAQ membership in
`universe_membership` — new symbols open an interval at `as_of`, disappeared
symbols close theirs. Run daily and real point-in-time membership accrues.
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
from qtdata.providers.base import retry_transient
from qtdata.storage import parquet_store

logger = logging.getLogger(__name__)

COMMON_STOCK_NAME_EXCLUDE = re.compile(
    r"warrant|right|unit|preferred|notes|debenture|bond", re.IGNORECASE
)
SYMBOL_RE = re.compile(r"^[A-Z]{1,5}$")
GOOD_FINANCIAL_STATUS = frozenset({"N", "D"})

INITIAL_SNAPSHOT_NOTE = (
    "INITIAL SNAPSHOT: forward-PIT from this date; pre-history unknown "
    "(use Norgate/Sharadar for historical membership)."
)

_COLUMNS = {
    "Symbol": "ticker",
    "Security Name": "security_name",
    "Market Category": "market_category",
    "Test Issue": "test_issue",
    "Financial Status": "financial_status",
    "Round Lot Size": "round_lot_size",
    "ETF": "etf",
    "NextShares": "nextshares",
}


@dataclass
class UniverseRefreshSummary:
    as_of: date
    run_id: str
    directory_rows: int = 0
    common_stocks: int = 0
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: int = 0


@retry_transient
def download_directory(settings: Settings) -> str:
    resp = requests.get(settings.nasdaq_listed_url, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_nasdaq_listed(text: str) -> pd.DataFrame:
    """Parse the pipe-delimited directory, stripping the File Creation Time footer."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if lines and lines[-1].startswith("File Creation Time"):
        lines = lines[:-1]
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|", dtype=str)
    missing = set(_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"nasdaqlisted.txt schema changed; missing columns: {missing}")
    df = df.rename(columns=_COLUMNS)[list(_COLUMNS.values())]
    df["ticker"] = df["ticker"].astype(str).str.strip()
    for col in ("market_category", "test_issue", "financial_status", "etf", "nextshares"):
        df[col] = df[col].astype(str).str.strip()
    df["round_lot_size"] = pd.to_numeric(df["round_lot_size"], errors="coerce")
    return df.reset_index(drop=True)


def filter_common_stocks(directory: pd.DataFrame) -> pd.DataFrame:
    """Common stocks only: no test issues, ETFs, derivatives-like names, bad status."""
    d = directory
    mask = (
        (d["test_issue"] == "N")
        & (d["etf"] == "N")
        & d["financial_status"].isin(GOOD_FINANCIAL_STATUS)
        & ~d["security_name"].fillna("").str.contains(COMMON_STOCK_NAME_EXCLUDE)
        & d["ticker"].str.fullmatch(SYMBOL_RE)
    )
    return d[mask].reset_index(drop=True)


def refresh_nasdaq(
    settings: Settings,
    as_of: date | None = None,
    raw_text: str | None = None,
    index_name: str = "NASDAQ",
) -> UniverseRefreshSummary:
    as_of = as_of or date.today()
    run_id = uuid4().hex[:12]
    text = raw_text if raw_text is not None else download_directory(settings)

    directory = parse_nasdaq_listed(text)
    common = filter_common_stocks(directory)
    summary = UniverseRefreshSummary(
        as_of=as_of, run_id=run_id,
        directory_rows=len(directory), common_stocks=len(common),
    )

    # 1. immutable raw snapshot (audit) — date-partitioned raw layout
    snapshot = directory.assign(
        as_of=pd.Timestamp(as_of),
        is_common_stock=directory["ticker"].isin(set(common["ticker"])),
        source="nasdaq_trader",
        run_id=run_id,
        ingested_at=pd.Timestamp.now(tz="UTC"),
    )
    raw_path = (
        settings.raw_dir / "provider=nasdaq_trader" / "dataset=listing_directory"
        / f"as_of={as_of}" / f"{run_id}.parquet"
    )
    parquet_store.write_raw(snapshot, raw_path)

    # 2. curated dated directory (PIT history of the listing file)
    curated = snapshot.copy()
    curated["year"] = as_of.year
    parquet_store.upsert(
        curated, settings.curated_dir / "listing_directory", LISTING_KEY, partition_col="year"
    )

    # 3. diff filtered set vs open membership
    members = parquet_store.read(settings.curated_dir / "universe_membership")
    if members.empty:
        open_rows = pd.DataFrame(
            columns=["index_name", "ticker", "effective_from", "effective_to", "source", "note"]
        )
    else:
        open_rows = members[
            (members["index_name"] == index_name) & members["effective_to"].isna()
        ]

    current = set(open_rows["ticker"]) if not open_rows.empty else set()
    target = set(common["ticker"])
    added = sorted(target - current)
    removed = sorted(current - target)
    summary.added = added
    summary.removed = removed
    summary.unchanged = len(target & current)

    table_dir = settings.curated_dir / "universe_membership"
    if added:
        note = INITIAL_SNAPSHOT_NOTE if not current else ""
        new_rows = pd.DataFrame(
            {
                "index_name": index_name,
                "ticker": added,
                "effective_from": pd.Timestamp(as_of),
                "effective_to": pd.NaT,
                "source": "nasdaq_trader",
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
            index_name, as_of, len(added), len(removed), summary.unchanged,
        )
    return summary
