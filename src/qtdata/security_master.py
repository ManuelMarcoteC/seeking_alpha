"""Security master: ticker -> exchange, GICS 4-level classification, round lot.

B0 of the seeking-alpha plan. GICS comes from EODHD fundamentals (verified live
2026-07-05: ``General.GicSector/GicGroup/GicIndustry/GicSubIndustry``), cached
to a curated parquet table. Classification changes are VERSIONED (interval rows,
flag-never-mutate): a re-fetch that finds a different classification closes the
old row at `as_of` and opens a new one — history is never overwritten.

Sector hierarchy use (A6'/C13): neutralization at Sector/Group level, rotation
signal at Industry/SubIndustry level. Rows missing GICS are kept with a
``gics_missing`` note (acceptance gate in B0: >95% coverage).
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import requests

from qtdata.config import Settings
from qtdata.models import ProviderNotConfiguredError
from qtdata.providers.base import RateLimiter, retry_transient
from qtdata.storage import parquet_store

logger = logging.getLogger(__name__)

_FUND_BASE = "https://eodhd.com/api/fundamentals"
_TABLE = "security_master"

SECURITY_MASTER_KEY = ["ticker", "effective_from"]

COLUMNS = [
    "ticker", "exchange", "gic_sector", "gic_group", "gic_industry",
    "gic_sub_industry", "round_lot_size", "effective_from", "effective_to",
    "source", "note",
]


class GICSFetcher:
    """Minimal EODHD fundamentals client (General section only)."""

    def __init__(self, settings: Settings):
        key = getattr(settings, "eodhd_api_key", None)
        self._token = key.get_secret_value() if key else None
        self._limiter = RateLimiter(settings.eodhd_rate_limit_per_min)

    def _require_token(self) -> str:
        if not self._token:
            raise ProviderNotConfiguredError(
                "GICS fetch requires QT_EODHD_API_KEY (fundamentals endpoint)."
            )
        return self._token

    @retry_transient
    def fetch_general(self, ticker: str, exchange_suffix: str = "US") -> dict:
        token = self._require_token()
        self._limiter.acquire()
        resp = requests.get(
            f"{_FUND_BASE}/{ticker.upper()}.{exchange_suffix}",
            params={"api_token": token, "fmt": "json", "filter": "General"},
            timeout=60,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else {}


def row_from_general(
    ticker: str, general: dict, as_of: date, exchange: str = "",
    round_lot_size: float | None = None,
) -> dict:
    """Map the EODHD General section to one security_master interval row."""
    gics = {
        "gic_sector": general.get("GicSector"),
        "gic_group": general.get("GicGroup"),
        "gic_industry": general.get("GicIndustry"),
        "gic_sub_industry": general.get("GicSubIndustry"),
    }
    missing = not any(v for v in gics.values())
    return {
        "ticker": ticker.upper(),
        "exchange": exchange or (general.get("Exchange") or ""),
        **{k: (v or "") for k, v in gics.items()},
        "round_lot_size": round_lot_size,
        "effective_from": pd.Timestamp(as_of),
        "effective_to": pd.NaT,
        "source": "eodhd_fundamentals",
        "note": "gics_missing" if missing else "",
    }


def upsert_security_master(settings: Settings, rows: list[dict], as_of: date) -> int:
    """Versioned upsert: close changed rows at as_of, open the new state.

    Never mutates history — a classification change produces a new interval.
    """
    if not rows:
        return 0
    incoming = pd.DataFrame(rows)[COLUMNS]
    table_dir = settings.curated_dir / _TABLE
    existing = parquet_store.read(table_dir)

    to_write = [incoming]
    if not existing.empty:
        open_rows = existing[existing["effective_to"].isna()]
        merged = open_rows.merge(incoming, on="ticker", suffixes=("_old", ""))
        if not merged.empty:
            compare_cols = ["gic_sector", "gic_group", "gic_industry", "gic_sub_industry", "exchange"]
            changed_mask = pd.Series(False, index=merged.index)
            for col in compare_cols:
                changed_mask |= merged[f"{col}_old"].astype(str) != merged[col].astype(str)
            changed_tickers = set(merged.loc[changed_mask, "ticker"])
            unchanged_tickers = set(merged.loc[~changed_mask, "ticker"])
            # unchanged tickers: keep the existing open row, drop the incoming duplicate
            to_write[0] = incoming[~incoming["ticker"].isin(sorted(unchanged_tickers))]
            if changed_tickers:
                closing = open_rows[open_rows["ticker"].isin(sorted(changed_tickers))].copy()
                closing["effective_to"] = pd.Timestamp(as_of)
                to_write.append(closing[COLUMNS])

    out = pd.concat(to_write, ignore_index=True)
    if out.empty:
        return 0
    res = parquet_store.upsert(out, table_dir, SECURITY_MASTER_KEY, partition_col=None)
    return res.rows_written


def refresh_security_master(
    settings: Settings,
    tickers: list[str],
    as_of: date | None = None,
    exchange_suffix: str = "US",
) -> dict[str, int]:
    """Fetch GICS for `tickers` and version-upsert. Returns coverage counts."""
    as_of = as_of or date.today()
    fetcher = GICSFetcher(settings)
    rows: list[dict] = []
    errors = 0
    for t in tickers:
        try:
            general = fetcher.fetch_general(t, exchange_suffix)
            rows.append(row_from_general(t, general, as_of))
        except requests.HTTPError as exc:  # 404 = vendor has no fundamentals
            logger.warning("fundamentals %s: %s", t, exc)
            rows.append(row_from_general(t, {}, as_of))
            errors += 1
    written = upsert_security_master(settings, rows, as_of)
    with_gics = sum(1 for r in rows if r["note"] != "gics_missing")
    coverage = {
        "requested": len(tickers), "rows_written": written,
        "with_gics": with_gics, "gics_missing": len(rows) - with_gics,
        "http_errors": errors,
    }
    logger.info("security_master refresh: %s", coverage)
    return coverage
