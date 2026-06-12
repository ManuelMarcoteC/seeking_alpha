"""Cross-source reconciliation with tolerance bands.

Compares the raw layers of two providers on (ticker, date): per-field relative
differences are classified as match / within_tolerance / discrepancy, and rows
present in only one source are reported as missing. Output feeds a discrepancy
report — it never modifies data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qtdata.config import Settings
from qtdata.models import Dataset

PRICE_FIELDS = ["open", "high", "low", "close"]
_EPS = 1e-12


@dataclass
class ReconciliationResult:
    summary: pd.DataFrame        # per ticker: counts by classification
    discrepancies: pd.DataFrame  # row level: ticker, date, field, value_a, value_b, rel_diff


def load_raw_frame(
    settings: Settings,
    provider: str,
    dataset: Dataset = Dataset.OHLCV_DAILY,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Latest raw payload rows per (ticker, date) for one provider."""
    base = settings.raw_dir / f"provider={provider}" / f"dataset={dataset}"
    files = sorted(base.glob("ticker=*/*.parquet"))
    if tickers is not None:
        wanted = {f"ticker={t}" for t in tickers}
        files = [f for f in files if f.parent.name in wanted]
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return (
        df.sort_values("ingested_at")
        .drop_duplicates(subset=["ticker", "date"], keep="last")
        .reset_index(drop=True)
    )


def reconcile(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str,
    label_b: str,
    price_rel_tol: float = 0.005,
    volume_rel_tol: float = 0.10,
) -> ReconciliationResult:
    cols = ["ticker", "date", *PRICE_FIELDS, "volume"]
    a = df_a[cols].copy()
    b = df_b[cols].copy()
    merged = a.merge(b, on=["ticker", "date"], how="outer", suffixes=("_a", "_b"), indicator=True)

    records: list[dict] = []
    both = merged[merged["_merge"] == "both"]
    for f in [*PRICE_FIELDS, "volume"]:
        va = both[f"{f}_a"].to_numpy(dtype=float)
        vb = both[f"{f}_b"].to_numpy(dtype=float)
        denom = np.maximum(np.maximum(np.abs(va), np.abs(vb)), _EPS)
        rel = np.abs(va - vb) / denom
        tol = volume_rel_tol if f == "volume" else price_rel_tol
        classification = np.where(
            rel == 0, "match", np.where(rel <= tol, "within_tolerance", "discrepancy")
        )
        for i in np.flatnonzero(classification != "match"):
            row = both.iloc[i]
            records.append(
                {
                    "ticker": row["ticker"],
                    "date": row["date"],
                    "field": f,
                    f"value_{label_a}": va[i],
                    f"value_{label_b}": vb[i],
                    "rel_diff": rel[i],
                    "classification": classification[i],
                }
            )

    for side, label_missing in (("left_only", label_b), ("right_only", label_a)):
        for _, row in merged[merged["_merge"] == side].iterrows():
            records.append(
                {
                    "ticker": row["ticker"],
                    "date": row["date"],
                    "field": None,
                    f"value_{label_a}": np.nan,
                    f"value_{label_b}": np.nan,
                    "rel_diff": np.nan,
                    "classification": f"missing_in_{label_missing}",
                }
            )

    discrepancies = pd.DataFrame(records)
    if discrepancies.empty:
        summary = pd.DataFrame(columns=["ticker", "classification", "count"])
    else:
        summary = (
            discrepancies.groupby(["ticker", "classification"]).size().rename("count").reset_index()
        )
    return ReconciliationResult(summary=summary, discrepancies=discrepancies)


def persist_reconciliation(
    result: ReconciliationResult, label_a: str, label_b: str, run_id: str, settings: Settings
) -> None:
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    if not result.discrepancies.empty:
        result.discrepancies.to_parquet(
            settings.reports_dir / f"reconcile_{run_id}.parquet", index=False
        )
    lines = [f"# Reconciliation — {label_a} vs {label_b} (run `{run_id}`)", ""]
    if result.discrepancies.empty:
        lines.append("All compared rows match exactly.")
    else:
        counts = result.discrepancies["classification"].value_counts()
        lines.append("| classification | count |")
        lines.append("|---|---|")
        for cls, n in counts.items():
            lines.append(f"| {cls} | {n} |")
    (settings.reports_dir / f"reconcile_{run_id}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
