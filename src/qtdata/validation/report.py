"""Validation report assembly and persistence.

Flags are upserted into curated/validation_flags (queryable next to prices);
a human-readable markdown summary lands in data/reports/.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from qtdata.config import Settings
from qtdata.models import FLAGS_KEY
from qtdata.storage import parquet_store


@dataclass
class ValidationReport:
    run_id: str
    flags: pd.DataFrame
    quarantined: pd.DataFrame = field(default_factory=pd.DataFrame)


def persist_report(report: ValidationReport, settings: Settings) -> None:
    if not report.flags.empty:
        flags = report.flags.copy()
        flags["run_id"] = report.run_id
        flags["flagged_at"] = pd.Timestamp.now(tz="UTC")
        flags["year"] = pd.to_datetime(flags["date"]).dt.year
        parquet_store.upsert(
            flags, settings.curated_dir / "validation_flags", FLAGS_KEY, partition_col="year"
        )

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# Validation report — run `{report.run_id}`", ""]
    if report.flags.empty:
        lines.append("No anomalies flagged.")
    else:
        lines.append(f"Total flags: **{len(report.flags)}** (data left untouched — flags only)\n")
        by_type = (
            report.flags.groupby(["flag_type", "severity"]).size().rename("count").reset_index()
        )
        lines.append("| flag_type | severity | count |")
        lines.append("|---|---|---|")
        for _, r in by_type.iterrows():
            lines.append(f"| {r['flag_type']} | {r['severity']} | {r['count']} |")
        top = report.flags["ticker"].value_counts().head(10)
        lines.append("\nMost-flagged tickers: " + ", ".join(f"{t} ({n})" for t, n in top.items()))
    if not report.quarantined.empty:
        lines.append(
            f"\n**Quarantined rows (schema violations): {len(report.quarantined)}** — "
            f"see quarantine parquet for this run."
        )
    out = settings.reports_dir / f"validation_{report.run_id}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def persist_quarantine(failure_cases: pd.DataFrame, run_id: str, settings: Settings) -> None:
    if failure_cases.empty:
        return
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    cases = failure_cases.copy()
    for col in cases.columns:
        if cases[col].dtype == object:
            cases[col] = cases[col].astype(str)
    cases.to_parquet(settings.reports_dir / f"quarantine_{run_id}.parquet", index=False)
