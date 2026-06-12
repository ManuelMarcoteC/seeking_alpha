"""Deterministic post-checks on agent proposals — code, never an LLM.

"La explicación no es prueba; la verificación es el recibo": every proposal is
re-checked against the actual data — hallucinated tickers, sector concentration,
nonexistent cited columns, and numeric spot-checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from qtdata.storage.catalog import Catalog

if TYPE_CHECKING:  # avoid circular import at runtime
    from qtdata.agents.screener import Proposal


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class VerificationReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def render(self) -> str:
        lines = ["Verificación determinista:"]
        for c in self.checks:
            mark = "✓" if c.passed else "✗"
            lines.append(f"  {mark} {c.name}: {c.detail}")
        return "\n".join(lines)


def _existing_views(catalog: Catalog) -> set[str]:
    return {
        r[0]
        for r in catalog.conn.execute(
            "SELECT table_name FROM information_schema.tables"
        ).fetchall()
    }


def verify_proposal(
    catalog: Catalog, proposal: Proposal, max_sector_pct: float = 0.40
) -> VerificationReport:
    report = VerificationReport()
    tickers = [c.ticker.upper() for c in proposal.candidates]
    views = _existing_views(catalog)

    # 1. hallucinated tickers: every candidate must exist somewhere we know
    known: set[str] = set()
    if "fundamentals_snapshot" in views:
        rows = catalog.conn.execute(
            "SELECT DISTINCT ticker FROM fundamentals_snapshot"
        ).fetchall()
        known |= {r[0] for r in rows}
    if "universe_membership" in views:
        rows = catalog.conn.execute(
            "SELECT DISTINCT ticker FROM universe_membership"
        ).fetchall()
        known |= {r[0] for r in rows}
    if known:
        ghosts = sorted(set(tickers) - known)
        report.checks.append(
            Check(
                "tickers existen",
                not ghosts,
                "todos en el universo/fundamentals" if not ghosts
                else f"NO están en los datos (citados sin verlos): {ghosts}",
            )
        )

    # 2. sector concentration — only meaningful for actual books (a 40% cap is
    # mathematically unsatisfiable below 1/0.40 ≈ 3 names; webinar books are 12-15)
    if "fundamentals_snapshot" in views and len(tickers) >= 5:
        placeholders = ", ".join("?" for _ in tickers)
        rows = catalog.conn.execute(
            f"SELECT ticker, sector FROM fundamentals_snapshot "
            f"WHERE ticker IN ({placeholders})",
            tickers,
        ).fetchall()
        sectors: dict[str, int] = {}
        for _, sector in rows:
            if sector:
                sectors[sector] = sectors.get(sector, 0) + 1
        if sectors:
            top_sector, top_n = max(sectors.items(), key=lambda kv: kv[1])
            share = top_n / len(tickers)
            report.checks.append(
                Check(
                    f"concentración sectorial ≤ {max_sector_pct:.0%}",
                    share <= max_sector_pct,
                    f"{top_sector}: {share:.0%} ({top_n}/{len(tickers)})",
                )
            )

    # 3. cited columns exist in their cited views
    bad_citations: list[str] = []
    described: dict[str, set[str]] = {}
    for cand in proposal.candidates:
        for m in cand.metrics:
            view = m.source_view
            if view not in views:
                bad_citations.append(f"{cand.ticker}: vista inexistente {view}")
                continue
            if view not in described:
                cols = catalog.conn.execute(f"DESCRIBE {view}").fetchall()
                described[view] = {c[0] for c in cols}
            if m.column not in described[view]:
                bad_citations.append(f"{cand.ticker}: {view}.{m.column} no existe")
    report.checks.append(
        Check(
            "columnas citadas existen",
            not bad_citations,
            "todas verificadas" if not bad_citations else "; ".join(bad_citations[:5]),
        )
    )

    # 4. numeric spot-check against single-row-per-ticker views
    mismatches: list[str] = []
    checked = 0
    for cand in proposal.candidates:
        for m in cand.metrics:
            if not isinstance(m.value, (int, float)):
                continue
            if m.source_view != "fundamentals_snapshot" or "fundamentals_snapshot" not in views:
                continue
            if m.column not in described.get("fundamentals_snapshot", set()):
                continue
            row = catalog.conn.execute(
                f'SELECT "{m.column}" FROM fundamentals_snapshot WHERE ticker = ?',
                [cand.ticker.upper()],
            ).fetchone()
            if row is None or row[0] is None:
                continue
            try:
                actual = float(row[0])
            except (TypeError, ValueError):
                continue
            checked += 1
            tolerance = max(abs(actual) * 0.01, 1e-9)
            if abs(actual - float(m.value)) > tolerance:
                mismatches.append(
                    f"{cand.ticker}.{m.column}: dijo {m.value}, real {actual:g}"
                )
    if checked:
        report.checks.append(
            Check(
                "spot-check numérico (±1%)",
                not mismatches,
                f"{checked} cifras verificadas" if not mismatches
                else "; ".join(mismatches[:5]),
            )
        )
    return report
