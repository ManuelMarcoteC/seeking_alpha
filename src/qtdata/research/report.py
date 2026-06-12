"""Markdown persistence for the sentiment-factor validation (house report style)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from qtdata.config import Settings
from qtdata.research.event_study import EventStudyResult
from qtdata.research.ic import ICSummary

# always printed — a validation number without its caveats is a trap
STANDING_CAVEATS = [
    "El t-stat asume ICs diarios i.i.d.; con horizontes solapados (h>1) está "
    "sobreestimado (sin corrección Newey-West en v1). Con historia corta, "
    "tratar |IC medio| < ~0.02 o t < 2 como ruido.",
    "Cobertura de noticias sesgada a lo reciente: yfinance solo expone el stream "
    "actual por ticker; la muestra de IC empieza en la fecha de inicio del harvest.",
    "Retornos forward close(D)->close(D+h): asume fill a las 16:00 de una señal "
    "completa a las 15:30 (colchón de 30 min); variante open(D+1) pendiente como "
    "robustness check.",
    "La membresía del universo es point-in-time solo hacia delante desde el primer "
    "`qt universe refresh`; nada de lo anterior es backtesteable sin sesgo.",
    "El offset 0 del event study incluye la propia sesión del evento; la lectura "
    "limpia son los offsets >= +1.",
]


@dataclass
class SentimentValidationReport:
    run_id: str
    params: dict
    ic: list[ICSummary]  # one per horizon — doubles as the decay curve
    events: EventStudyResult | None
    n_factor_days: int
    n_tickers: int
    caveats: list[str] = field(default_factory=list)
    path: Path | None = None


def _f(value: float, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/d"
    return f"{value:.{digits}f}"


def persist_research_report(report: SentimentValidationReport, settings: Settings) -> Path:
    lines: list[str] = [
        f"# Validación del factor de sentimiento — run `{report.run_id}`",
        "",
        "**Parámetros:** " + ", ".join(f"{k}={v}" for k, v in report.params.items()),
        f"**Muestra:** {report.n_factor_days} días de factor, "
        f"{report.n_tickers} tickers",
        "",
        "## IC de Spearman por horizonte (curva de decaimiento)",
        "",
        "| horizonte (sesiones) | n días | IC medio | std | t-stat | ICIR | hit rate |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in report.ic:
        lines.append(
            f"| {s.horizon} | {s.n_days} | {_f(s.mean_ic)} | {_f(s.std_ic)} | "
            f"{_f(s.t_stat, 2)} | {_f(s.icir, 2)} | {_f(s.hit_rate, 2)} |"
        )
    lines.append("")

    lines.append("## Event study (CAR medio, ajustado por mercado equiponderado)")
    lines.append("")
    if report.events is None:
        lines.append("Sin eventos que superen el umbral — sección omitida.")
    else:
        ev = report.events
        lines.append(
            f"Ventana {ev.window} sesiones; eventos positivos: {ev.n_pos}, "
            f"negativos: {ev.n_neg}."
        )
        lines.append("")
        lines.append("| offset | CAR eventos + | CAR eventos − |")
        lines.append("|---|---|---|")
        offsets = sorted(set(ev.car_pos.index) | set(ev.car_neg.index))
        for off in offsets:
            pos = _f(float(ev.car_pos.get(off, float("nan"))))
            neg = _f(float(ev.car_neg.get(off, float("nan"))))
            lines.append(f"| {off:+d} | {pos} | {neg} |")
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    for caveat in list(report.caveats) + STANDING_CAVEATS:
        lines.append(f"- {caveat}")
    lines.append("")

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    out = settings.reports_dir / f"sentiment_ic_{report.run_id}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    report.path = out
    return out
