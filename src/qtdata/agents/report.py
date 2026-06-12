"""One-pager Markdown report per ticker, in the webinar's bquantfunds style.

Deterministic sections pull from fundamentals_snapshot, OUR adjusted prices
(real momentum from ohlcv_daily_adj) and sentiment_daily when present — every
figure cites its source as `view.column`, missing data prints "n/d". The LLM
writes ONLY the judgment section ("Lo que el dato dice"); without an API key
the section is cleanly omitted. Never a price target, never a buy/sell call.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import pandas as pd

from qtdata.agents.llm import LLMClient
from qtdata.config import Settings
from qtdata.storage.catalog import Catalog

if TYPE_CHECKING:  # avoid circular import at runtime
    from qtdata.agents.screener import ScreenerResult

logger = logging.getLogger(__name__)

CLOSING_LINE = (
    "Esto es una sola fuente; un análisis serio cruza varias "
    "(filings, smart money, macro point-in-time)."
)

# section -> [(label, fundamentals_snapshot column)]
SECTIONS: dict[str, list[tuple[str, str]]] = {
    "Identidad": [
        ("Nombre", "name"), ("Sector", "sector"), ("Industria", "industry"),
        ("País", "country"), ("Capitalización", "marketCap"),
        ("Categoría", "marketCapCategory"), ("Bolsa", "exchange"),
    ],
    "Valoración": [
        ("PER", "peRatio"), ("PER forward", "peForward"), ("P/B", "pbRatio"),
        ("P/S", "psRatio"), ("PEG", "pegRatio"), ("EV/EBITDA", "evEbitda"),
        ("FCF yield", "fcfYield"), ("Earnings yield", "earningsYield"),
    ],
    "Calidad / Rentabilidad": [
        ("ROE", "roe"), ("ROA", "roa"), ("ROIC", "roic"),
        ("Margen bruto", "grossMargin"), ("Margen operativo", "operatingMargin"),
        ("Margen neto", "profitMargin"), ("Margen FCF", "fcfMargin"),
    ],
    "Crecimiento": [
        ("Ingresos (últ. trimestre)", "revenueGrowthQ"),
        ("Ingresos 3A", "revenueGrowth3Y"), ("Ingresos 5A", "revenueGrowth5Y"),
        ("BPA (últ. trimestre)", "epsGrowthQ"), ("BPA 3A", "epsGrowth3Y"),
        ("BPA 5A", "epsGrowth5Y"),
    ],
    "Balance / Solvencia": [
        ("Deuda/Equity", "debtEquity"), ("Deuda/EBITDA", "debtEbitda"),
        ("Current ratio", "currentRatio"), ("Cobertura intereses", "interestCoverage"),
        ("Altman Z", "zScore"), ("Piotroski F", "fScore"),
    ],
    "Dividendo / Capital": [
        ("Yield", "dividendYield"), ("Payout", "payoutRatio"),
        ("Años de crecimiento", "dividendGrowthYears"),
        ("Buyback yield", "buybackYield"), ("Dilución (acciones YoY)", "sharesYoY"),
    ],
}

ANALYST_SECTION = [
    ("Rating consenso", "analystRatings"),
    ("Precio objetivo (dato de TERCEROS, no nuestro)", "priceTarget"),
    ("Nº analistas", "analystCount"),
]

LLM_SYSTEM = (
    "Eres un analista sobrio. Te paso los datos verificados de una acción "
    "(cada cifra cita su columna). Escribe SOLO la sección 'Lo que el dato dice': "
    "3 a 5 frases sobrias sobre qué destaca a favor y en contra, señalando "
    "tensiones (p.ej. crecimiento alto con balance apretado). "
    "PROHIBIDO: precio objetivo propio, recomendación de compra/venta/mantener, "
    "predicción de precios, hype. No inventes cifras: usa solo las del bloque."
)


def _fmt(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/d"
    if isinstance(value, float):
        if abs(value) >= 1_000_000:
            return f"{value:,.0f}"
        return f"{value:,.4g}"
    return str(value)


def _momentum_lines(catalog: Catalog, ticker: str) -> list[str]:
    try:
        df = catalog.query(
            f"SELECT date, close FROM ohlcv_daily_adj WHERE ticker = '{ticker}' ORDER BY date"
        )
    except Exception:
        return []
    if df.empty:
        return []
    close = df["close"].reset_index(drop=True)
    last = close.iloc[-1]
    lines = []
    for label, sessions in (("1M", 21), ("3M", 63), ("6M", 126), ("1A", 252)):
        if len(close) > sessions:
            ret = last / close.iloc[-1 - sessions] - 1.0
            lines.append(
                f"- **Retorno {label}** (`ohlcv_daily_adj.close`): {ret:+.1%}"
            )
    window = close.tail(252)
    high = window.max()
    if high > 0:
        lines.append(
            f"- **Distancia al máximo 252 sesiones** (`ohlcv_daily_adj.close`): "
            f"{last / high - 1.0:+.1%}"
        )
    return lines


def _sentiment_lines(catalog: Catalog, ticker: str) -> list[str]:
    try:
        df = catalog.query(
            f"SELECT date, sent_av, sent_finbert, n_articles FROM sentiment_daily "
            f"WHERE ticker = '{ticker}' ORDER BY date DESC LIMIT 5"
        )
    except Exception:
        return []
    if df.empty:
        return []
    latest = df.iloc[0]
    lines = [
        f"- **Sentimiento AV (último)** (`sentiment_daily.sent_av`): "
        f"{_fmt(latest['sent_av'])}",
        f"- **Sentimiento FinBERT (último)** (`sentiment_daily.sent_finbert`): "
        f"{_fmt(latest['sent_finbert'])}",
        f"- **Artículos (último día)** (`sentiment_daily.n_articles`): "
        f"{_fmt(latest['n_articles'])}",
    ]
    return lines


def generate_report(
    settings: Settings,
    ticker: str,
    llm: LLMClient | None = None,
    catalog_ro: Catalog | None = None,
) -> Path:
    ticker = ticker.upper()
    owns_catalog = catalog_ro is None
    if owns_catalog:
        catalog_ro = Catalog(settings, read_only=True)
    try:
        fundamentals = catalog_ro.query(
            f"SELECT * FROM fundamentals_snapshot WHERE ticker = '{ticker}' "
            f"ORDER BY as_of DESC LIMIT 1"
        )
        if fundamentals.empty:
            raise ValueError(
                f"{ticker} no está en fundamentals_snapshot — "
                f"ingesta primero con `qt fundamentals ingest`"
            )
        row = fundamentals.iloc[0]
        as_of = pd.Timestamp(row["as_of"]).date()

        lines: list[str] = [
            f"# {row.get('name', ticker)} ({ticker}) — "
            f"{_fmt(row.get('sector'))} / {_fmt(row.get('industry'))}",
            f"Fuente principal: fundamentals_snapshot (as_of {as_of}) — "
            f"precios propios: ohlcv_daily_adj",
            "",
        ]

        data_block: list[str] = []
        for section, fields in SECTIONS.items():
            data_block.append(f"## {section}")
            for label, col in fields:
                value = row.get(col) if col in fundamentals.columns else None
                data_block.append(
                    f"- **{label}** (`fundamentals_snapshot.{col}`): {_fmt(value)}"
                )
            data_block.append("")

        momentum = _momentum_lines(catalog_ro, ticker)
        data_block.append("## Momentum / Retornos (precios propios, ajustados)")
        data_block.extend(momentum or ["- n/d (sin precios curados para este ticker)"])
        data_block.append("")

        sentiment = _sentiment_lines(catalog_ro, ticker)
        if sentiment:
            data_block.append("## Sentimiento de noticias")
            data_block.extend(sentiment)
            data_block.append("")

        data_block.append("## Analistas / Estimaciones (datos de terceros, informados)")
        for label, col in ANALYST_SECTION:
            value = row.get(col) if col in fundamentals.columns else None
            data_block.append(
                f"- **{label}** (`fundamentals_snapshot.{col}`): {_fmt(value)}"
            )
        data_block.append("")
        lines.extend(data_block)

        # LLM judgment — the only non-deterministic section
        lines.append("## Lo que el dato dice")
        judgment = _llm_judgment(settings, llm, "\n".join(data_block))
        lines.append(judgment)
        lines.append("")

        lines.append("## Lo que NO sabemos")
        lines.append(
            f"- fundamentals_snapshot es un snapshot estático (as_of {as_of}), "
            f"sesgado por supervivencia; sin fundamentales point-in-time."
        )
        lines.append("- Cobertura de una sola fuente para fundamentales; sin filings cruzados.")
        lines.append("- Sin predicción de precios: este informe nunca la incluirá.")
        lines.append("")
        lines.append(f"*{CLOSING_LINE}*")

        settings.reports_dir.mkdir(parents=True, exist_ok=True)
        out = settings.reports_dir / f"report_{ticker}.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        return out
    finally:
        if owns_catalog:
            catalog_ro.close()


def persist_screener_result(result: ScreenerResult, settings: Settings) -> Path:
    """Write the full screener case (proposal, review, verification, cost) to reports/."""
    run_id = uuid4().hex[:12]
    lines: list[str] = [
        f"# Screener — run `{run_id}`",
        "",
        f"**Mandato:** {result.mandate}",
        f"**Estado:** {result.status}"
        + (" (corregido tras revisión)" if result.corrected else ""),
        f"**Rondas SQL:** {result.rounds}",
        "",
    ]
    if result.refusal:
        lines += ["## Negativa del agente", "", result.refusal, ""]
    if result.proposal is not None:
        p = result.proposal
        lines += ["## Candidatos", "", "| ticker | tesis | métricas citadas |", "|---|---|---|"]
        for c in p.candidates:
            cited = "; ".join(f"{m.source_view}.{m.column}={m.value}" for m in c.metrics)
            lines.append(f"| {c.ticker} | {c.thesis} | {cited} |")
        lines += ["", "## Metodología", "", p.methodology, ""]
        if p.caveats:
            lines += ["## Caveats", ""] + [f"- {c}" for c in p.caveats] + [""]
    if result.review is not None:
        verdict = "PASS" if result.review.pass_ else "FAIL"
        lines += [f"## Revisor independiente: {verdict}", ""]
        lines += [f"- {issue}" for issue in result.review.issues]
        if result.review.fix:
            lines.append(f"- Sugerencia: {result.review.fix}")
        lines.append("")
    if result.verification is not None:
        lines += ["## Verificación determinista", "", "```",
                  result.verification.render(), "```", ""]
    if result.usage is not None:
        lines += ["## Coste", "", result.usage.summary_line(settings.agent_model), ""]
    if result.transcript:
        lines += ["## Transcript (resumen)", ""]
        lines += [f"- {step.get('action')}: " +
                  (step.get("sql", step.get("reason", str(step.get("tickers", "")))) or "")
                  for step in result.transcript]
        lines.append("")

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    out = settings.reports_dir / f"screener_{run_id}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _llm_judgment(settings: Settings, llm: LLMClient | None, data_block: str) -> str:
    import os

    has_key = bool(
        settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
    )
    if llm is None and not has_key:
        return "(sección LLM omitida — sin API key configurada)"
    llm = llm or LLMClient(settings)
    try:
        response = llm.create(
            system=LLM_SYSTEM,
            max_tokens=1000,
            messages=[{"role": "user", "content": data_block}],
        )
        texts = [
            getattr(b, "text", "") for b in response.content
            if getattr(b, "type", None) == "text"
        ]
        return "\n".join(t for t in texts if t).strip() or "(sin texto del modelo)"
    except Exception:  # noqa: BLE001 — the report must still be written
        logger.warning("LLM judgment failed; omitting section", exc_info=True)
        return "(sección LLM no disponible — error de API)"
