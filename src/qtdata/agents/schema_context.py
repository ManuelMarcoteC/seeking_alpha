"""Runtime schema discovery -> byte-stable system prompt for the agent layer.

The schema text is the cached prompt prefix: it must be deterministic across
rounds and cases (sorted views, sorted-by-position columns, NO dates, counts or
anything that changes between runs) or every call pays full input price.
"""

from __future__ import annotations

import pandas as pd

from qtdata.storage.catalog import Catalog

AGENT_VIEWS = (
    "fundamentals_snapshot",
    "ohlcv_daily",
    "ohlcv_daily_adj",
    "ohlcv_daily_clean",
    "universe_membership",
    "listing_directory",
    "sentiment_daily",
    "sentiment_daily_decayed",
    "validation_flags",
)

_VIEW_BLURBS = {
    "fundamentals_snapshot": (
        "valoración/calidad/crecimiento/balance por ticker — SNAPSHOT estático "
        "sesgado (ver columna note); ~300 columnas"
    ),
    "ohlcv_daily": "precios diarios SIN ajustar (verdad curada) + linaje",
    "ohlcv_daily_adj": "precios ajustados-en-lectura (splits+dividendos); usar para retornos",
    "ohlcv_daily_clean": "ohlcv_daily + n_flags/flag_types de validación por fila",
    "universe_membership": "membresía point-in-time por índice (effective_from/to)",
    "listing_directory": "directorio NASDAQ datado (market_category, financial_status)",
    "sentiment_daily": "factor diario de sentimiento de noticias por ticker",
    "sentiment_daily_decayed": "sentimiento con carry-forward decaído (vista derivada)",
    "validation_flags": "anomalías detectadas (flag, nunca mutación)",
}


def _example_values(catalog: Catalog, view: str, column: str) -> str:
    try:
        rows = catalog.conn.execute(
            f'SELECT DISTINCT "{column}" FROM {view} '
            f'WHERE "{column}" IS NOT NULL ORDER BY 1 LIMIT 2'
        ).fetchall()
    except Exception:
        return ""
    vals = []
    for (v,) in rows:
        s = str(v)
        vals.append(s[:24] + "…" if len(s) > 24 else s)
    return ", ".join(vals)


def build_schema_context(catalog: Catalog, max_cols_per_view: int = 60) -> str:
    """Deterministic textual schema of the agent-visible views."""
    existing = {
        r[0] for r in catalog.conn.execute(
            "SELECT table_name FROM information_schema.tables"
        ).fetchall()
    }
    parts: list[str] = []
    for view in AGENT_VIEWS:  # fixed order
        if view not in existing:
            continue
        desc: pd.DataFrame = catalog.conn.execute(f"DESCRIBE {view}").df()
        lines = [f"## {view} — {_VIEW_BLURBS.get(view, '')}"]
        truncated = len(desc) > max_cols_per_view
        for _, row in desc.head(max_cols_per_view).iterrows():
            name, dtype = row["column_name"], row["column_type"]
            ex = _example_values(catalog, view, name)
            suffix = f"  ej. [{ex}]" if ex else ""
            lines.append(f'- "{name}" ({dtype}){suffix}')
        if truncated:
            lines.append(
                f"- … {len(desc) - max_cols_per_view} columnas más: "
                f"descúbrelas con DESCRIBE-style queries sobre information_schema.columns"
            )
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


SYSTEM_RULES = """\
Eres un analista buy-side cuantitativo. Tu ÚNICA vía de acceso a datos es la \
herramienta run_sql: una consulta SQL de SOLO LECTURA (SELECT/WITH, un único \
statement) sobre las vistas DuckDB descritas abajo.

Reglas duras (innegociables):
1. NUNCA predigas precios, retornos futuros ni des recomendaciones de compra/venta. \
Si el mandato lo exige, usa la herramienta `refuse` y explica por qué.
2. Cada cifra de tu propuesta final debe citar su origen como vista.columna \
(campo metrics de submit_proposal).
3. Para rankear o combinar métricas, NORMALIZA primero: winsoriza outliers y usa \
percentiles o z-scores frente al universo (funciones ventana SQL como \
percent_rank() OVER (...), o filtros de rango razonable). Nunca rankees datos crudos.
4. fundamentals_snapshot es un snapshot estático y sesgado (columna note): dilo en \
los caveats si lo usas.
5. Explora lo justo (1-3 sondeos suelen bastar) y entrega con submit_proposal en \
cuanto tengas un resultado defendible. Mira lo que devuelve cada consulta: si no \
cumple el mandato, reescribe.
6. Las consultas devuelven máximo 50 filas — selecciona columnas y agrega; no \
intentes traer tablas enteras.

Esquema disponible (vistas y columnas):

"""


def build_system_prompt(catalog: Catalog) -> str:
    return SYSTEM_RULES + build_schema_context(catalog)
