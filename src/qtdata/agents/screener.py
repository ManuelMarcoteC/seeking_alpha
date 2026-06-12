"""Screener agent: natural-language mandate -> SQL exploration loop -> proposal.

The webinar pattern with native tool use: the model explores via the read-only
`run_sql` tool, delivers via the strict `submit_proposal` tool (validated
Pydantic schema), or declines via `refuse` (price predictions are forbidden by
system rule). An independent reviewer grades the proposal; one surgical
correction round max; deterministic verification always runs at the end.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, ValidationError

from qtdata.agents.llm import LLMClient, UsageMeter
from qtdata.agents.reviewer import ReviewVerdict, review
from qtdata.agents.schema_context import build_system_prompt
from qtdata.agents.sql_tool import run_sql
from qtdata.agents.verification import VerificationReport, verify_proposal
from qtdata.config import Settings
from qtdata.storage.catalog import Catalog

logger = logging.getLogger(__name__)


class MetricCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    column: str
    source_view: str
    value: float | str


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ticker: str
    thesis: str
    metrics: list[MetricCitation]


class Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidates: list[Candidate]
    methodology: str
    caveats: list[str]


def _tools() -> list[dict]:
    """Fixed, byte-stable tool list (order matters for the prompt cache)."""
    return [
        {
            "name": "run_sql",
            "description": (
                "Ejecuta UNA consulta SQL de solo lectura (SELECT/WITH) sobre las "
                "vistas DuckDB del esquema. Devuelve como máximo 50 filas como texto. "
                "Errores SQL vuelven como texto para que corrijas la consulta."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"sql": {"type": "string", "description": "La consulta SQL"}},
                "required": ["sql"],
                "additionalProperties": False,
            },
        },
        {
            "name": "submit_proposal",
            "description": (
                "Entrega tu propuesta final. Cada métrica citada debe existir como "
                "columna real de la vista indicada. Llámala exactamente una vez."
            ),
            "strict": True,
            "input_schema": Proposal.model_json_schema(),
        },
        {
            "name": "refuse",
            "description": (
                "Niégate cuando el mandato exija predecir precios/retornos futuros "
                "o recomendaciones de compra/venta, explicando por qué."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
                "additionalProperties": False,
            },
        },
    ]


@dataclass
class ScreenerResult:
    mandate: str
    proposal: Proposal | None = None
    refusal: str | None = None
    review: ReviewVerdict | None = None
    verification: VerificationReport | None = None
    rounds: int = 0
    corrected: bool = False
    status: str = "max_rounds"  # final | refused | max_rounds
    usage: UsageMeter | None = None
    transcript: list[dict] = field(default_factory=list)


def _loop(
    llm: LLMClient,
    catalog_ro: Catalog,
    settings: Settings,
    system: list[dict],
    messages: list[dict],
    result: ScreenerResult,
    max_rounds: int,
) -> Proposal | None:
    """Run the explore loop until proposal/refusal/round-cap. Mutates messages."""
    tools = _tools()
    sql_rounds = 0
    force_final = False
    while True:
        kwargs: dict = {"system": system, "tools": tools, "messages": messages}
        if force_final:
            kwargs["tool_choice"] = {"type": "tool", "name": "submit_proposal"}
        response = llm.create(**kwargs)
        blocks = list(response.content)
        messages.append({"role": "assistant", "content": blocks})

        tool_results: list[dict] = []
        for block in blocks:
            btype = getattr(block, "type", None)
            if btype != "tool_use":
                continue
            if block.name == "submit_proposal":
                try:
                    proposal = Proposal.model_validate(block.input)
                except ValidationError as exc:
                    logger.warning("Invalid proposal payload: %s", exc)
                    return None
                result.transcript.append({"action": "submit_proposal",
                                          "tickers": [c.ticker for c in proposal.candidates]})
                return proposal
            if block.name == "refuse":
                result.refusal = str(block.input.get("reason", ""))
                result.status = "refused"
                result.transcript.append({"action": "refuse", "reason": result.refusal})
                return None
            if block.name == "run_sql":
                sql = str(block.input.get("sql", ""))
                obs = run_sql(
                    catalog_ro.conn, sql,
                    row_cap=settings.agent_sql_row_cap,
                    col_cap=settings.agent_sql_col_cap,
                    timeout_s=settings.agent_sql_timeout_s,
                )
                result.transcript.append({"action": "run_sql", "sql": sql,
                                          "observation_chars": len(obs)})
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": obs}
                )

        if tool_results:
            sql_rounds += 1
            result.rounds = sql_rounds
            messages.append({"role": "user", "content": tool_results})
            if sql_rounds >= max_rounds:
                force_final = True
            continue

        if force_final:  # forced submit produced nothing usable
            return None
        # no tool call at all (plain text turn): nudge once toward the tools
        if getattr(response, "stop_reason", None) == "end_turn":
            sql_rounds += 1
            result.rounds = sql_rounds
            if sql_rounds >= max_rounds:
                force_final = True
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Responde usando las herramientas: run_sql para consultar, "
                        "submit_proposal para entregar, refuse para negarte."
                    ),
                }
            )
            continue
        return None


def run_screener(
    settings: Settings,
    mandate: str,
    *,
    max_rounds: int | None = None,
    review_enabled: bool = True,
    llm: LLMClient | None = None,
    catalog_ro: Catalog | None = None,
) -> ScreenerResult:
    max_rounds = max_rounds or settings.agent_max_rounds
    llm = llm or LLMClient(settings)
    owns_catalog = catalog_ro is None
    if owns_catalog:
        catalog_ro = Catalog(settings, read_only=True)

    result = ScreenerResult(mandate=mandate, usage=llm.meter)
    try:
        system = [
            {
                "type": "text",
                "text": build_system_prompt(catalog_ro),
                "cache_control": {"type": "ephemeral"},
            }
        ]
        messages: list[dict] = [{"role": "user", "content": f"Mandato: {mandate}"}]
        proposal = _loop(llm, catalog_ro, settings, system, messages, result, max_rounds)

        if proposal is not None and review_enabled:
            verdict = review(llm, mandate, proposal.model_dump_json())
            result.review = verdict
            if not verdict.pass_:
                # ONE surgical correction: keep the book, fix only the issues
                correction = (
                    f"Un revisor independiente rechazó tu propuesta por: "
                    f"{json.dumps(verdict.issues, ensure_ascii=False)}. "
                    f"Sugerencia: {verdict.fix or '(sin sugerencia)'}\n"
                    "Parte de TU propuesta anterior y corrige SOLO eso (sustituye lo "
                    "justo). NO re-explores el universo entero. Entrega con "
                    "submit_proposal en 1-2 rondas."
                )
                messages.append({"role": "user", "content": correction})
                corrected = _loop(llm, catalog_ro, settings, system, messages, result, 2)
                if corrected is not None:
                    proposal = corrected
                    result.corrected = True
                    result.review = review(llm, mandate, proposal.model_dump_json())
                # else: keep the first proposal (webinar rule — always verifiable)

        if proposal is not None:
            result.proposal = proposal
            result.status = "final"
            result.verification = verify_proposal(
                catalog_ro, proposal, max_sector_pct=settings.agent_max_sector_pct
            )
    finally:
        if owns_catalog:
            catalog_ro.close()
    return result
