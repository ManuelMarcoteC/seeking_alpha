"""Independent reviewer (evaluator-optimizer): write -> grade -> rewrite.

Separating the doer from the judge reduces self-deception (webinar rule).
FAIL only on blocking issues, capped at 2; fail-open if the reviewer itself
breaks — a downed reviewer never kills a case.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, Field, field_validator

from qtdata.agents.llm import LLMClient

logger = logging.getLogger(__name__)


class ReviewVerdict(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    pass_: bool = Field(alias="pass")
    issues: list[str] = Field(default_factory=list)
    fix: str | None = None

    @field_validator("issues")
    @classmethod
    def _cap_issues(cls, v: list[str]) -> list[str]:
        return v[:2]  # belt: the prompt says max 2, the code enforces it


REVIEWER_SYS = """\
Eres un revisor independiente. Te dan un MANDATO y la propuesta final de un \
agente screener. Juzga si la propuesta cumple el mandato.

FAIL SOLO por problemas BLOQUEANTES, que son exactamente estos:
(a) viola el mandato (p.ej. un límite sectorial pedido y excedido),
(b) una métrica clave no aplica al activo (p.ej. ROIC como calidad en un banco),
(c) una cifra usada es indefendible (artefacto evidente, p.ej. ROIC de 1000%)
    o no cita su columna de origen,
(d) se rankeó sobre datos crudos sin normalizar cuando el mandato implica ranking.

Reglas duras:
- MÁXIMO 2 issues, los dos más graves.
- PROHIBIDO bloquear por: riesgos de mercado/geopolíticos, riesgo regulatorio,
  clasificaciones sectoriales discutibles, estilo, o mejoras opcionales.
- Si no hay nada bloqueante, pass=true aunque la propuesta sea mejorable.
- Cada issue en 1 frase; fix en 1 frase concreta.
"""


def review(llm: LLMClient, mandate: str, proposal_json: str) -> ReviewVerdict:
    """Grade the proposal; FAIL-OPEN on any reviewer malfunction."""
    try:
        response = llm.parse(
            system=REVIEWER_SYS,
            output_format=ReviewVerdict,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"MANDATO:\n{mandate}\n\nPROPUESTA DEL AGENTE (JSON):\n{proposal_json}"
                    ),
                }
            ],
        )
        verdict = response.parsed_output
        if verdict is None:
            raise ValueError("reviewer returned no parsed output")
        return verdict
    except Exception:  # noqa: BLE001 — availability over strictness, logged
        logger.warning("Reviewer failed; failing open (pass=True)", exc_info=True)
        return ReviewVerdict(**{"pass": True, "issues": ["(revisor no disponible — auto-PASS)"]})
