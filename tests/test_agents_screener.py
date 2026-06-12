"""Screener loop mechanics with a scripted FakeAnthropic — no API, no network."""

from datetime import date

import pandas as pd
import pytest

from qtdata.agents.llm import LLMClient
from qtdata.agents.reviewer import ReviewVerdict
from qtdata.agents.screener import run_screener
from qtdata.fundamentals import ingest_screener_csv
from qtdata.storage.catalog import Catalog
from tests.fake_anthropic import (
    FakeAnthropic,
    parsed_response,
    response,
    text_block,
    tool_use,
)

PROPOSAL = {
    "candidates": [
        {
            "ticker": "AAPL",
            "thesis": "Calidad con caja neta",
            "metrics": [
                {"column": "roe", "source_view": "fundamentals_snapshot", "value": 1.4}
            ],
        },
        {
            "ticker": "XOM",
            "thesis": "Energía barata",
            "metrics": [
                {"column": "roe", "source_view": "fundamentals_snapshot", "value": 0.2}
            ],
        },
    ],
    "methodology": "ROE winsorizado, percentil sectorial",
    "caveats": ["fundamentals_snapshot es snapshot estático sesgado"],
}

PASS = ReviewVerdict(**{"pass": True})
FAIL = ReviewVerdict(**{"pass": False, "issues": ["concentración"], "fix": "diversifica"})


@pytest.fixture
def seeded(settings, catalog, tmp_path):
    csv = tmp_path / "mini.csv"
    pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "XOM"],
            "sector": ["Technology", "Technology", "Energy"],
            "roe": ["1.4", "0.4", "0.2"],
        }
    ).to_csv(csv, index=False)
    ingest_screener_csv(settings, catalog, csv, as_of=date(2026, 5, 22))
    catalog.close()  # DuckDB: read-only open requires no rw connection in-process
    ro = Catalog(settings, read_only=True)
    yield ro
    ro.close()


def _llm(settings, fake):
    return LLMClient(settings, client=fake)


def test_happy_path_sql_then_proposal(settings, seeded):
    fake = FakeAnthropic()
    fake.messages.create_queue = [
        response([tool_use("run_sql", {"sql": "SELECT ticker, roe FROM fundamentals_snapshot"})]),
        response([tool_use("submit_proposal", PROPOSAL)], stop_reason="end_turn"),
    ]
    fake.messages.parse_queue = [parsed_response(PASS)]

    res = run_screener(settings, "calidad diversificada", llm=_llm(settings, fake),
                       catalog_ro=seeded)
    assert res.status == "final"
    assert res.proposal is not None
    assert [c.ticker for c in res.proposal.candidates] == ["AAPL", "XOM"]
    assert res.rounds == 1
    assert res.review.pass_
    assert res.verification is not None and res.verification.passed
    # SQL observation was fed back as a tool_result (message index 2:
    # [mandate, assistant#1, tool_results]; the list mutates after capture)
    second_call = fake.messages.create_calls[1]
    contents = second_call["messages"][2]["content"]
    assert contents[0]["type"] == "tool_result"
    assert "AAPL" in contents[0]["content"]


def test_refusal_path(settings, seeded):
    fake = FakeAnthropic()
    fake.messages.create_queue = [
        response([tool_use("refuse", {"reason": "predicción de precios no defendible"})]),
    ]
    res = run_screener(settings, "¿qué acción subirá más?", llm=_llm(settings, fake),
                       catalog_ro=seeded)
    assert res.status == "refused"
    assert "defendible" in res.refusal
    assert res.proposal is None


def test_round_cap_forces_submit(settings, seeded):
    fake = FakeAnthropic()
    sql = tool_use("run_sql", {"sql": "SELECT 1"})
    fake.messages.create_queue = [
        response([sql]),
        response([sql]),
        # forced round (tool_choice) delivers the proposal
        response([tool_use("submit_proposal", PROPOSAL)], stop_reason="end_turn"),
    ]
    fake.messages.parse_queue = [parsed_response(PASS)]
    res = run_screener(settings, "mandato", max_rounds=2, llm=_llm(settings, fake),
                       catalog_ro=seeded)
    assert res.status == "final"
    forced = fake.messages.create_calls[2]
    assert forced.get("tool_choice") == {"type": "tool", "name": "submit_proposal"}


def test_reviewer_fail_open(settings, seeded):
    fake = FakeAnthropic()
    fake.messages.create_queue = [
        response([tool_use("submit_proposal", PROPOSAL)], stop_reason="end_turn"),
    ]
    fake.messages.parse_queue = [RuntimeError("reviewer exploded")]
    res = run_screener(settings, "mandato", llm=_llm(settings, fake), catalog_ro=seeded)
    assert res.status == "final"          # the case survived
    assert res.review.pass_               # fail-open
    assert "auto-PASS" in res.review.issues[0]


def test_one_correction_max_keeps_first_on_nonconvergence(settings, seeded):
    fake = FakeAnthropic()
    fake.messages.create_queue = [
        response([tool_use("submit_proposal", PROPOSAL)], stop_reason="end_turn"),
        # correction round: model rambles text, then forced submit yields garbage
        response([text_block("mmm")], stop_reason="end_turn"),
        response([text_block("nada")], stop_reason="end_turn"),
        response([text_block("sigo sin entregar")], stop_reason="end_turn"),
    ]
    fake.messages.parse_queue = [parsed_response(FAIL)]
    res = run_screener(settings, "mandato", llm=_llm(settings, fake), catalog_ro=seeded)
    # first proposal kept despite failed review + failed correction
    assert res.status == "final"
    assert res.proposal is not None
    assert not res.corrected
    assert not res.review.pass_


def test_correction_converges_and_rereviews(settings, seeded):
    corrected = dict(PROPOSAL)
    corrected = {**PROPOSAL, "methodology": "corregida"}
    fake = FakeAnthropic()
    fake.messages.create_queue = [
        response([tool_use("submit_proposal", PROPOSAL)], stop_reason="end_turn"),
        response([tool_use("submit_proposal", corrected)], stop_reason="end_turn"),
    ]
    fake.messages.parse_queue = [parsed_response(FAIL), parsed_response(PASS)]
    res = run_screener(settings, "mandato", llm=_llm(settings, fake), catalog_ro=seeded)
    assert res.corrected
    assert res.proposal.methodology == "corregida"
    assert res.review.pass_


def test_usage_meter_accumulates(settings, seeded):
    fake = FakeAnthropic()
    fake.messages.create_queue = [
        response([tool_use("run_sql", {"sql": "SELECT 1"})], in_=5000, out=300,
                 cache_write=4000),
        response([tool_use("submit_proposal", PROPOSAL)], stop_reason="end_turn",
                 in_=200, out=400, cache_read=4000),
    ]
    fake.messages.parse_queue = [parsed_response(PASS, in_=1000, out=50)]
    llm = _llm(settings, fake)
    run_screener(settings, "mandato", llm=llm, catalog_ro=seeded)
    assert llm.meter.calls == 3
    assert llm.meter.input_tokens == 6200
    assert llm.meter.output_tokens == 750
    assert llm.meter.cache_read_tokens == 4000
    assert llm.meter.cache_creation_tokens == 4000
    assert llm.meter.cost_usd("claude-opus-4-8") > 0


def test_system_prompt_is_cached_block(settings, seeded):
    fake = FakeAnthropic()
    fake.messages.create_queue = [
        response([tool_use("submit_proposal", PROPOSAL)], stop_reason="end_turn"),
    ]
    fake.messages.parse_queue = [parsed_response(PASS)]
    run_screener(settings, "mandato", llm=_llm(settings, fake), catalog_ro=seeded)
    system = fake.messages.create_calls[0]["system"]
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    # opus-4-8 contract: adaptive thinking, no temperature
    assert fake.messages.create_calls[0]["thinking"] == {"type": "adaptive"}
    assert "temperature" not in fake.messages.create_calls[0]
