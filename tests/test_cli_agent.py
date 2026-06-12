import pytest
from typer.testing import CliRunner

import qtdata.cli as cli
from qtdata.agents.llm import UsageMeter
from qtdata.agents.reviewer import ReviewVerdict
from qtdata.agents.screener import Candidate, MetricCitation, Proposal, ScreenerResult
from qtdata.agents.verification import Check, VerificationReport
from qtdata.config import Settings

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path / "data", anthropic_api_key="k", _env_file=None)
    monkeypatch.setattr(cli, "get_settings", lambda: s)
    return s


def _scripted_result() -> ScreenerResult:
    meter = UsageMeter(input_tokens=1_000, output_tokens=200, calls=2)
    proposal = Proposal(
        candidates=[
            Candidate(
                ticker="AAPL",
                thesis="calidad con sentimiento positivo",
                metrics=[
                    MetricCitation(
                        column="roe", source_view="fundamentals_snapshot", value=1.47
                    )
                ],
            )
        ],
        methodology="rank por ROE normalizado",
        caveats=["snapshot estático, sesgo de supervivencia"],
    )
    return ScreenerResult(
        mandate="calidad",
        proposal=proposal,
        review=ReviewVerdict(**{"pass": True}),
        verification=VerificationReport(
            checks=[Check("tickers existen", True, "todos en el universo")]
        ),
        rounds=2,
        status="final",
        usage=meter,
        transcript=[{"action": "run_sql", "sql": "SELECT 1", "observation_chars": 10}],
    )


def test_agent_screener_without_key_exits(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path / "data", _env_file=None)
    monkeypatch.setattr(cli, "get_settings", lambda: s)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = runner.invoke(cli.app, ["agent", "screener", "calidad"])
    assert result.exit_code == 1
    assert "QT_ANTHROPIC_API_KEY" in result.output


def test_agent_screener_prints_result_and_persists(isolated_settings, monkeypatch):
    monkeypatch.setattr(
        "qtdata.agents.screener.run_screener",
        lambda settings, mandate, **kw: _scripted_result(),
    )
    runner.invoke(cli.app, ["init"])
    result = runner.invoke(cli.app, ["agent", "screener", "calidad"])
    assert result.exit_code == 0, result.output
    assert "AAPL" in result.output
    assert "Verificación determinista" in result.output
    assert "llamadas" in result.output  # cost line from UsageMeter

    reports = list(isolated_settings.reports_dir.glob("screener_*.md"))
    assert len(reports) == 1
    text = reports[0].read_text(encoding="utf-8")
    assert "calidad" in text
    assert "fundamentals_snapshot.roe=1.47" in text
    assert "## Coste" in text


def test_agent_report_missing_ticker_exits(isolated_settings):
    runner.invoke(cli.app, ["init"])
    result = runner.invoke(cli.app, ["agent", "report", "AAPL"])
    assert result.exit_code == 1
    assert "fundamentals_snapshot" in result.output


def test_agent_report_keyless_omits_llm_section(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path / "data", _env_file=None)
    monkeypatch.setattr(cli, "get_settings", lambda: s)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    csv = tmp_path / "screener.csv"
    csv.write_text("symbol,name,sector,industry,roe\nAAPL,Apple,Tech,Hardware,1.47\n")
    runner.invoke(cli.app, ["init"])
    runner.invoke(cli.app, ["fundamentals", "ingest", str(csv), "--as-of", "2026-06-12"])
    result = runner.invoke(cli.app, ["agent", "report", "AAPL"])
    assert result.exit_code == 0, result.output
    report = (s.reports_dir / "report_AAPL.md").read_text(encoding="utf-8")
    assert "sección LLM omitida" in report
