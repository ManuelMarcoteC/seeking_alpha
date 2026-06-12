import pytest
from typer.testing import CliRunner

import qtdata.cli as cli
from qtdata.config import Settings

runner = CliRunner()

MINI_CSV = """symbol,name,sector,industry,peRatio,roe
aapl,Apple Inc.,Technology,Consumer Electronics,32.5,1.47
MSFT,Microsoft Corporation,Technology,Software,35.1,0.39
KO,Coca-Cola Co,Consumer Staples,Beverages,24.2,0.42
"""


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path / "data", _env_file=None)
    monkeypatch.setattr(cli, "get_settings", lambda: s)
    return s


def test_fundamentals_ingest_via_cli(isolated_settings, tmp_path):
    csv = tmp_path / "screener.csv"
    csv.write_text(MINI_CSV)
    runner.invoke(cli.app, ["init"])
    result = runner.invoke(
        cli.app, ["fundamentals", "ingest", str(csv), "--as-of", "2026-06-12"]
    )
    assert result.exit_code == 0, result.output
    assert "3 tickers" in result.output
    assert "survivorship" in result.output

    result = runner.invoke(
        cli.app,
        ["query", "SELECT ticker, sector FROM fundamentals_snapshot ORDER BY ticker"],
    )
    assert result.exit_code == 0, result.output
    assert "AAPL" in result.output  # lowercased symbol normalized


def test_fundamentals_ingest_rejects_non_screener_csv(isolated_settings, tmp_path):
    csv = tmp_path / "other.csv"
    csv.write_text("a,b\n1,2\n")
    runner.invoke(cli.app, ["init"])
    result = runner.invoke(cli.app, ["fundamentals", "ingest", str(csv)])
    assert result.exit_code != 0
