import pandas as pd
import pytest
from typer.testing import CliRunner

import qtdata.cli as cli
from qtdata.config import Settings

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path / "data", _env_file=None)
    monkeypatch.setattr(cli, "get_settings", lambda: s)
    return s


def test_init_creates_lake(isolated_settings):
    result = runner.invoke(cli.app, ["init"])
    assert result.exit_code == 0
    assert isolated_settings.raw_dir.exists()
    assert isolated_settings.catalog_path.exists()


def test_universe_seed_prints_bias_warning():
    runner.invoke(cli.app, ["init"])
    result = runner.invoke(cli.app, ["universe", "seed"])
    assert result.exit_code == 0
    assert "SURVIVORSHIP" in result.output


def test_full_synthetic_loop_via_cli(isolated_settings):
    assert runner.invoke(cli.app, ["init"]).exit_code == 0
    result = runner.invoke(
        cli.app,
        ["ingest", "--tickers", "AAA,BBB", "--provider", "synthetic",
         "--start", "2024-01-02", "--end", "2024-03-28"],
    )
    assert result.exit_code == 0, result.output
    assert "ok=2" in result.output or "ok=4" in result.output

    result = runner.invoke(cli.app, ["curate"])
    assert result.exit_code == 0, result.output

    result = runner.invoke(
        cli.app, ["query", "SELECT ticker, COUNT(*) AS n FROM ohlcv_daily GROUP BY ticker"]
    )
    assert result.exit_code == 0, result.output
    assert "AAA" in result.output

    result = runner.invoke(cli.app, ["status"])
    assert result.exit_code == 0, result.output
    assert "synthetic" in result.output


def test_query_export_to_csv(isolated_settings, tmp_path):
    runner.invoke(cli.app, ["init"])
    runner.invoke(
        cli.app,
        ["ingest", "--tickers", "AAA", "--provider", "synthetic",
         "--start", "2024-01-02", "--end", "2024-01-31", "--datasets", "ohlcv"],
    )
    runner.invoke(cli.app, ["curate"])
    out = tmp_path / "export.csv"
    result = runner.invoke(cli.app, ["query", "SELECT * FROM ohlcv_daily", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert len(pd.read_csv(out)) > 0


def test_ingest_requires_tickers_or_universe():
    runner.invoke(cli.app, ["init"])
    result = runner.invoke(cli.app, ["ingest"])
    assert result.exit_code == 1
