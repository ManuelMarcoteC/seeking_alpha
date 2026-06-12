from pathlib import Path

import pytest
from typer.testing import CliRunner

import qtdata.cli as cli
import qtdata.nasdaq_directory as nasdaq_directory
from qtdata.config import Settings
from qtdata.storage import parquet_store

runner = CliRunner()

FIXTURE = (Path(__file__).parent / "fixtures" / "nasdaqlisted_sample.txt").read_text()


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path / "data", _env_file=None)
    monkeypatch.setattr(cli, "get_settings", lambda: s)
    monkeypatch.setattr(nasdaq_directory, "download_directory", lambda settings: FIXTURE)
    return s


def test_refresh_initial_snapshot_warns(isolated_settings):
    runner.invoke(cli.app, ["init"])
    result = runner.invoke(cli.app, ["universe", "refresh", "--as-of", "2026-06-12"])
    assert result.exit_code == 0, result.output
    assert "common stocks=4" in result.output
    assert "+4" in result.output
    assert "INITIAL SNAPSHOT" in result.output

    members = parquet_store.read(isolated_settings.curated_dir / "universe_membership")
    assert sorted(members["ticker"]) == ["AAPL", "DEFI", "GOOGL", "MSFT"]


def test_refresh_second_run_is_noop_diff(isolated_settings):
    runner.invoke(cli.app, ["init"])
    runner.invoke(cli.app, ["universe", "refresh", "--as-of", "2026-06-12"])
    result = runner.invoke(cli.app, ["universe", "refresh", "--as-of", "2026-06-12"])
    assert result.exit_code == 0, result.output
    assert "+0 / -0 (unchanged 4)" in result.output
    assert "INITIAL SNAPSHOT" not in result.output


def test_refresh_registers_views(isolated_settings):
    runner.invoke(cli.app, ["init"])
    runner.invoke(cli.app, ["universe", "refresh", "--as-of", "2026-06-12"])
    result = runner.invoke(
        cli.app, ["query", "SELECT COUNT(*) AS n FROM listing_directory"]
    )
    assert result.exit_code == 0, result.output
    assert "13" in result.output
