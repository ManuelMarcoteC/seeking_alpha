from pathlib import Path

from qtdata.config import Settings


def test_defaults(tmp_path):
    s = Settings(data_dir=tmp_path, _env_file=None)
    assert s.raw_dir == tmp_path / "raw"
    assert s.curated_dir == tmp_path / "curated"
    assert s.catalog_path == tmp_path / "catalog.duckdb"
    assert s.default_provider == "yfinance"
    assert s.alpha_vantage_api_key is None


def test_env_override(monkeypatch):
    monkeypatch.setenv("QT_DATA_DIR", "X:/lake")
    monkeypatch.setenv("QT_MAD_THRESHOLD", "6.5")
    s = Settings(_env_file=None)
    assert s.data_dir == Path("X:/lake")
    assert s.mad_threshold == 6.5


def test_empty_api_key_is_none(monkeypatch):
    monkeypatch.setenv("QT_ALPHA_VANTAGE_API_KEY", "")
    s = Settings(_env_file=None)
    assert s.alpha_vantage_api_key is None


def test_api_key_is_secret(monkeypatch):
    monkeypatch.setenv("QT_ALPHA_VANTAGE_API_KEY", "sekret")
    s = Settings(_env_file=None)
    assert "sekret" not in repr(s)
    assert s.alpha_vantage_api_key.get_secret_value() == "sekret"


def test_news_dedup_defaults_off():
    s = Settings(_env_file=None)
    assert s.news_dedup_enabled is False
    assert s.news_dedup_threshold == 0.5


def test_news_dedup_env_override(monkeypatch):
    monkeypatch.setenv("QT_NEWS_DEDUP_ENABLED", "true")
    monkeypatch.setenv("QT_NEWS_DEDUP_THRESHOLD", "0.6")
    s = Settings(_env_file=None)
    assert s.news_dedup_enabled is True
    assert s.news_dedup_threshold == 0.6
