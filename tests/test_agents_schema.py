from datetime import date

from qtdata.agents.schema_context import build_schema_context, build_system_prompt
from qtdata.fundamentals import ingest_screener_csv


def _seed(settings, catalog, tmp_path):
    import pandas as pd

    csv = tmp_path / "mini.csv"
    pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "sector": ["Technology", "Technology"],
            "roe": ["1.4", "0.4"],
        }
    ).to_csv(csv, index=False)
    ingest_screener_csv(settings, catalog, csv, as_of=date(2026, 5, 22))


def test_schema_context_lists_views_and_columns(settings, catalog, tmp_path):
    _seed(settings, catalog, tmp_path)
    ctx = build_schema_context(catalog)
    assert "## fundamentals_snapshot" in ctx
    assert '"roe"' in ctx
    assert "ej. [" in ctx  # example values present


def test_schema_context_is_deterministic(settings, catalog, tmp_path):
    """Byte-stable across builds — anything else silently breaks the prompt cache."""
    _seed(settings, catalog, tmp_path)
    assert build_schema_context(catalog) == build_schema_context(catalog)
    assert build_system_prompt(catalog) == build_system_prompt(catalog)


def test_system_prompt_carries_hard_rules(settings, catalog, tmp_path):
    _seed(settings, catalog, tmp_path)
    prompt = build_system_prompt(catalog)
    assert "NUNCA predigas precios" in prompt
    assert "run_sql" in prompt
    assert "NORMALIZA" in prompt


def test_missing_views_are_skipped(settings, catalog):
    # empty lake: no curated tables yet -> no view sections, no crash
    ctx = build_schema_context(catalog)
    assert "## ohlcv_daily" not in ctx
