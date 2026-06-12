from datetime import date, timedelta

from qtdata.storage import parquet_store
from qtdata.universe import BIAS_NOTE, SP500_SEED, members_as_of, seed_universe


def test_seed_list_is_large_and_clean():
    assert len(SP500_SEED) >= 450
    assert len(set(SP500_SEED)) == len(SP500_SEED)
    assert all(t == t.upper() and " " not in t for t in SP500_SEED)
    assert "AAPL" in SP500_SEED
    assert "BRK-B" in SP500_SEED  # yfinance-style class shares


def test_seed_and_members_as_of(settings):
    today = date(2026, 6, 12)
    n = seed_universe(settings, as_of=today)
    assert n == len(SP500_SEED)
    members = members_as_of(settings, today)
    assert "AAPL" in members
    assert len(members) == len(SP500_SEED)


def test_membership_is_not_backdated(settings):
    """The biased seed is effective only FROM the seed date — never before."""
    seed_day = date(2026, 6, 12)
    seed_universe(settings, as_of=seed_day)
    assert members_as_of(settings, seed_day - timedelta(days=1)) == []


def test_bias_note_recorded_in_table(settings):
    seed_universe(settings, as_of=date(2026, 6, 12))
    df = parquet_store.read(settings.curated_dir / "universe_membership")
    assert (df["note"] == BIAS_NOTE).all()
    assert "SURVIVORSHIP" in BIAS_NOTE


def test_seed_is_idempotent(settings):
    seed_universe(settings, as_of=date(2026, 6, 12))
    seed_universe(settings, as_of=date(2026, 6, 12))
    df = parquet_store.read(settings.curated_dir / "universe_membership")
    assert len(df) == len(SP500_SEED)
