from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from qtdata.nasdaq_directory import (
    filter_common_stocks,
    parse_nasdaq_listed,
    refresh_nasdaq,
)
from qtdata.storage import parquet_store
from qtdata.universe import members_as_of

FIXTURE = (Path(__file__).parent / "fixtures" / "nasdaqlisted_sample.txt").read_text()


def test_parse_strips_footer_and_renames():
    df = parse_nasdaq_listed(FIXTURE)
    assert len(df) == 13  # footer stripped
    assert "ticker" in df.columns
    assert not df["ticker"].str.startswith("File").any()


def test_parse_rejects_schema_change():
    broken = "Symbol|Nombre\nAAPL|Apple\n"
    with pytest.raises(ValueError, match="schema changed"):
        parse_nasdaq_listed(broken)


def test_filter_common_stocks():
    common = filter_common_stocks(parse_nasdaq_listed(FIXTURE))
    kept = set(common["ticker"])
    # real common stocks survive, including Financial Status D (deficient-but-listed)
    assert kept == {"AAPL", "MSFT", "GOOGL", "DEFI"}
    # excluded: ETF, warrant, right, unit, preferred, notes, status E,
    # test issue, dotted test symbol
    for bad in ("QQQ", "ABCW", "DEFR", "GHIU", "JKLP", "MNOQ", "BADCO", "ZAZZT", "ZXYZ.A"):
        assert bad not in kept


def test_refresh_initial_snapshot(settings):
    s = refresh_nasdaq(settings, as_of=date(2026, 6, 12), raw_text=FIXTURE)
    assert s.directory_rows == 13
    assert s.common_stocks == 4
    assert sorted(s.added) == ["AAPL", "DEFI", "GOOGL", "MSFT"]
    assert s.removed == []

    members = members_as_of(settings, date(2026, 6, 12), index_name="NASDAQ")
    assert members == ["AAPL", "DEFI", "GOOGL", "MSFT"]
    # forward-PIT: nothing before the snapshot date
    assert members_as_of(settings, date(2026, 6, 11), index_name="NASDAQ") == []

    # raw snapshot is immutable and dated
    raw_files = list(
        (settings.raw_dir / "provider=nasdaq_trader").rglob("*.parquet")
    )
    assert len(raw_files) == 1
    # curated dated directory accrues
    listing = parquet_store.read(settings.curated_dir / "listing_directory")
    assert len(listing) == 13
    assert listing["is_common_stock"].sum() == 4


def test_refresh_diff_add_and_close(settings):
    refresh_nasdaq(settings, as_of=date(2026, 6, 12), raw_text=FIXTURE)

    # next day: MSFT delisted, NEWCO appears
    updated = FIXTURE.replace(
        "MSFT|Microsoft Corporation - Common Stock|Q|N|N|100|N|N\n", ""
    ).replace(
        "AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N",
        "AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N\n"
        "NEWCO|New Company Inc. - Common Stock|S|N|N|100|N|N",
    )
    s = refresh_nasdaq(settings, as_of=date(2026, 6, 13), raw_text=updated)
    assert s.added == ["NEWCO"]
    assert s.removed == ["MSFT"]

    # membership intervals: MSFT closed on the 13th, NEWCO open from the 13th
    assert "MSFT" in members_as_of(settings, date(2026, 6, 12), index_name="NASDAQ")
    after = members_as_of(settings, date(2026, 6, 13), index_name="NASDAQ")
    assert "MSFT" not in after
    assert "NEWCO" in after

    df = parquet_store.read(settings.curated_dir / "universe_membership")
    msft = df[df["ticker"] == "MSFT"].iloc[0]
    assert pd.Timestamp(msft["effective_to"]) == pd.Timestamp("2026-06-13")


def test_refresh_is_idempotent(settings):
    refresh_nasdaq(settings, as_of=date(2026, 6, 12), raw_text=FIXTURE)
    s2 = refresh_nasdaq(settings, as_of=date(2026, 6, 12), raw_text=FIXTURE)
    assert s2.added == []
    assert s2.removed == []
    assert s2.unchanged == 4
    df = parquet_store.read(settings.curated_dir / "universe_membership")
    assert len(df) == 4  # no duplicate intervals


def test_sp500_seed_coexists(settings):
    from qtdata.universe import seed_universe

    seed_universe(settings, as_of=date(2026, 6, 12))
    refresh_nasdaq(settings, as_of=date(2026, 6, 12), raw_text=FIXTURE)
    sp = members_as_of(settings, date(2026, 6, 12), index_name="SP500")
    nq = members_as_of(settings, date(2026, 6, 12), index_name="NASDAQ")
    assert len(sp) > 400
    assert nq == ["AAPL", "DEFI", "GOOGL", "MSFT"]
