"""Tests for other_directory (B0): parsing, filtering, PIT roster accrual."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from qtdata.config import Settings
from qtdata.other_directory import (
    EXCHANGE_ROSTERS,
    filter_common_stocks,
    parse_other_listed,
    refresh_other,
)
from qtdata.storage import parquet_store

SAMPLE = """ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
A|Agilent Technologies, Inc. Common Stock|N|A|N|100|N|A
SPY|SPDR S&P 500 ETF Trust|P|SPY|Y|100|N|SPY
ZTEST|NYSE Test Symbol|N|ZTEST|N|100|Y|ZTEST
BRK PRA|Berkshire Preferred|N|BRK PRA|N|100|N|BRK-A
WSO|Watsco, Inc. Common Stock|N|WSO|N|100|N|WSO
CBOT|Cboe Common Stock|Z|CBOT|N|100|N|CBOT
AMEXX|Amex Name Common Stock|A|AMEXX|N|100|N|AMEXX
ARCAX|Arca Common Stock|P|ARCAX|N|100|N|ARCAX
WARR|Some Warrant Series A|N|WARR|N|100|N|WARR
File Creation Time: 0705202617:31|||||||
"""


def test_parse_strips_footer_and_maps_columns():
    df = parse_other_listed(SAMPLE)
    assert len(df) == 9
    assert list(df.columns) == [
        "ticker", "security_name", "exchange", "cqs_symbol", "etf",
        "round_lot_size", "test_issue", "nasdaq_symbol",
    ]


def test_filter_common_stocks_nyse():
    df = parse_other_listed(SAMPLE)
    common = filter_common_stocks(df, "N")
    tickers = set(common["ticker"])
    assert "A" in tickers and "WSO" in tickers
    assert "SPY" not in tickers      # ETF (and Arca anyway)
    assert "ZTEST" not in tickers    # test issue
    assert "BRK PRA" not in tickers  # symbol with space -> not plain common
    assert "WARR" not in tickers     # warrant by name


def test_filter_is_per_exchange():
    df = parse_other_listed(SAMPLE)
    assert set(filter_common_stocks(df, "Z")["ticker"]) == {"CBOT"}
    assert set(filter_common_stocks(df, "A")["ticker"]) == {"AMEXX"}
    assert set(filter_common_stocks(df, "P")["ticker"]) == {"ARCAX"}


@pytest.fixture()
def tmp_settings(tmp_path):
    return Settings(data_dir=tmp_path)


def test_refresh_seeds_and_accrues_pit(tmp_settings):
    s1 = refresh_other(
        tmp_settings, as_of=date(2026, 7, 3), raw_text=SAMPLE, exchanges=("N", "P")
    )
    assert s1.per_roster["NYSE"]["common"] == 2
    assert s1.per_roster["NYSE"]["added"] == 2
    assert s1.per_roster["ARCA"]["added"] == 1

    members = parquet_store.read(tmp_settings.curated_dir / "universe_membership")
    nyse_open = members[(members["index_name"] == "NYSE") & members["effective_to"].isna()]
    assert set(nyse_open["ticker"]) == {"A", "WSO"}
    # rosters never mixed
    assert set(members["index_name"].unique()) == {"NYSE", "ARCA"}
    # initial snapshot note recorded (survivorship boundary documented)
    assert (nyse_open["note"].str.contains("INITIAL SNAPSHOT")).all()

    # day 2: WSO disappears -> its interval closes; A stays open
    sample2 = SAMPLE.replace("WSO|Watsco, Inc. Common Stock|N|WSO|N|100|N|WSO\n", "")
    s2 = refresh_other(
        tmp_settings, as_of=date(2026, 7, 4), raw_text=sample2, exchanges=("N",)
    )
    assert s2.per_roster["NYSE"]["removed"] == 1
    members = parquet_store.read(tmp_settings.curated_dir / "universe_membership")
    wso = members[(members["index_name"] == "NYSE") & (members["ticker"] == "WSO")]
    assert len(wso) == 1 and wso.iloc[0]["effective_to"] == pd.Timestamp(date(2026, 7, 4))
    a_row = members[(members["index_name"] == "NYSE") & (members["ticker"] == "A")]
    assert a_row.iloc[0]["effective_to"] is pd.NaT or pd.isna(a_row.iloc[0]["effective_to"])


def test_exchange_roster_map_complete():
    assert EXCHANGE_ROSTERS == {"N": "NYSE", "P": "ARCA", "A": "AMEX", "Z": "CBOE"}
