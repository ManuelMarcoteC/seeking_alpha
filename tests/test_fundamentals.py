from datetime import date

import pandas as pd
import pytest

from qtdata.fundamentals import SNAPSHOT_NOTE, ingest_screener_csv
from qtdata.storage import parquet_store


@pytest.fixture
def screener_csv(tmp_path):
    path = tmp_path / "screener_mini.csv"
    pd.DataFrame(
        {
            "symbol": ["aapl", "MSFT", "GOOGL", "NVDA", "BAD"],
            "name": ["Apple", "Microsoft", "Alphabet", "NVIDIA", "Bad Co"],
            "sector": ["Technology", "Technology", "Communication", "Technology", "Energy"],
            "exchange": ["NASDAQ"] * 5,
            "marketCap": ["3000000000000", "2800000000000", "2100000000000",
                          "2900000000000", "n/a"],
            "peRatio": ["29.1", "33.4", "24.7", "65.2", "—"],
            "roe": ["1.47", "0.39", "0.27", "0.91", "0.05"],
            "ipoDate": ["1980-12-12", "1986-03-13", "2004-08-19", "1999-01-22", "2020-01-01"],
        }
    ).to_csv(path, index=False)
    return path


def test_ingest_screener_csv(settings, catalog, screener_csv):
    n = ingest_screener_csv(settings, catalog, screener_csv, as_of=date(2026, 5, 22))
    assert n == 5

    df = parquet_store.read(settings.curated_dir / "fundamentals_snapshot")
    assert len(df) == 5
    assert (df["note"] == SNAPSHOT_NOTE).all()
    assert "SURVIVORSHIP" in SNAPSHOT_NOTE.upper() or "biased" in SNAPSHOT_NOTE
    assert set(df["ticker"]) == {"AAPL", "MSFT", "GOOGL", "NVDA", "BAD"}  # upper-cased

    # numeric coercion: >=80% convertible columns adopted, dirty cells -> NaN
    assert pd.api.types.is_float_dtype(df["marketCap"])
    assert pd.api.types.is_float_dtype(df["peRatio"])
    assert df.loc[df["ticker"] == "BAD", "marketCap"].isna().all()
    # text columns untouched
    assert df["sector"].dtype == object
    assert df["ipoDate"].dtype == object  # *Date columns never coerced

    # raw snapshot exists and is immutable
    raw_files = list((settings.raw_dir / "provider=stockanalysis").rglob("*.parquet"))
    assert len(raw_files) == 1


def test_ingest_is_idempotent_per_as_of(settings, catalog, screener_csv):
    ingest_screener_csv(settings, catalog, screener_csv, as_of=date(2026, 5, 22))
    # re-ingesting the same snapshot date replaces on (as_of, ticker), no duplicates
    ingest_screener_csv(settings, catalog, screener_csv, as_of=date(2026, 5, 22))
    df = parquet_store.read(settings.curated_dir / "fundamentals_snapshot")
    assert len(df) == 5
    # each run appends its own raw payload (raw stays append-only)
    raw_files = list((settings.raw_dir / "provider=stockanalysis").rglob("*.parquet"))
    assert len(raw_files) == 2


def test_view_is_queryable(settings, catalog, screener_csv):
    ingest_screener_csv(settings, catalog, screener_csv, as_of=date(2026, 5, 22))
    out = catalog.query(
        "SELECT ticker, peRatio FROM fundamentals_snapshot "
        "WHERE sector = 'Technology' ORDER BY peRatio DESC"
    )
    assert list(out["ticker"]) == ["NVDA", "MSFT", "AAPL"]


def test_rejects_non_screener_csv(settings, catalog, tmp_path):
    bad = tmp_path / "x.csv"
    bad.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="symbol"):
        ingest_screener_csv(settings, catalog, bad, as_of=date(2026, 5, 22))
