import pandas as pd
import pytest

from qtdata.validation.schemas import ACTIONS_SCHEMA, OHLCV_SCHEMA, validate_frame
from tests.conftest import make_ohlcv


def test_clean_frame_passes():
    df = make_ohlcv(n=30)
    valid, failures = validate_frame(df, OHLCV_SCHEMA)
    assert len(valid) == 30
    assert failures.empty


def test_high_below_low_is_quarantined():
    df = make_ohlcv(n=30)
    df.loc[5, "high"] = df.loc[5, "low"] - 1.0
    valid, failures = validate_frame(df, OHLCV_SCHEMA)
    assert len(valid) == 29
    assert not failures.empty
    assert 5 in failures["index"].to_numpy()


def test_negative_price_is_quarantined():
    df = make_ohlcv(n=30)
    df.loc[10, "close"] = -3.0
    df.loc[10, "low"] = -3.0
    valid, failures = validate_frame(df, OHLCV_SCHEMA)
    assert len(valid) == 29


def test_null_volume_is_quarantined():
    df = make_ohlcv(n=30)
    df["volume"] = df["volume"].astype("Int64")
    df.loc[3, "volume"] = pd.NA
    valid, failures = validate_frame(df, OHLCV_SCHEMA)
    assert len(valid) == 29


def test_duplicate_key_is_flagged():
    df = make_ohlcv(n=30)
    dup = pd.concat([df, df.iloc[[7]]], ignore_index=True)
    valid, failures = validate_frame(dup, OHLCV_SCHEMA)
    assert not failures.empty


def test_missing_column_raises():
    df = make_ohlcv(n=10).drop(columns=["close"])
    with pytest.raises(ValueError, match="Structural"):
        validate_frame(df, OHLCV_SCHEMA)


def test_actions_schema_rejects_unknown_type():
    df = pd.DataFrame(
        {
            "ticker": ["TEST"],
            "ex_date": [pd.Timestamp("2024-05-01")],
            "action_type": ["merger"],
            "value": [1.0],
            "source": ["test"],
            "run_id": ["r"],
            "ingested_at": [pd.Timestamp.now(tz="UTC")],
        }
    )
    valid, failures = validate_frame(df, ACTIONS_SCHEMA)
    assert valid.empty
    assert not failures.empty


def _news_ticker_row(ticker: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "article_id": ["a" * 64],
            "ticker": [ticker],
            "published_at": [pd.Timestamp.now(tz="UTC")],
            "ingested_at": [pd.Timestamp.now(tz="UTC")],
            "relevance": [0.5],
            "score_av": [None],
            "score_finbert": [None],
            "finbert_revision": [None],
            "scored_at": [pd.NaT],
            "run_id": ["r"],
        }
    )


@pytest.mark.parametrize("ticker", ["AAPL", "2513.HK", "0100.HK", "005930.KS", "247540.KQ"])
def test_news_ticker_schema_accepts_market_suffixes(ticker):
    # Foreign tickers carry a market suffix (.HK/.KS/.KQ) and exceed 6 chars;
    # the schema must NOT quarantine them (regression: str_length was (1,6)).
    from qtdata.validation.schemas import NEWS_TICKER_SCHEMA

    valid, failures = validate_frame(_news_ticker_row(ticker), NEWS_TICKER_SCHEMA)
    assert len(valid) == 1, f"{ticker} was wrongly quarantined: {failures.to_dict()}"
    assert failures.empty
