import hashlib

import pandas as pd
import pytest

from qtdata.storage import parquet_store
from tests.conftest import make_ohlcv


def _content_hash(table_dir) -> str:
    df = parquet_store.read(table_dir)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()


def test_upsert_and_read_roundtrip(tmp_path):
    df = make_ohlcv(n=50)
    df["year"] = df["date"].dt.year
    res = parquet_store.upsert(df, tmp_path / "t", ["ticker", "date"], partition_col="year")
    assert res.rows_written == 50
    assert res.rows_new == 50
    out = parquet_store.read(tmp_path / "t")
    assert len(out) == 50


def test_upsert_is_idempotent(tmp_path):
    df = make_ohlcv(n=50)
    df["year"] = df["date"].dt.year
    parquet_store.upsert(df, tmp_path / "t", ["ticker", "date"], partition_col="year")
    h1 = _content_hash(tmp_path / "t")
    parquet_store.upsert(df, tmp_path / "t", ["ticker", "date"], partition_col="year")
    h2 = _content_hash(tmp_path / "t")
    assert h1 == h2
    assert len(parquet_store.read(tmp_path / "t")) == 50


def test_upsert_new_row_wins_on_collision(tmp_path):
    df = make_ohlcv(n=10)
    parquet_store.upsert(df, tmp_path / "t", ["ticker", "date"])
    updated = df.iloc[[4]].copy()
    updated["close"] = 999.0
    res = parquet_store.upsert(updated, tmp_path / "t", ["ticker", "date"])
    assert res.rows_replaced == 1
    out = parquet_store.read(tmp_path / "t")
    assert len(out) == 10
    assert out.loc[out["date"] == df.loc[4, "date"], "close"].iloc[0] == 999.0


def test_partition_routing(tmp_path):
    df = make_ohlcv(start="2023-12-20", n=15)  # straddles the year boundary
    df["year"] = df["date"].dt.year
    parquet_store.upsert(df, tmp_path / "t", ["ticker", "date"], partition_col="year")
    assert (tmp_path / "t" / "year=2023" / "part-0.parquet").exists()
    assert (tmp_path / "t" / "year=2024" / "part-0.parquet").exists()
    out = parquet_store.read(tmp_path / "t")
    assert len(out) == 15
    assert "year" in out.columns  # hive partition restored on read


def test_write_raw_refuses_overwrite(tmp_path):
    df = make_ohlcv(n=5)
    path = tmp_path / "raw" / "x.parquet"
    parquet_store.write_raw(df, path)
    with pytest.raises(FileExistsError):
        parquet_store.write_raw(df, path)


def test_crash_mid_write_leaves_original_intact(tmp_path, monkeypatch):
    df = make_ohlcv(n=20)
    parquet_store.upsert(df, tmp_path / "t", ["ticker", "date"])
    h1 = _content_hash(tmp_path / "t")

    original = pd.DataFrame.to_parquet

    def explode(self, *args, **kwargs):
        path = args[0] if args else kwargs.get("path")
        # write garbage then fail, simulating a torn write to the temp file
        with open(path, "wb") as fh:
            fh.write(b"garbage")
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", explode)
    update = df.iloc[[0]].copy()
    update["close"] = 1.0
    with pytest.raises(OSError):
        parquet_store.upsert(update, tmp_path / "t", ["ticker", "date"])
    monkeypatch.setattr(pd.DataFrame, "to_parquet", original)

    assert _content_hash(tmp_path / "t") == h1  # stranded temp never poisons reads


def test_read_missing_table_returns_empty(tmp_path):
    assert parquet_store.read(tmp_path / "nope").empty


def test_upsert_empty_frame_is_noop(tmp_path):
    res = parquet_store.upsert(pd.DataFrame(), tmp_path / "t", ["ticker"])
    assert res.rows_written == 0
    assert not (tmp_path / "t").exists()


def test_read_unifies_null_and_string_columns_across_partitions(tmp_path):
    """Regression: a column that is all-null in one partition (Arrow ``null``)
    and string in another must read back without ArrowNotImplementedError."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table_dir = tmp_path / "news"
    # Partition 2025: the column has a concrete string value.
    p25 = table_dir / "year=2025"
    p25.mkdir(parents=True)
    pq.write_table(
        pa.table({"article_id": ["a"], "finbert_revision": ["rev123"]}),
        p25 / "part-0.parquet",
    )
    # Partition 2026: the column is entirely null -> stored as Arrow ``null``.
    p26 = table_dir / "year=2026"
    p26.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {"article_id": ["b"], "finbert_revision": pa.array([None], type=pa.null())}
        ),
        p26 / "part-0.parquet",
    )

    out = parquet_store.read(table_dir)
    assert len(out) == 2
    revs = set(out["finbert_revision"].dropna())
    assert revs == {"rev123"}
    assert out["finbert_revision"].isna().sum() == 1


def test_read_unifies_mixed_timestamp_precision(tmp_path):
    """Partitions with ns vs us timestamp precision must read back unified."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table_dir = tmp_path / "t"
    pa_ns = pa.array([pd.Timestamp("2025-01-01", tz="UTC")], type=pa.timestamp("ns", "UTC"))
    pa_us = pa.array([pd.Timestamp("2026-01-01", tz="UTC")], type=pa.timestamp("us", "UTC"))
    (table_dir / "year=2025").mkdir(parents=True)
    (table_dir / "year=2026").mkdir(parents=True)
    pq.write_table(
        pa.table({"id": ["a"], "scored_at": pa_ns}),
        table_dir / "year=2025" / "part-0.parquet",
    )
    pq.write_table(
        pa.table({"id": ["b"], "scored_at": pa_us}),
        table_dir / "year=2026" / "part-0.parquet",
    )

    out = parquet_store.read(table_dir)
    assert len(out) == 2
    assert pd.api.types.is_datetime64_any_dtype(out["scored_at"])


def test_read_with_filters_still_works(tmp_path):
    df = make_ohlcv(n=10)
    df["year"] = df["date"].dt.year
    parquet_store.upsert(df, tmp_path / "t", ["ticker", "date"], partition_col="year")
    out = parquet_store.read(tmp_path / "t", filters=[("ticker", "=", df.loc[0, "ticker"])])
    assert len(out) == 10
    assert (out["ticker"] == df.loc[0, "ticker"]).all()
