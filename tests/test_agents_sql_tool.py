"""Hostile-input tests for the agent SQL guard — the security boundary."""

import duckdb
import pytest

from qtdata.agents.sql_tool import SQLGuardError, guard_sql, run_sql


@pytest.fixture
def conn():
    c = duckdb.connect(":memory:")
    c.execute("CREATE TABLE t AS SELECT range AS i, 'x' || range AS s FROM range(100)")
    yield c
    c.close()


HOSTILE = [
    "INSTALL httpfs",
    "LOAD httpfs",
    "COPY t TO 'out.csv'",
    "ATTACH 'other.db'",
    "CREATE TABLE evil AS SELECT 1",
    "INSERT INTO t VALUES (1, 'x')",
    "UPDATE t SET i = 0",
    "DELETE FROM t",
    "DROP TABLE t",
    "PRAGMA database_list",
    "SET memory_limit='1GB'",
    "EXPORT DATABASE 'dir'",
    "SELECT * FROM read_parquet('secret.parquet')",
    "SELECT * FROM read_csv('x.csv')",
    "SELECT * FROM glob('*')",
    "SELECT getenv('PATH')",
    "SELECT 1; DROP TABLE t",
    "SELECT 1 /* hidden */; DELETE FROM t",
    "-- comment\nDROP TABLE t",
    "",
    "   ",
]


@pytest.mark.parametrize("sql", HOSTILE)
def test_guard_rejects_hostile_input(sql):
    with pytest.raises(SQLGuardError):
        guard_sql(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM t",
        "select i, s from t where i > 5 order by i desc",
        "WITH x AS (SELECT i FROM t) SELECT count(*) FROM x",
        "SELECT * FROM t;",  # single trailing semicolon tolerated
        "SELECT i -- inline comment\nFROM t",
        "SELECT percent_rank() OVER (ORDER BY i) FROM t",
    ],
)
def test_guard_accepts_legitimate_queries(sql):
    assert guard_sql(sql)


def test_run_sql_returns_rejection_text(conn):
    out = run_sql(conn, "DROP TABLE t")
    assert out.startswith("REJECTED:")


def test_run_sql_returns_sql_error_text(conn):
    out = run_sql(conn, "SELECT nope FROM t")
    assert out.startswith("SQL ERROR:")


def test_run_sql_row_and_col_caps(conn):
    out = run_sql(conn, "SELECT * FROM t", row_cap=10)
    assert "[truncated to first 10 rows]" in out
    conn.execute(
        "CREATE TABLE wide AS SELECT "
        + ", ".join(f"{i} AS c{i}" for i in range(30))
    )
    out = run_sql(conn, "SELECT * FROM wide", col_cap=5)
    assert "[truncated to first 5 columns]" in out


def test_run_sql_happy_path(conn):
    out = run_sql(conn, "SELECT count(*) AS n FROM t")
    assert "100" in out


def test_readonly_connection_blocks_writes(settings, catalog):
    """Even if the guard were bypassed, the read-only connection refuses writes."""
    from qtdata.storage.catalog import Catalog

    catalog.init_schema()
    catalog.close()  # DuckDB: read-only open requires no rw connection in-process
    ro = Catalog(settings, read_only=True)
    try:
        with pytest.raises(duckdb.Error):
            ro.conn.execute("CREATE TABLE pwned AS SELECT 1")
    finally:
        ro.close()
