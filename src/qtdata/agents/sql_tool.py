"""Read-only SQL tool for the agent layer.

Defense in depth: (1) the connection is opened read-only at the DuckDB level,
(2) `guard_sql` statically rejects anything that is not a single SELECT/WITH
statement, (3) results are capped in rows and columns before reaching the
model's context. SQL errors come back as text so the agent can self-correct;
guard violations come back as "REJECTED: ..." so it learns the boundary.
"""

from __future__ import annotations

import re
import threading

import duckdb
import pandas as pd


class SQLGuardError(ValueError):
    pass


_COMMENT_LINE = re.compile(r"--[^\n]*")
_COMMENT_BLOCK = re.compile(r"/\*.*?\*/", re.DOTALL)
_BANNED_KEYWORDS = re.compile(
    r"\b(attach|detach|copy|install|load|pragma|create|insert|update|delete|drop|"
    r"alter|export|import|call|set|reset|vacuum|checkpoint|begin|commit|use|grant)\b",
    re.IGNORECASE,
)
_BANNED_FUNCTIONS = re.compile(r"\b(read_\w+|glob|getenv|st_read)\s*\(", re.IGNORECASE)


def guard_sql(sql: str) -> str:
    """Return the cleaned single SELECT statement or raise SQLGuardError."""
    cleaned = _COMMENT_BLOCK.sub(" ", _COMMENT_LINE.sub(" ", sql)).strip()
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].rstrip()
    if not cleaned:
        raise SQLGuardError("empty statement")
    if ";" in cleaned:
        raise SQLGuardError("multiple statements are not allowed")
    if not re.match(r"^(select|with)\b", cleaned, re.IGNORECASE):
        raise SQLGuardError("only SELECT/WITH queries are allowed")
    if m := _BANNED_KEYWORDS.search(cleaned):
        raise SQLGuardError(f"keyword not allowed: {m.group(1).upper()}")
    if m := _BANNED_FUNCTIONS.search(cleaned):
        raise SQLGuardError(f"function not allowed: {m.group(0).strip()}")
    return cleaned


def _render(df: pd.DataFrame, row_cap: int, col_cap: int) -> str:
    notes = []
    if len(df.columns) > col_cap:
        df = df.iloc[:, :col_cap]
        notes.append(f"[truncated to first {col_cap} columns]")
    truncated_rows = len(df) > row_cap
    if truncated_rows:
        df = df.head(row_cap)
        notes.append(f"[truncated to first {row_cap} rows]")
    if df.empty:
        body = "(0 rows)"
    else:
        body = df.to_string(index=False, max_colwidth=60)
    return "\n".join([body, *notes])


def run_sql(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    *,
    row_cap: int = 50,
    col_cap: int = 20,
    timeout_s: int = 15,
) -> str:
    """Execute one guarded read-only query; always returns text for the model."""
    try:
        cleaned = guard_sql(sql)
    except SQLGuardError as exc:
        return f"REJECTED: {exc}"

    timer = threading.Timer(timeout_s, conn.interrupt)
    timer.start()
    try:
        df = conn.execute(cleaned).df()
    except Exception as exc:  # noqa: BLE001 — model-facing error text
        return f"SQL ERROR: {exc}"
    finally:
        timer.cancel()
    return _render(df, row_cap, col_cap)
