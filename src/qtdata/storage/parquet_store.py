"""Partitioned Parquet storage with idempotent upserts and atomic replacement.

Write protocol: data goes to a `_`-prefixed temp file in the target directory,
then `os.replace` swaps it in. A crash mid-write leaves either the old or the
new partition, never a torn file — and pyarrow's dataset reader ignores
`_`/`.`-prefixed files, so stranded temps never poison reads.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class UpsertResult:
    rows_written: int = 0
    rows_new: int = 0
    rows_replaced: int = 0
    partitions: list[str] = field(default_factory=list)


def _write_atomic(df: pd.DataFrame, file: Path) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.parent / f"_{file.name}.tmp"
    if tmp.exists():
        tmp.unlink()
    df.to_parquet(tmp, engine="pyarrow", index=False)
    os.replace(tmp, file)


def write_raw(df: pd.DataFrame, path: Path) -> None:
    """Append-only write for the raw layer: refuses to overwrite an existing payload."""
    if path.exists():
        raise FileExistsError(f"Raw payload already exists (raw layer is immutable): {path}")
    _write_atomic(df, path)


def upsert(
    df: pd.DataFrame,
    table_dir: Path,
    key_cols: list[str],
    partition_col: str | None = None,
) -> UpsertResult:
    """Merge rows into a (optionally hive-partitioned) parquet table.

    On key collision the incoming row wins. Re-running with identical input is
    a no-op in content terms (idempotent).
    """
    result = UpsertResult()
    if df.empty:
        return result

    if partition_col is None:
        groups: list[tuple[object, pd.DataFrame]] = [(None, df)]
    else:
        groups = [(value, g) for value, g in df.groupby(partition_col, observed=True)]

    for value, g in groups:
        part_dir = table_dir if value is None else table_dir / f"{partition_col}={value}"
        file = part_dir / "part-0.parquet"
        out = g.drop(columns=[partition_col]) if partition_col else g

        replaced = 0
        if file.exists():
            existing = pd.read_parquet(file, engine="pyarrow")
            ex_keys = existing.set_index(key_cols).index
            new_keys = out.set_index(key_cols).index
            keep = existing[~ex_keys.isin(new_keys)]
            replaced = len(existing) - len(keep)
            if not keep.empty:
                out = pd.concat([keep, out], ignore_index=True)

        out = out.sort_values(key_cols, kind="stable").reset_index(drop=True)
        _write_atomic(out, file)

        result.rows_written += len(g)
        result.rows_new += len(g) - replaced
        result.rows_replaced += replaced
        result.partitions.append(str(part_dir.relative_to(table_dir)) if value is not None else ".")

    return result


def read(
    table_dir: Path,
    columns: list[str] | None = None,
    filters: list | None = None,
) -> pd.DataFrame:
    """Read a parquet table directory (hive partitions resolved automatically).

    Returns an empty DataFrame when the table does not exist yet.

    Partitions written at different times can disagree on a column's Arrow type:
    a column that is entirely null in one partition (e.g. ``finbert_revision``
    before scoring) is stored as Arrow ``null``, while another partition stores
    it as ``string``; mixing also happens with timestamp precision (``ns`` vs
    ``us``). A naive concat then raises ``ArrowNotImplementedError: Unsupported
    cast from string to null``. We unify the schema across all fragments before
    reading so null-typed columns are promoted to their concrete type.
    """
    import pyarrow as pa
    import pyarrow.dataset as pds

    if not table_dir.exists() or not any(table_dir.rglob("*.parquet")):
        return pd.DataFrame()

    dataset = pds.dataset(table_dir, format="parquet", partitioning="hive")

    # Unify per-fragment schemas, dropping pure-``null`` fields in favour of a
    # concrete type from any other fragment; promote ``null`` -> that type.
    fragment_schemas = [frag.physical_schema for frag in dataset.get_fragments()]
    if fragment_schemas:
        try:
            unified = pa.unify_schemas(
                [dataset.schema, *fragment_schemas], promote_options="permissive"
            )
            dataset = pds.dataset(
                table_dir, format="parquet", partitioning="hive", schema=unified
            )
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError, TypeError):
            # Fall back to the dataset's own schema; the table() call below still
            # benefits from Arrow's per-fragment casting where possible.
            pass

    table = dataset.to_table(columns=columns, filter=_build_filter(filters))
    return table.to_pandas()


def _build_filter(filters: list | None):
    """Translate the legacy ``pd.read_parquet`` filter list into a pyarrow
    dataset expression. ``None`` when no filters are supplied."""
    if not filters:
        return None
    import operator

    import pyarrow.dataset as pds

    ops = {
        "=": operator.eq,
        "==": operator.eq,
        "!=": operator.ne,
        "<": operator.lt,
        "<=": operator.le,
        ">": operator.gt,
        ">=": operator.ge,
    }
    expr = None
    for col, op, val in filters:  # DNF single-conjunction form is all we use
        field = pds.field(col)
        if op in ("in", "not in"):
            field_expr = field.isin(list(val))
            if op == "not in":
                field_expr = ~field_expr
        else:
            field_expr = ops[op](field, val)
        expr = field_expr if expr is None else (expr & field_expr)
    return expr
