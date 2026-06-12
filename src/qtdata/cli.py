"""qt — command-line entry point for the pipeline.

Daily loop: `qt ingest --universe SP500` -> `qt curate` (or just `qt update`).
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from uuid import uuid4

import typer
from rich.console import Console
from rich.table import Table

from qtdata.config import Settings, get_settings
from qtdata.ingestion.ingest import DATASET_ALIASES, IngestSummary
from qtdata.ingestion.ingest import ingest as run_ingest
from qtdata.models import Dataset
from qtdata.storage.catalog import Catalog

app = typer.Typer(no_args_is_help=True, help="Quantitative data pipeline")
universe_app = typer.Typer(help="Point-in-time universe management")
app.add_typer(universe_app, name="universe")

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _resolve_tickers(
    settings: Settings, tickers: str | None, universe: str | None
) -> list[str]:
    if tickers:
        return [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if universe:
        from qtdata.universe import members_as_of

        members = members_as_of(settings, date.today(), index_name=universe)
        if not members:
            console.print(
                f"[red]Universe {universe!r} is empty — run `qt universe seed` first.[/red]"
            )
            raise typer.Exit(1)
        return members
    console.print("[red]Provide --tickers or --universe.[/red]")
    raise typer.Exit(1)


def _parse_datasets(datasets: str) -> tuple[Dataset, ...]:
    out = []
    for token in datasets.split(","):
        token = token.strip().lower()
        if token not in DATASET_ALIASES:
            console.print(f"[red]Unknown dataset {token!r} (use: ohlcv, actions)[/red]")
            raise typer.Exit(1)
        out.append(DATASET_ALIASES[token])
    return tuple(dict.fromkeys(out))


def _print_ingest_summary(s: IngestSummary) -> None:
    console.print(
        f"run [bold]{s.run_id}[/bold]: ok={s.ok} empty={s.empty} "
        f"skipped={s.skipped} failed={s.failed} rows={s.rows}"
    )
    for ticker, ds, err in s.failures[:10]:
        console.print(f"  [red]FAILED[/red] {ticker}/{ds}: {err}")
    if len(s.failures) > 10:
        console.print(f"  ... and {len(s.failures) - 10} more failures (see manifest)")


@app.command()
def init() -> None:
    """Create the data-lake directories and the DuckDB catalog schema."""
    settings = get_settings()
    for d in (settings.raw_dir, settings.curated_dir, settings.reports_dir):
        d.mkdir(parents=True, exist_ok=True)
    with Catalog(settings) as cat:
        cat.init_schema()
        cat.refresh_views()
    if not Path(".env").exists():
        console.print("[yellow]No .env found — copy .env.example if you need API keys.[/yellow]")
    console.print(f"[green]Initialized data lake at {settings.data_dir.resolve()}[/green]")


@universe_app.command("seed")
def universe_seed(index: str = typer.Option("SP500", help="Index to seed")) -> None:
    """Seed universe membership from the static snapshot (SURVIVORSHIP-BIASED)."""
    from qtdata.universe import BIAS_NOTE, seed_universe

    settings = get_settings()
    n = seed_universe(settings, index_name=index)
    with Catalog(settings) as cat:
        cat.init_schema()
        cat.refresh_views()
    console.print(f"[green]Seeded {n} members for {index}.[/green]")
    console.print(f"[yellow]{BIAS_NOTE}[/yellow]")


@app.command()
def ingest(
    tickers: str = typer.Option(None, help="Comma-separated tickers"),
    universe: str = typer.Option(None, help="Use members of this universe"),
    provider: str = typer.Option(None, help="yfinance | synthetic | alpha_vantage"),
    start: str = typer.Option(None, help="ISO date; default = watermark + 1 session"),
    end: str = typer.Option(None, help="ISO date; default = today"),
    full_refresh: bool = typer.Option(False, help="Ignore watermarks"),
    datasets: str = typer.Option("ohlcv,actions"),
) -> None:
    """Fetch provider data into the immutable raw layer (incremental by default)."""
    settings = get_settings()
    symbols = _resolve_tickers(settings, tickers, universe)
    with Catalog(settings) as cat:
        cat.init_schema()
        summary = run_ingest(
            settings,
            cat,
            symbols,
            provider_name=provider,
            start=date.fromisoformat(start) if start else None,
            end=date.fromisoformat(end) if end else None,
            datasets=_parse_datasets(datasets),
            full_refresh=full_refresh,
        )
    _print_ingest_summary(summary)


@app.command()
def curate(
    tickers: str = typer.Option(None, help="Limit to comma-separated tickers"),
) -> None:
    """Promote raw payloads to the curated layer (validate, quarantine, flag)."""
    from qtdata.curation.curate import curate_all

    settings = get_settings()
    subset = [t.strip().upper() for t in tickers.split(",")] if tickers else None
    with Catalog(settings) as cat:
        cat.init_schema()
        actions_summary, ohlcv_summary = curate_all(settings, cat, subset)
    console.print(
        f"actions: files={actions_summary.files_processed} rows={actions_summary.rows_upserted} "
        f"quarantined={actions_summary.rows_quarantined}"
    )
    console.print(
        f"ohlcv:   files={ohlcv_summary.files_processed} rows={ohlcv_summary.rows_upserted} "
        f"quarantined={ohlcv_summary.rows_quarantined} flags={ohlcv_summary.flags_written}"
    )


@app.command()
def validate(
    tickers: str = typer.Option(None, help="Limit to comma-separated tickers"),
) -> None:
    """Re-run anomaly detectors over the curated layer and regenerate the report."""
    from qtdata.storage import parquet_store
    from qtdata.validation.anomalies import run_detectors
    from qtdata.validation.report import ValidationReport, persist_report

    settings = get_settings()
    ohlcv = parquet_store.read(settings.curated_dir / "ohlcv_daily")
    if ohlcv.empty:
        console.print("[yellow]Curated layer is empty — run `qt curate` first.[/yellow]")
        raise typer.Exit(0)
    if tickers:
        subset = [t.strip().upper() for t in tickers.split(",")]
        ohlcv = ohlcv[ohlcv["ticker"].isin(subset)]
    actions = parquet_store.read(settings.curated_dir / "corporate_actions")
    flags = run_detectors(ohlcv, actions, settings)
    run_id = uuid4().hex[:12]
    persist_report(ValidationReport(run_id=run_id, flags=flags), settings)
    with Catalog(settings) as cat:
        cat.init_schema()
        cat.refresh_views()
    console.print(
        f"[green]{len(flags)} flags written; report: "
        f"{settings.reports_dir / f'validation_{run_id}.md'}[/green]"
    )


@app.command()
def reconcile(
    provider_a: str = typer.Option(..., help="First provider (raw layer)"),
    provider_b: str = typer.Option(..., help="Second provider (raw layer)"),
    tickers: str = typer.Option(None, help="Limit to comma-separated tickers"),
) -> None:
    """Cross-source comparison with tolerance bands; writes a discrepancy report."""
    from qtdata.reconciliation.reconcile import load_raw_frame, persist_reconciliation
    from qtdata.reconciliation.reconcile import reconcile as run_reconcile

    settings = get_settings()
    subset = [t.strip().upper() for t in tickers.split(",")] if tickers else None
    df_a = load_raw_frame(settings, provider_a, tickers=subset)
    df_b = load_raw_frame(settings, provider_b, tickers=subset)
    if df_a.empty or df_b.empty:
        console.print("[red]One of the providers has no raw data for that selection.[/red]")
        raise typer.Exit(1)
    result = run_reconcile(
        df_a,
        df_b,
        provider_a,
        provider_b,
        settings.reconcile_price_rel_tol,
        settings.reconcile_volume_rel_tol,
    )
    run_id = uuid4().hex[:12]
    persist_reconciliation(result, provider_a, provider_b, run_id, settings)
    if result.discrepancies.empty:
        console.print("[green]All compared rows match exactly.[/green]")
    else:
        counts = result.discrepancies["classification"].value_counts()
        for cls, n in counts.items():
            console.print(f"  {cls}: {n}")
        console.print(f"Report: {settings.reports_dir / f'reconcile_{run_id}.md'}")


@app.command()
def query(
    sql: str = typer.Argument(..., help="SQL over the research views"),
    out: Path = typer.Option(None, help="Write result to CSV/parquet by extension"),
) -> None:
    """Run SQL against the DuckDB views (ohlcv_daily, ohlcv_daily_adj, ...)."""
    settings = get_settings()
    with Catalog(settings) as cat:
        cat.init_schema()
        cat.refresh_views()
        df = cat.query(sql)
    if out is not None:
        if out.suffix == ".parquet":
            df.to_parquet(out, index=False)
        else:
            df.to_csv(out, index=False)
        console.print(f"[green]{len(df)} rows -> {out}[/green]")
        return
    table = Table(show_header=True, header_style="bold")
    for col in df.columns:
        table.add_column(str(col))
    for _, row in df.head(50).iterrows():
        table.add_row(*(str(v) for v in row))
    console.print(table)
    if len(df) > 50:
        console.print(f"... {len(df) - 50} more rows (use --out to export)")


@app.command()
def status() -> None:
    """Watermarks, recent ingestion runs and flag counts."""
    from qtdata.ingestion.manifest import recent_runs

    settings = get_settings()
    with Catalog(settings) as cat:
        cat.init_schema()
        wm = cat.query(
            "SELECT provider, dataset, COUNT(*) AS tickers, "
            "MIN(high_water_date) AS oldest, MAX(high_water_date) AS newest "
            "FROM watermarks GROUP BY provider, dataset"
        )
        runs = recent_runs(cat.conn, limit=10)
        views = cat.refresh_views()
        flags = (
            cat.query(
                "SELECT flag_type, severity, COUNT(*) AS n FROM validation_flags "
                "GROUP BY flag_type, severity ORDER BY n DESC"
            )
            if "validation_flags" in views
            else None
        )
    console.print("[bold]Watermarks[/bold]")
    console.print(wm if not wm.empty else "  (none)")
    console.print("[bold]Recent runs[/bold]")
    console.print(runs if not runs.empty else "  (none)")
    if flags is not None and not flags.empty:
        console.print("[bold]Validation flags[/bold]")
        console.print(flags)


@app.command()
def update(
    universe: str = typer.Option("SP500"),
    provider: str = typer.Option(None),
) -> None:
    """Convenience daily loop: ingest -> curate -> refresh views for a universe."""
    from qtdata.curation.curate import curate_all

    settings = get_settings()
    symbols = _resolve_tickers(settings, None, universe)
    with Catalog(settings) as cat:
        cat.init_schema()
        summary = run_ingest(settings, cat, symbols, provider_name=provider)
        _print_ingest_summary(summary)
        actions_summary, ohlcv_summary = curate_all(settings, cat)
    console.print(
        f"[green]curated rows: actions={actions_summary.rows_upserted} "
        f"ohlcv={ohlcv_summary.rows_upserted}; flags={ohlcv_summary.flags_written}[/green]"
    )


if __name__ == "__main__":
    app()
