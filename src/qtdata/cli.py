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
news_app = typer.Typer(help="News & sentiment pipeline")
app.add_typer(news_app, name="news")
fundamentals_app = typer.Typer(help="Fundamentals snapshots (static, survivorship-biased)")
app.add_typer(fundamentals_app, name="fundamentals")
agent_app = typer.Typer(help="LLM agent layer (read-only SQL, never price predictions)")
app.add_typer(agent_app, name="agent")
research_app = typer.Typer(help="Quantitative research / factor validation")
app.add_typer(research_app, name="research")

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


@universe_app.command("refresh")
def universe_refresh(
    index: str = typer.Option("NASDAQ", help="Universe to refresh"),
    as_of: str = typer.Option(None, help="ISO date; default = today"),
) -> None:
    """Refresh membership from the NASDAQ symbol directory (forward point-in-time)."""
    from qtdata.nasdaq_directory import INITIAL_SNAPSHOT_NOTE, refresh_nasdaq

    settings = get_settings()
    summary = refresh_nasdaq(
        settings,
        as_of=date.fromisoformat(as_of) if as_of else None,
        index_name=index,
    )
    with Catalog(settings) as cat:
        cat.init_schema()
        cat.refresh_views()
    console.print(
        f"directory rows={summary.directory_rows} common stocks={summary.common_stocks}"
    )
    console.print(
        f"[green]{index} as of {summary.as_of}: +{len(summary.added)} / "
        f"-{len(summary.removed)} (unchanged {summary.unchanged})[/green]"
    )
    # added-only with nothing pre-existing == the very first snapshot
    if summary.added and not summary.removed and summary.unchanged == 0:
        console.print(f"[yellow]{INITIAL_SNAPSHOT_NOTE}[/yellow]")


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


@news_app.command("ingest")
def news_ingest(
    provider: str = typer.Option(
        "alpha_vantage_news", help="alpha_vantage_news | yfinance_news"
    ),
    date_from: str = typer.Option(
        None, "--from", help="ISO date (firehose); default = watermark + 1 day"
    ),
    date_to: str = typer.Option(
        None, "--to", help="ISO date (firehose); default = last completed session"
    ),
    tickers: str = typer.Option(None, help="Comma-separated tickers (yfinance_news)"),
    universe: str = typer.Option(None, help="Use members of this universe (yfinance_news)"),
) -> None:
    """Fetch news into the immutable raw layer (firehose or per-ticker harvest)."""
    from qtdata.models import ProviderNotConfiguredError
    from qtdata.news.ingest import ingest_news

    settings = get_settings()
    symbols = None
    if provider == "yfinance_news":
        symbols = _resolve_tickers(settings, tickers, universe)
    elif provider == "alpha_vantage_news" and settings.alpha_vantage_api_key is None:
        console.print(
            "[red]alpha_vantage_news requires QT_ALPHA_VANTAGE_API_KEY "
            "(free tier: 25 req/day, ~1 firehose day per calendar day).[/red]"
        )
        raise typer.Exit(1)
    with Catalog(settings) as cat:
        cat.init_schema()
        try:
            summary = ingest_news(
                settings,
                cat,
                date_from=date.fromisoformat(date_from) if date_from else None,
                date_to=date.fromisoformat(date_to) if date_to else None,
                provider_name=provider,
                tickers=symbols,
            )
        except ProviderNotConfiguredError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
    _print_ingest_summary(summary)


@news_app.command("curate")
def news_curate() -> None:
    """Promote raw news to curated news_articles + news_ticker_sentiment."""
    from qtdata.news.curate import curate_news

    settings = get_settings()
    with Catalog(settings) as cat:
        cat.init_schema()
        summary = curate_news(settings, cat)
    console.print(
        f"news: files={summary.files_processed} rows={summary.rows_upserted} "
        f"quarantined={summary.rows_quarantined}"
    )


@news_app.command("score")
def news_score(
    batch_size: int = typer.Option(32, help="FinBERT batch size"),
    limit: int = typer.Option(None, help="Max rows this run (checkpoint lever)"),
) -> None:
    """Score curated headlines with the pinned FinBERT revision."""
    from qtdata.news.scoring import score_pending

    settings = get_settings()
    with Catalog(settings) as cat:
        cat.init_schema()
        try:
            n = score_pending(settings, cat, batch_size=batch_size, limit=limit)
        except ImportError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
    console.print(f"[green]FinBERT scored {n} rows.[/green]")


@news_app.command("build-factor")
def news_build_factor(
    since: str = typer.Option(None, help="Only rebuild (ticker, day) rows from this ISO date"),
) -> None:
    """(Re)build the daily sentiment factor `sentiment_daily`."""
    from qtdata.news.aggregate import build_sentiment_daily

    settings = get_settings()
    with Catalog(settings) as cat:
        cat.init_schema()
        n = build_sentiment_daily(
            settings, cat, since=date.fromisoformat(since) if since else None
        )
    console.print(f"[green]sentiment_daily: {n} (ticker, day) rows upserted.[/green]")


@news_app.command("update")
def news_update(
    universe: str = typer.Option(None, help="Universe for the harvest; default from settings"),
    skip_firehose: bool = typer.Option(False, help="Skip the Alpha Vantage firehose"),
    skip_score: bool = typer.Option(False, help="Skip FinBERT scoring"),
) -> None:
    """Daily news loop: firehose + harvest -> curate -> score -> factor."""
    from qtdata.news.aggregate import build_sentiment_daily
    from qtdata.news.curate import curate_news
    from qtdata.news.ingest import ingest_news
    from qtdata.news.scoring import score_pending

    settings = get_settings()
    symbols = _resolve_tickers(settings, None, universe or settings.default_universe)
    with Catalog(settings) as cat:
        cat.init_schema()
        if skip_firehose or settings.alpha_vantage_api_key is None:
            if not skip_firehose:
                console.print(
                    "[yellow]No QT_ALPHA_VANTAGE_API_KEY — skipping firehose.[/yellow]"
                )
        else:
            _print_ingest_summary(
                ingest_news(settings, cat, provider_name="alpha_vantage_news")
            )
        _print_ingest_summary(
            ingest_news(settings, cat, provider_name="yfinance_news", tickers=symbols)
        )
        summary = curate_news(settings, cat)
        console.print(
            f"news: files={summary.files_processed} rows={summary.rows_upserted} "
            f"quarantined={summary.rows_quarantined}"
        )
        if not skip_score:
            try:
                n = score_pending(settings, cat)
                console.print(f"FinBERT scored {n} rows")
            except ImportError:
                console.print(
                    "[yellow]FinBERT extras not installed — scores stay pending "
                    "(pip install -r requirements-sentiment.txt).[/yellow]"
                )
        n = build_sentiment_daily(settings, cat)
    console.print(f"[green]sentiment_daily: {n} (ticker, day) rows upserted.[/green]")


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


@fundamentals_app.command("ingest")
def fundamentals_ingest(
    csv: Path = typer.Argument(..., help="stockanalysis.com screener export (CSV)"),
    as_of: str = typer.Option(None, help="ISO snapshot date; default = today"),
) -> None:
    """Load a screener CSV into curated fundamentals_snapshot (agent/research use only)."""
    from qtdata.fundamentals import SNAPSHOT_NOTE, ingest_screener_csv

    settings = get_settings()
    with Catalog(settings) as cat:
        cat.init_schema()
        n = ingest_screener_csv(
            settings, cat, csv,
            as_of=date.fromisoformat(as_of) if as_of else date.today(),
        )
    console.print(f"[green]fundamentals_snapshot: {n} tickers ingested.[/green]")
    console.print(f"[yellow]{SNAPSHOT_NOTE}[/yellow]")


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


def _require_anthropic_key(settings: Settings) -> None:
    import os

    if settings.anthropic_api_key is None and not os.environ.get("ANTHROPIC_API_KEY"):
        console.print(
            "[red]No Anthropic key — set QT_ANTHROPIC_API_KEY (or ANTHROPIC_API_KEY).[/red]"
        )
        raise typer.Exit(1)


@agent_app.command("screener")
def agent_screener(
    mandate: str = typer.Argument(..., help="Natural-language screening mandate"),
    max_rounds: int = typer.Option(None, help="SQL exploration round cap"),
    review: bool = typer.Option(True, "--review/--no-review", help="Independent reviewer pass"),
    reports: bool = typer.Option(False, help="Also write a one-pager per candidate"),
) -> None:
    """Run the screener agent: mandate -> SQL loop -> reviewed, verified proposal."""
    from qtdata.agents.report import generate_report, persist_screener_result
    from qtdata.agents.screener import run_screener

    settings = get_settings()
    _require_anthropic_key(settings)
    result = run_screener(settings, mandate, max_rounds=max_rounds, review_enabled=review)

    console.print(f"[bold]Estado:[/bold] {result.status} (rondas SQL: {result.rounds})")
    if result.refusal:
        console.print(f"[yellow]El agente se negó: {result.refusal}[/yellow]")
    if result.proposal is not None:
        table = Table(show_header=True, header_style="bold")
        table.add_column("ticker")
        table.add_column("tesis")
        for c in result.proposal.candidates:
            table.add_row(c.ticker, c.thesis)
        console.print(table)
        if result.review is not None:
            verdict = "[green]PASS[/green]" if result.review.pass_ else "[red]FAIL[/red]"
            console.print(f"Revisor: {verdict} {result.review.issues or ''}")
        if result.verification is not None:
            console.print(result.verification.render())
    if result.usage is not None:
        console.print(f"[dim]{result.usage.summary_line(settings.agent_model)}[/dim]")

    out = persist_screener_result(result, settings)
    console.print(f"Informe: {out}")

    if reports and result.proposal is not None:
        import duckdb

        for c in result.proposal.candidates:
            try:
                path = generate_report(settings, c.ticker)
                console.print(f"  one-pager: {path}")
            except (ValueError, duckdb.Error) as exc:
                console.print(f"  [yellow]{c.ticker}: {exc}[/yellow]")


@agent_app.command("report")
def agent_report(ticker: str = typer.Argument(..., help="Ticker in fundamentals_snapshot")) -> None:
    """One-pager report for a ticker (works keyless — LLM section omitted)."""
    import duckdb

    from qtdata.agents.report import generate_report

    settings = get_settings()
    try:
        out = generate_report(settings, ticker)
    except (ValueError, duckdb.Error) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[green]Informe: {out}[/green]")


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


@research_app.command("sentiment-ic")
def research_sentiment_ic(
    horizons: str = typer.Option("1,5,20", help="Comma-separated forward horizons (sessions)"),
    score: str = typer.Option("sent_finbert", help="sent_finbert | sent_av"),
    min_breadth: int = typer.Option(10, help="Min names per day for a cross-sectional IC"),
    event_threshold: float = typer.Option(0.5, help="|score| threshold for the event study"),
    min_articles: int = typer.Option(3, help="Min articles behind an event day"),
    start: str = typer.Option(None, help="ISO date — factor sample start"),
    end: str = typer.Option(None, help="ISO date — factor sample end"),
) -> None:
    """Validate the sentiment factor: Spearman IC, decay curve, event study."""
    from qtdata.research.sentiment_validation import run_sentiment_validation

    settings = get_settings()
    hz = tuple(int(t.strip()) for t in horizons.split(",") if t.strip())
    with Catalog(settings) as cat:
        cat.init_schema()
        cat.refresh_views()
        try:
            report = run_sentiment_validation(
                settings,
                cat,
                horizons=hz,
                score_col=score,
                min_breadth=min_breadth,
                event_threshold=event_threshold,
                min_articles=min_articles,
                start=date.fromisoformat(start) if start else None,
                end=date.fromisoformat(end) if end else None,
            )
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc

    table = Table(show_header=True, header_style="bold")
    for col in ("horizonte", "n días", "IC medio", "t-stat", "ICIR", "hit rate"):
        table.add_column(col)
    for s in report.ic:
        table.add_row(
            str(s.horizon), str(s.n_days), f"{s.mean_ic:.4f}",
            f"{s.t_stat:.2f}", f"{s.icir:.2f}", f"{s.hit_rate:.2f}",
        )
    console.print(table)
    if report.events is not None:
        console.print(
            f"Event study: {report.events.n_pos} eventos +, {report.events.n_neg} eventos −"
        )
    console.print(f"[green]Informe: {report.path}[/green]")


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
    universe: str = typer.Option(None, help="Universe to update; default from settings"),
    provider: str = typer.Option(None),
) -> None:
    """Convenience daily loop: ingest -> curate -> refresh views for a universe."""
    from qtdata.curation.curate import curate_all

    settings = get_settings()
    symbols = _resolve_tickers(settings, None, universe or settings.default_universe)
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
