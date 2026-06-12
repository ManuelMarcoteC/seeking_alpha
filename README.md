# qtdata — Quantitative Data Pipeline

Provider-agnostic market-data pipeline with institutional data-quality discipline:

- **Medallion storage** — `data/raw/` (immutable vendor payloads, append-only) →
  `data/curated/` (validated canonical parquet, hive-partitioned) → DuckDB research views.
- **Unadjusted prices as truth** — corporate actions live in their own table; adjusted
  series (`ohlcv_daily_adj`) are derived on read with CRSP-style factors, so vendor
  restatements flow through instead of corrupting stored history.
- **Flag, never mutate** — schema violations are quarantined before promotion; statistical
  anomalies (rolling-MAD outliers, stale prices, zero-volume runs, unexplained gaps,
  missing sessions) become rows in `validation_flags`, and the prices stay exactly as the
  vendor sent them. A −12% crash day is research signal, not noise to "smooth".
- **No look-ahead** — no `bfill` anywhere (enforced by a test); outlier statistics use
  trailing windows only; the universe table is point-in-time (`members_as_of(date)`).
- **Incremental & auditable** — watermark-based fetches, idempotent upserts, an ingestion
  manifest with payload hashes, and atomic parquet writes (crash-safe on Windows).
- **Cross-source reconciliation** — compare any two providers' raw layers with tolerance
  bands and get a discrepancy report.

## Setup

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
pip install -e . --no-deps
copy .env.example .env   # optional: add QT_ALPHA_VANTAGE_API_KEY
```

## Quick start

```powershell
qt init
qt universe seed --index SP500        # static snapshot — survivorship-biased, says so itself

# Offline demo (no API keys, deterministic synthetic data)
qt ingest --tickers AAA,BBB --provider synthetic --start 2024-01-02 --end 2024-06-28
qt curate
qt query "SELECT * FROM ohlcv_daily_adj LIMIT 5"

# Real data (yfinance, keyless)
qt ingest --tickers AAPL,MSFT --start 2020-01-02
qt curate
qt status

# Daily loop for the whole universe
qt update --universe SP500
```

Re-running `qt ingest` only fetches sessions after each ticker's watermark; `qt curate`
only processes raw files it hasn't promoted yet. Both are idempotent.

## Research views (DuckDB)

| view | contents |
|---|---|
| `ohlcv_daily` | curated unadjusted prices + lineage (`source`, `run_id`) |
| `ohlcv_daily_adj` | adjusted-on-read prices (`close_raw`, `adj_factor` included) |
| `ohlcv_daily_clean` | prices joined with flag counts/types per row |
| `corporate_actions` | splits & dividends |
| `validation_flags` | every anomaly, with severity and JSON detail |
| `universe_membership` | point-in-time index membership |

Query from the CLI (`qt query "..."`), or connect directly:

```python
import duckdb
con = duckdb.connect("data/catalog.duckdb", read_only=True)
df = con.execute("SELECT * FROM ohlcv_daily_adj WHERE ticker='AAPL'").df()
```

## Providers

| provider | datasets | notes |
|---|---|---|
| `yfinance` | OHLCV, actions | keyless; scraper — reconcile before trusting (`qt reconcile`) |
| `alpha_vantage` | OHLCV, actions | needs `QT_ALPHA_VANTAGE_API_KEY`; free tier ≈ 25 req/day |
| `synthetic` | OHLCV, actions | deterministic GBM with injectable events; tests & demos |

## Tests

```powershell
pytest            # offline suite (82 tests, no network)
pytest -m live    # optional live yfinance smoke test
ruff check src tests
```

## Design notes / roadmap

The architecture decisions (why unadjusted prices, why flag-never-mutate, why the
universe table is interval-based) and the forward roadmap (point-in-time fundamentals
via SEC EDGAR, survivorship-free membership via Norgate/Sharadar, intraday, Dagster
orchestration, data versioning, total-return series) are documented in the module
docstrings and the project plan.
