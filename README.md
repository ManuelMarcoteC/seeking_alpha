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
- **NASDAQ point-in-time universe** — `qt universe refresh` diffs the daily NASDAQ Trader
  symbol directory against open membership intervals; run it daily and real PIT membership
  accrues (forward only — see *Known biases*).
- **News & sentiment layer** — Alpha Vantage firehose + yfinance harvest into an
  append-only, first-capture-wins news store; FinBERT scoring at a **pinned revision**;
  daily factor `sentiment_daily` built under strict PIT rules
  (`effective_ts = max(published_at, ingested_at)`, 15:30 ET cutoff → next session).
- **Agent layer (Claude)** — natural-language screening mandates over a single read-only
  SQL tool; independent reviewer, one surgical correction round, deterministic
  verification of every cited figure, cost metering. Refuses price prediction by rule.
- **Factor validation** — `qt research sentiment-ic`: cross-sectional Spearman IC vs
  forward returns, signal decay curve, event study — with the caveats printed in the
  report, always.

## Setup

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
pip install -e . --no-deps
copy .env.example .env   # optional: QT_ALPHA_VANTAGE_API_KEY, QT_ANTHROPIC_API_KEY

# optional extras
pip install -r requirements-sentiment.txt   # FinBERT scoring (torch CPU, ~2 GB)
pip install -r requirements-agents.txt      # agent layer (anthropic SDK)
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
```

## Full NASDAQ runbook

```powershell
qt universe refresh                       # NASDAQ directory -> PIT membership (~3,000 common stocks)
qt ingest --universe NASDAQ               # OHLCV, batched yfinance; resumable via watermarks
qt curate                                 # promote + validate (resumable via the file ledger)
qt validate                               # optional anomaly sweep + report

qt news ingest                            # AV firehose (~1 day of history per calendar day, free tier)
qt news ingest --provider yfinance_news --universe NASDAQ   # breadth harvest (recent stream)
qt news curate
qt news score --limit 5000                # FinBERT; repeat — resumes on unscored rows
qt news build-factor

qt fundamentals ingest Webinar_Agentes/screener_us.csv      # static snapshot for the agent layer
qt research sentiment-ic                  # IC / decay / event study report
qt agent screener "calidad con sentimiento positivo reciente, máx 40% por sector"
```

Daily loop thereafter: `qt universe refresh` + `qt update` + `qt news update`
(all stages watermarked/idempotent; defaults come from `QT_DEFAULT_UNIVERSE`).

## Research views (DuckDB)

| view | contents |
|---|---|
| `ohlcv_daily` | curated unadjusted prices + lineage (`source`, `run_id`) |
| `ohlcv_daily_adj` | adjusted-on-read prices (`close_raw`, `adj_factor` included) |
| `ohlcv_daily_clean` | prices joined with flag counts/types per row |
| `corporate_actions` | splits & dividends |
| `validation_flags` | every anomaly, with severity and JSON detail |
| `universe_membership` | point-in-time index membership |
| `listing_directory` | dated NASDAQ symbol directory snapshots |
| `fundamentals_snapshot` | static screener snapshot (survivorship-biased, research/agent only) |
| `news_articles` | deduped articles (first capture wins) |
| `news_ticker_sentiment` | per (article, ticker) relevance + vendor & FinBERT scores |
| `sentiment_daily` | daily sentiment factor per (ticker, session) |
| `sentiment_daily_decayed` | carry-forward factor decayed by exp(−days/τ), derived on read |

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
| `alpha_vantage_news` | news firehose | budget-guarded (23 of 25 free req/day); vendor relevance + sentiment |
| `yfinance_news` | news | keyless; recent per-ticker stream only — forward harvesting |

## Agent layer

`qt agent screener "MANDATO"` runs the webinar pattern: mandate → SQL exploration
loop (read-only `run_sql`, capped rows/cols/time) → schema-validated proposal →
**independent reviewer** (max 2 blocking issues) → at most ONE surgical correction →
**deterministic verification** (ticker existence, sector caps, cited columns, ±1%
numeric spot-checks). Output includes the token/cost line; the full case persists to
`data/reports/screener_<run>.md`. `qt agent report TICKER` writes a sober one-pager
(works keyless — the LLM judgment section is simply omitted). The agent **never**
gives price targets or buy/sell calls; mandates demanding them are refused.

## Known biases (read before backtesting)

- `fundamentals_snapshot` is current-constituents-only: survivorship-biased, never a
  PIT factor source (every row carries the warning in its `note` column).
- NASDAQ membership is point-in-time **forward from the first refresh**; earlier
  dates need Norgate/Sharadar-style history.
- yfinance news exposes only the recent stream: sentiment breadth is a function of
  when harvesting started; the IC sample effectively starts there.
- The AV firehose backfills ~1 day of history per calendar day on the free tier; the
  vendor's `score_av` is produced by its *current* model (not PIT) — which is why the
  frozen, revision-pinned FinBERT score (`sent_finbert`) is the primary factor.
- IC t-stats ignore the serial correlation induced by overlapping forward windows;
  with short history treat |IC| < ~0.02 or t < 2 as noise.

## Tests

```powershell
pytest            # offline suite, no network
pytest -m live    # optional live yfinance smoke test
ruff check src tests
```

## Design notes / roadmap

The architecture decisions (why unadjusted prices, why flag-never-mutate, why the
universe table is interval-based) and the forward roadmap (point-in-time fundamentals
via SEC EDGAR, survivorship-free membership via Norgate/Sharadar, intraday, Dagster
orchestration, data versioning, total-return series, Newey-West IC errors, open(D+1)
robustness checks) are documented in the module docstrings and the project plan.
