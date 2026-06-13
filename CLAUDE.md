# CLAUDE.md — guía para agentes en `qtdata`

Contexto persistente para Claude / agentes que trabajen en este repositorio. Léelo antes
de actuar. Para replicar el proyecto en otra máquina, ver `docs/REPLICAR_EN_OTRO_EQUIPO.md`;
para el estado real del lago que montamos nosotros, `docs/CONTEXTO_OPERATIVO_NASDAQ.md`.

## Qué es

Pipeline de datos cuantitativos con disciplina institucional. Paquete Python `qtdata`,
CLI `qt`. Tres frentes vivos: (1) factor de **sentimiento** de noticias con FinBERT,
(2) universo **NASDAQ** point-in-time, (3) capa **agente** (Claude Opus 4.8) que explora
los datos vía SQL de solo lectura.

## Comandos

```powershell
# entorno (Python 3.12 EXCLUSIVAMENTE; pyproject exige >=3.12,<3.13)
py -3.12 -m venv venv; .\venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt; pip install -e . --no-deps
# extras OPCIONALES (imports perezosos; la suite pasa sin ellos):
pip install -r requirements-sentiment.txt   # torch CPU + transformers (qt news score)
pip install -r requirements-agents.txt       # anthropic (qt agent)

# calidad (lo que corre la CI en cada push/PR)
pytest                 # suite offline, sin red (marker 'live' deselectado por defecto)
ruff check src tests

# pipeline (ver runbook completo en docs/CONTEXTO_OPERATIVO_NASDAQ.md)
qt init | qt universe refresh | qt ingest --universe NASDAQ | qt curate
qt news ingest --provider yfinance_news --universe NASDAQ | qt news curate
qt news score --limit 20000 | qt news build-factor
qt fundamentals ingest <csv> | qt research sentiment-ic | qt agent screener "..."
qt query "SELECT ... FROM ohlcv_daily_adj"   # SQL ad-hoc sobre las vistas DuckDB
```

## Arquitectura

- **Medallion**: `data/raw/` (payloads de vendor inmutables, append-only) → `data/curated/`
  (parquet validado, hive-particionado por `year`) → vistas DuckDB en `data/catalog.duckdb`.
- **Mapa de `src/qtdata/`**: `cli.py` (typer, sub-apps universe/news/fundamentals/agent/
  research) · `config.py` (pydantic-settings, prefijo `QT_`) · `ingestion/` · `curation/`
  · `validation/` · `providers/` (yfinance, alpha_vantage, synthetic, *_news) ·
  `storage/` (parquet_store, catalog DuckDB) · `nasdaq_directory.py` · `news/`
  (ingest/curate/scoring FinBERT/aggregate) · `fundamentals.py` · `agents/`
  (llm/sql_tool/schema_context/screener/reviewer/verification/report) · `research/`
  (returns/ic/event_study/report/sentiment_validation).

## Reglas duras (invariantes — no las rompas)

1. **DuckDB es single-writer**: nunca ejecutes dos comandos `qt` de escritura a la vez
   sobre el mismo lago. `qt agent` abre el catálogo en **solo lectura**.
2. **Point-in-time, sin look-ahead**: nada de `bfill` (hay un test que lo prohíbe); las
   estadísticas de outliers usan ventanas trailing; la membresía del universo es por
   intervalos (`members_as_of(date)`). El factor de noticias usa
   `effective_ts = max(published_at, ingested_at)` y corte 15:30 ET → siguiente sesión.
3. **Flag, never mutate**: las anomalías se registran en `validation_flags`; los precios
   se quedan como los mandó el vendor. Precios sin ajustar = verdad; el ajuste se deriva
   al leer (`ohlcv_daily_adj`).
4. **Noticias: first-capture-wins** (lo contrario de OHLCV, que es latest-wins) — la
   primera observación es el hecho point-in-time.
5. **FinBERT con revisión PINEADA** (`QT_FINBERT_REVISION`): scorer congelado =
   factor reproducible. No actualizar el modelo sin una razón explícita.
6. **El agente NUNCA predice precios** ni da recomendaciones de compra/venta: las niega
   por regla de sistema. Toda propuesta pasa por verificación determinista
   (`agents/verification.py`), no por la narrativa del modelo.
7. **Imports perezosos**: torch/transformers/anthropic se importan dentro de funciones,
   nunca a nivel de módulo — mantiene `qt --help` y la suite offline ligeros.

## Convenciones de código y tests

- Estilo: ruff (`E,F,I,UP,B`, line-length 100). Type hints, `from __future__ import
  annotations`. Imports perezosos para dependencias pesadas/opcionales.
- Tests: offline por defecto, fixtures en `tests/conftest.py` (`settings`, `catalog`,
  `make_ohlcv`); CLI con `CliRunner` + `isolated_settings`; LLM con `tests/fake_anthropic.py`
  (jamás red). Los tests `@pytest.mark.live` quedan deselectados salvo `pytest -m live`.
- Commits: por workstream, cada uno dejando la suite en verde. El secreto del webinar
  (`Webinar_Agentes/`, con `.env` y CSV) está gitignorado — nunca trackear secretos.

## Gotchas

- `Webinar_Agentes/screener_us.csv` (lo que come `qt fundamentals ingest`) NO está en el
  repo ni es reproducible por código: hay que exportarlo de stockanalysis.com.
- Sin `QT_ALPHA_VANTAGE_API_KEY` el firehose de noticias se omite (degrada con aviso).
- Sin clave Anthropic, `qt agent screener` sale con error controlado; `qt agent report`
  funciona pero omite la sección redactada por el LLM.
- El factor de sentimiento solo tiene historia desde que arrancó el harvest de yfinance
  (stream reciente, sin archivo): el IC sale con `n_days=0` hasta que acumule sesiones.
