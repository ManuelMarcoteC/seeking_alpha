# Replicar el proyecto en otro equipo

Resumen orientativo de cómo está montado `qtdata` en el equipo de origen y qué hace
falta para reconstruirlo en una máquina nueva que **clona el repo desde GitHub**.

La idea clave: **el repo solo lleva el código**. El entorno (venv), la configuración
(`.env`), el **lago de datos** (`data/`), la caché del modelo FinBERT y los activos del
webinar **NO viajan en git** (están en `.gitignore`) — se reconstruyen ejecutando el
pipeline. Esta guía dice exactamente qué falta y cómo regenerarlo.

---

## 1. Qué obtienes al clonar (y qué no)

| En el repo (lo clonas) | Fuera del repo (lo reconstruyes) | Tamaño aprox. aquí |
|---|---|---|
| `src/qtdata/` (paquete + CLI `qt`) | `venv/` (entorno virtual, específico de SO) | 1,2 GB |
| `tests/`, `.github/workflows/ci.yml` | `data/` (lago raw → curated → DuckDB) | ~390 MB |
| `README.md`, `pyproject.toml` | `.env` (config real; copiar de `.env.example`) | <1 KB |
| `requirements*.txt`, `.env.example` | Caché FinBERT (`~/.cache/huggingface`) | ~440 MB (solo FinBERT) |
| `.gitignore` | `Webinar_Agentes/screener_us.csv` (activo externo) | ~12 MB |

> El `venv/` es específico del sistema operativo (aquí Windows, con `torch==2.12.0+cpu`):
> **no se puede copiar** a otra máquina/SO — hay que recrearlo. La CI demuestra que la
> suite offline también corre en Linux (ubuntu-latest).

---

## 2. Requisitos de la máquina

- **Python 3.12** exclusivamente (`pyproject.toml` exige `>=3.12,<3.13`; aquí es 3.12.10).
  Si hay varias versiones instaladas, invocar explícitamente `py -3.12` (Windows) o
  `python3.12` (Linux/Mac). NO usar 3.13+/3.15 (no hay wheels de torch y el build rechaza).
- **git** + acceso al repo privado (`gh auth login` o clave SSH autorizada en GitHub).
- ~3 GB libres de disco para venv + lago + modelo.

---

## 3. Montaje del entorno

```powershell
# Windows / PowerShell
git clone https://github.com/ManuelMarcoteC/quantitative-trading.git
cd quantitative-trading
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
pip install -e . --no-deps                 # registra el comando `qt`

# Extras OPCIONALES (instalar solo si se van a usar):
pip install -r requirements-sentiment.txt  # FinBERT: torch CPU + transformers (~2 GB)
pip install -r requirements-agents.txt      # capa agente: SDK anthropic
```

En Linux/Mac es idéntico cambiando la activación del venv
(`source venv/bin/activate`). La CI (`.github/workflows/ci.yml`) usa exactamente estos
pasos sobre ubuntu-latest, así que sirve de referencia ejecutable.

**Capas de dependencias** (por qué están separadas):
- `requirements.txt` — núcleo (pandas, numpy, pyarrow, duckdb, yfinance, pydantic,
  pandera, exchange-calendars, typer, rich, requests, python-dotenv, tenacity).
- `requirements-dev.txt` — pytest, pytest-cov, ruff, freezegun.
- `requirements-sentiment.txt` — torch CPU + transformers. **Solo** para `qt news score`.
  Los imports son perezosos: sin estos extras todo lo demás funciona y los tests pasan.
- `requirements-agents.txt` — `anthropic`. **Solo** para `qt agent`.

---

## 4. Configuración (`.env`)

```powershell
copy .env.example .env
```

Todas las variables usan prefijo `QT_`. Claves relevantes:

| Variable | Para qué | ¿Obligatoria? |
|---|---|---|
| `QT_DATA_DIR=data` | Raíz del lago | no (default `data`) |
| `QT_DEFAULT_PROVIDER=yfinance` | Provider de ingesta OHLCV | no |
| `QT_DEFAULT_CALENDAR=XNYS` | Calendario de sesiones | no |
| `QT_DEFAULT_UNIVERSE=NASDAQ` | Universo por defecto de `qt update` / `qt news update` | no |
| `QT_ALPHA_VANTAGE_API_KEY=` | Firehose de noticias AV (free ≈ 25 req/día) | opcional — sin ella el firehose se omite |
| `QT_ANTHROPIC_API_KEY=` | Capa agente (Opus 4.8). Cae a `ANTHROPIC_API_KEY` | opcional — solo para `qt agent` |
| `QT_AGENT_MODEL=claude-opus-4-8` | Modelo del agente | no |
| `QT_FINBERT_REVISION=4556d130…` | Revisión PINEADA de FinBERT (reproducibilidad) | no — no cambiar sin motivo |

> El `.env` está gitignorado: **nunca** se sube. La clave de Anthropic se introduce a
> mano en cada equipo.

---

## 5. Activos externos que el código NO puede regenerar

1. **`Webinar_Agentes/screener_us.csv`** — export del screener de stockanalysis.com
   (~5.300 tickers × 308 columnas, snapshot estático). Lo necesita
   `qt fundamentals ingest`. Está gitignorado y hay que obtenerlo aparte (exportarlo de
   stockanalysis.com). Sin él, todo el pipeline de precios/noticias/factor funciona; solo
   se queda sin la tabla `fundamentals_snapshot` (que usan el agente y `qt agent report`).
2. **Clave de Anthropic** — para `qt agent`. Sin ella, `qt agent screener` sale con
   error controlado; `qt agent report` funciona pero omite la sección redactada por el LLM.

---

## 6. Reconstruir el lago de datos (runbook)

`data/` no está en git: se regenera ejecutando el pipeline. Orden y tiempos medidos en el
equipo de origen (Windows, CPU, red doméstica):

```powershell
qt init                                                   # crea data/ + esquema DuckDB (seg)
qt universe refresh                                       # directorio NASDAQ → ~3.300 acciones PIT (seg)
qt ingest --universe NASDAQ                               # OHLCV desde 2015 → ~6,1 M filas (~35 min)
qt curate                                                 # valida + promueve (~7 min)
qt validate                                               # barrido de anomalías (opcional)

qt news ingest --provider yfinance_news --universe NASDAQ # harvest de titulares (~50 min)
qt news curate
qt news score --limit 20000                               # FinBERT; repetir hasta "scored 0" (~60 min, 1ª vez baja el modelo)
qt news build-factor                                      # factor sentiment_daily

qt fundamentals ingest Webinar_Agentes/screener_us.csv    # requiere el CSV externo (sección 5)
qt research sentiment-ic                                  # informe IC en data/reports/
qt agent screener "..."                                   # requiere clave Anthropic
```

Tiempo total ≈ **1,5–2,5 h**. Todas las etapas son **idempotentes / reanudables**
(watermarks + ledger de ficheros curados): si se corta, se relanza el mismo comando.

**Importante — DuckDB es single-writer**: no ejecutar dos comandos `qt` de escritura a la
vez sobre el mismo lago.

Bucle diario una vez montado: `qt universe refresh` + `qt update` + `qt news update`.

Lo que produce un run completo (escala de referencia): ~6,13 M filas OHLCV, ~3.300
acciones NASDAQ comunes (la tabla de membresía además retiene un seed previo de SP500),
90.137 titulares puntuados, 3.049 filas de factor, 5.304 fundamentales.

---

## 7. Arquitectura en una pantalla (cómo está "montado")

- **Medallion**: `data/raw/` (payloads de vendor inmutables, append-only) → `data/curated/`
  (parquet canónico validado, hive-particionado por año) → vistas DuckDB en
  `data/catalog.duckdb`. Precios sin ajustar como verdad; el ajuste se deriva al leer
  (`ohlcv_daily_adj`). "Flag, never mutate": las anomalías se registran en
  `validation_flags`, los precios no se tocan.
- **Providers**: `yfinance` (keyless), `alpha_vantage` (con clave), `synthetic`
  (determinista, para tests/demo), `alpha_vantage_news`, `yfinance_news`.
- **CLI** (`qt`, definido en `pyproject [project.scripts]` → `qtdata.cli:app`), sub-apps:
  `universe`, `news`, `fundamentals`, `agent`, `research`, más
  `init/ingest/curate/validate/reconcile/query/status/update`.
- **Capa agente** (`src/qtdata/agents/`): patrón del webinar con SDK Anthropic puro —
  mandato → bucle de SQL **solo lectura** → propuesta validada (Pydantic) → revisor
  independiente → 1 corrección quirúrgica → **verificación determinista** → coste. Se
  niega a predecir precios por regla.
- **Research** (`src/qtdata/research/`): validación del factor — IC de Spearman vs
  retornos forward, curva de decaimiento, event study; informe en `data/reports/`.

---

## 8. Verificación de que la réplica funciona

```powershell
pytest                 # suite offline (sin red, sin torch/anthropic); debe quedar en verde
ruff check src tests   # lint limpio
qt query "SELECT COUNT(*) FROM ohlcv_daily"   # comprueba que el lago responde
```

La CI repite `ruff` + `pytest` en cada push/PR automáticamente, así que el equipo nuevo
hereda la red de seguridad sin configurar nada.
