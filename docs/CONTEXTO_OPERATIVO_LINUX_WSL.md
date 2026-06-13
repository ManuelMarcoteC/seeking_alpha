# Contexto operativo — montaje completo en equipo Linux/WSL

Complemento de `docs/CONTEXTO_OPERATIVO_NASDAQ.md` (run de referencia del equipo de origen,
Windows/CPU, 2026-06-12). Este documento registra el **montaje del pipeline completo en una
máquina distinta** —WSL2 Ubuntu sobre Windows, Python 3.12.3— con sus cifras reales, sus
diferencias frente al origen y la **idiosincrasia de esta instalación**.

> Snapshot de este montaje: **2026-06-13**, máquina WSL2/Ubuntu, **sin clave de Alpha
> Vantage**, capa agente facturando contra **suscripción Claude Pro/Max vía OAuth**
> (no API key). Reproducible con el runbook de la sección 5.

---

## 0. Alcance: NASDAQ COMPLETO (precios + noticias + sentimiento)

A diferencia de una validación de subconjunto, esta instalación montó el **universo NASDAQ
completo**: ~3.327 acciones comunes, ~6,1 M filas OHLCV desde 2015, y el harvest completo de
noticias con scoring FinBERT sobre las ~90 k binds. La única pieza no ejecutada es
`fundamentals` (falta el CSV externo, ver sección 6). El montaje se hizo por etapas en
background; cada una idempotente/reanudable.

---

## 1. Compilación del universo NASDAQ

- **Fuente**: directorio NASDAQ Trader (`nasdaqlisted.txt`), descargado en vivo por
  `qt universe refresh`.
- **Resultado**: **5.506 símbolos → 3.327 acciones comunes** (`+3327 / -0`, snapshot
  inicial). Frente a las 3.325 del equipo de origen, **+2 tickers**: variación normal del
  directorio NASDAQ entre el 2026-06-12 y el 2026-06-13.
- **Point-in-time SOLO hacia delante**: `effective_from = 2026-06-13`, nota `INITIAL
  SNAPSHOT` en cada fila. Sin prehistoria — backtest previo a esa fecha = sesgo de
  supervivencia. Para historia real de constituyentes: Norgate/Sharadar.
- **Diferencia con el origen**: aquí `universe_membership` **NO retiene seed de SP500** (no
  se ejecutó `qt universe seed`): `universe NASDAQ distinct == 3.327`. En el origen
  coexistían NASDAQ + SP500 (3.674 distintos). Filtrar siempre por `index_name='NASDAQ'`.

```sql
SELECT ticker FROM universe_membership
WHERE index_name='NASDAQ' AND effective_from <= DATE '2026-06-13'
  AND (effective_to IS NULL OR effective_to > DATE '2026-06-13');
```

---

## 2. Precios (OHLCV) — universo NASDAQ completo

- **Proveedor**: `yfinance` (keyless), `qt ingest --universe NASDAQ`.
- **Rango**: 2015-01-02 → 2026-06-12.
- **Ingesta raw**: `ok=5054 empty=1577 skipped=23 failed=0`, **6.143.679 filas**. Los 1.577
  "empty" son tickers sin histórico en yfinance (recién listados, OTC, ilíquidos); failed=0.
  Dos errores transitorios de yfinance al final (XXII curl/nghttp2, un 401 "Invalid Crumb")
  son fallos de sesión del vendor en tickers sueltos, no del pipeline.
- **Curación** (`qt curate`): **6.115.259 filas** OHLCV promovidas, **16 en cuarentena**,
  **28.404 corporate actions**. Dos WARNING de ajuste (CBIO, SVA: dividendo ≥ cierre previo)
  → el pipeline **salta el factor** en vez de corromper la serie ("flag, never mutate").
- **Vistas DuckDB**: `ohlcv_daily` y `ohlcv_daily_adj` con **6.149.795 filas / 3.327
  tickers**; `corporate_actions` con 28.711 filas / 1.750 tickers.

---

## 3. Anomalías (`qt validate`) — 355.029 flags

Barrido sobre las ~6,1 M filas. **355.029 flags** en 3.154 tickers. NO son errores del
pipeline: son materia prima de research (el precio nunca se muta, solo se anota).

| flag_type | severidad | n |
|---|---|---|
| `zero_volume_run` | info | 154.094 |
| `stale_price` | info / warn | 101.194 / 45.930 |
| `return_outlier_mad` | warn | 31.922 |
| `unexplained_gap` | error | 21.872 |
| `missing_session` | warn | 17 |

El grueso (zero_volume + stale, ~300 k) es el perfil esperado de un universo con miles de
micro-caps ilíquidas. En el origen las cifras eran ~10× menores porque su desglose se
reportó sobre un perfil de cobertura distinto; aquí se documenta lo que **este** lago tiene.
Informe: `data/reports/validation_<run>.md`.

---

## 4. Noticias y sentimiento — universo NASDAQ completo

Cadena: harvest → curación → scoring FinBERT → factor.

- **Proveedor**: `yfinance_news` (keyless), `--universe NASDAQ`. **El firehose de Alpha
  Vantage NO se ejecutó** (sin clave; además el tier gratuito da ~25 req/día, inútil a este
  volumen): no hay `score_av` ni relevancia de vendor; el sentimiento es **íntegramente
  FinBERT**.
- **Volumen**:
  - Harvest: `ok=3057 empty=270`, **90.298 filas raw**.
  - Curación → `news_articles` **74.171 artículos** deduplicados (first-capture-wins) +
    `news_ticker_sentiment` **90.320 binds** ticker↔artículo (curate reportó 163.408 filas
    procesadas pre-dedup).
  - Scoring FinBERT (revisión pineada `4556d130…`): **90.320 / 90.320 titulares puntuados
    (100 %)**; media **+0,202**, rango **[−0,970, +0,949]**. ~44 min de CPU en bucle de
    `--limit 20000` hasta "scored 0".
  - Factor `sentiment_daily`: **3.057 filas (ticker, día)**.
- **Cobertura temporal de los artículos**: **2025 → 17.647**, **2026 → 56.524**. Es decir,
  yfinance solo expone el **stream reciente** por ticker (~50 ítems, sin archivo): la
  cobertura está **concentrada en lo reciente** y las pocas filas de 2025 son las noticias
  más antiguas que aún quedan en cada stream — **NO es un histórico completo de 2025**.
- **Atribución PIT**: `effective_ts = max(published_at, ingested_at)` y corte 15:30 ET. El
  harvest se ejecutó **viernes 2026-06-13**, así que el factor se atribuye a la **siguiente
  sesión, lunes 2026-06-15** — confirmado: `sentiment_daily` va todo a `2026-06-15`.
  Correcto: no se opera una noticia antes de poseerla.

**Sesgo crítico de cobertura**: la historia del factor **empieza el día del harvest**; no
hay sentimiento "hacia atrás". Re-ejecutar a diario hace que la serie se acumule.

---

## 5. Estado del lago + comparativa con el origen

| Tabla / vista | Este montaje (Linux/WSL) | Origen (referencia, Windows) |
|---|---|---|
| `universe_membership` (NASDAQ) | 3.327 distintos | 3.674 (incl. seed SP500) |
| `ohlcv_daily` | 6.149.795 | 6.133.675 |
| `corporate_actions` | 28.711 | 28.543 |
| `validation_flags` | 355.029 | 28.985 |
| `news_articles` | 74.171 | (de 164.232 binds) |
| `news_ticker_sentiment` (scored) | 90.320 (100 %) | 90.137 |
| `sentiment_daily` | 3.057 | 3.049 |
| `fundamentals_snapshot` | **vacío** (sin CSV) | 5.304 |

**Informe IC** (`qt research sentiment-ic`): con un solo día de factor, `n_días=0` / `nan`
en los tres horizontes — **esperado y caveateado**; gana sentido al acumular sesiones.

**Capa agente — validada en vivo contra la suscripción**: `qt agent screener` con
`QT_AGENT_USE_SUBSCRIPTION=true` resolvió un mandato de momentum sobre `ohlcv_daily_adj`:
5 llamadas reales a `api.anthropic.com` (todas HTTP 200, OAuth), 3 rondas SQL de solo
lectura, propuesta verificada, revisor PASS, verificación determinista OK, coste contable
~$0,30 facturado a la suscripción. El agente puso en los caveats la ausencia de tabla de
sectores y descartó artefacto de split vía `adj_factor=1.0` — anclaje real al dato.

---

## 6. Runbook que produjo este estado

```bash
source venv/bin/activate
qt init
qt universe refresh                                        # → 3.327 acciones NASDAQ (PIT)
qt ingest --universe NASDAQ                                # → 6,14 M filas raw (~50 min)
qt curate                                                  # → 6,12 M filas, 16 cuarentena
qt validate                                                # → 355.029 flags
qt news ingest --provider yfinance_news --universe NASDAQ  # → 90.298 filas raw (~50 min)
qt news curate                                             # → 90.320 binds
qt news score --limit 20000   # en bucle hasta "scored 0"  # → 90.320 FinBERT (~44 min)
qt news build-factor                                       # → 3.057 filas sentiment_daily
qt research sentiment-ic                                   # informe IC en data/reports/
QT_AGENT_USE_SUBSCRIPTION=true qt agent screener "..."     # agente vía suscripción
```

Recortes deliberados frente al runbook completo:
- **Sin Alpha Vantage** (sin clave; tier gratuito inútil a volumen): firehose omitido, solo
  FinBERT.
- **Sin fundamentals**: `Webinar_Agentes/screener_us.csv` (export de stockanalysis.com,
  gitignorado, no reproducible por código) no está en esta máquina; `fundamentals_snapshot`
  queda vacío. Todo lo demás funciona sin él.

### Nota técnica: bug de esquema parquet corregido durante el montaje

A escala completa, `news_ticker_sentiment` quedó con una partición cuyas columnas
`finbert_revision`/`scored_at` eran **todo-nulas** (tipo Arrow `null`) coexistiendo con otra
partición de tipos concretos (string / timestamp). `pd.read_parquet` lanzaba entonces
`ArrowNotImplementedError: Unsupported cast from string to null`, bloqueando `qt news score`.
Se corrigió en `src/qtdata/storage/parquet_store.read` (unificación de esquema entre
fragmentos con `pa.unify_schemas`, promoviendo `null`→tipo concreto y reconciliando precisión
de timestamps), con 3 tests de regresión. Es un fallo latente que solo aflora cuando coexiste
una partición sin scorear con otra ya scoreada — por eso no salió en el origen.

---

## 7. Diferencia clave de esta instalación: agente vía suscripción

La capa agente **factura contra una suscripción Claude Pro/Max** mediante el camino OAuth de
`src/qtdata/agents/claude_subscription.py`:

```bash
claude auth login --claudeai      # una vez: autentica la CLI oficial de Claude Code
# en .env:
QT_AGENT_USE_SUBSCRIPTION=true
```

Con eso `QT_ANTHROPIC_API_KEY` se ignora y el agente reutiliza el token OAuth de
`~/.claude/.credentials.json`. Sin esa variable, el comportamiento por defecto (API key) se
mantiene intacto. Detalles del bypass en el README, sección "Agent billing via Claude
subscription".

---

## 8. Implicaciones para agentes y consultas

- `sentiment_daily`/`sentiment_daily_decayed`: la historia **empieza el día del harvest**
  (2026-06-15); ausencia previa = falta de cobertura, no señal neutra.
- `score_av` **vacío** en este lago (no se usó AV): usar `score_finbert`.
- `fundamentals_snapshot` **vacío** aquí; cuando exista, es estático y sesgado por
  supervivencia — research/agente, nunca factor PIT.
- Universo NASDAQ **forward-PIT desde 2026-06-13**: no proyectar el roster actual al pasado.
- Cobertura de noticias **concentrada en lo reciente**: la etiqueta "2025" no implica
  histórico completo de ese año, solo el alcance del stream de cada ticker.
