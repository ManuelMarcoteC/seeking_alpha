# Contexto operativo — validación end-to-end en equipo Linux/WSL

Complemento de `docs/CONTEXTO_OPERATIVO_NASDAQ.md` (que recoge el run de referencia del
equipo de origen, Windows/CPU, 2026-06-12). Este documento registra la **réplica del
pipeline en una máquina distinta** —WSL2 Ubuntu sobre Windows, Python 3.12.3— y, sobre
todo, deja por escrito la **idiosincrasia de esta instalación**: qué se ejecutó, con qué
cifras reales y con qué recortes deliberados respecto al montaje completo.

> Snapshot de esta validación: **2026-06-13**, máquina WSL2/Ubuntu, **sin clave de Alpha
> Vantage**, capa agente facturando contra **suscripción Claude Pro/Max vía OAuth**
> (no API key). Reproducible con el runbook de la sección 4.

---

## 0. Por qué este run es un SUBCONJUNTO (no el NASDAQ completo)

El propósito de esta pasada fue **verificar que toda la cadena encadena en Linux/WSL** y
que la capa agente enruta contra la suscripción, antes de comprometer la máquina ~2 h con
el universo completo. Por eso se ingirió un **subconjunto de 12 acciones líderes**
(AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA, AVGO, COST, PEP, AMD, NFLX) en lugar de las
~3.327 del NASDAQ. El universo PIT sí se compiló entero; lo acotado es la ingesta de
precios y noticias. El runbook del NASDAQ completo queda en la sección 5, listo para
relanzar.

---

## 1. Compilación del universo NASDAQ (completa)

- **Fuente**: directorio NASDAQ Trader (`nasdaqlisted.txt`), descargado en vivo por
  `qt universe refresh`.
- **Resultado del run**: **5.506 símbolos → 3.327 acciones comunes** (`+3327 / -0`,
  snapshot inicial). Frente a las 3.325 del equipo de origen, **+2 tickers**: variación
  normal del directorio NASDAQ entre el 2026-06-12 y el 2026-06-13.
- **Point-in-time SOLO hacia delante**: `effective_from = 2026-06-13`; nota
  `INITIAL SNAPSHOT` en cada fila. Sin prehistoria — backtest previo a esa fecha =
  sesgo de supervivencia. Para historia real: Norgate/Sharadar.
- **Diferencia con el origen**: aquí la tabla `universe_membership` **NO retiene seed de
  SP500** (no se ejecutó `qt universe seed`): `universe ALL distinct == NASDAQ distinct ==
  3.327`. En el origen coexistían NASDAQ + SP500 (3.674 distintos).

```sql
SELECT ticker FROM universe_membership
WHERE index_name='NASDAQ' AND effective_from <= DATE '2026-06-13'
  AND (effective_to IS NULL OR effective_to > DATE '2026-06-13');
```

---

## 2. Precios (OHLCV) — subconjunto de 12 tickers

- **Proveedor**: `yfinance` (keyless), `--tickers` con las 12 acciones líderes.
- **Rango**: 2015-01-02 → 2026-06-12 (mismo arranque por defecto que el origen).
- **Volumen del run**:
  - Ingesta raw: `ok=23 empty=1 failed=0`, **34.843 filas** (OHLCV + actions), ~3 s.
  - Curación → **34.536 filas** OHLCV + **307 corporate actions**, **0 cuarentena**.
- **Validación (`qt validate`)**: **57 flags** — `return_outlier_mad` (warn) 55,
  `unexplained_gap` (error) 2. Materia prima de research (días de crash/halt en
  large-caps), no errores del pipeline. El precio no se muta: se marca.

---

## 3. Noticias y sentimiento — subconjunto de 12 tickers

Cadena: harvest → curación (artículos + binds ticker) → scoring FinBERT → factor.

- **Proveedor**: `yfinance_news` (keyless), harvest por ticker sobre las 12 acciones.
  **El firehose de Alpha Vantage NO se ejecutó** (sin clave): no hay `score_av` ni
  relevancia de vendor; el sentimiento proviene **íntegramente de FinBERT**.
- **Volumen del run**:
  - Harvest: `ok=12 empty=0`, **600 filas raw**.
  - Curación → `news_ticker_sentiment` (bind ticker↔artículo): **1.083 binds curados**.
  - Scoring FinBERT (revisión pineada `4556d130…`): **600 titulares puntuados**;
    media **+0,161**, rango **[−0,968, +0,94]** (~1 min, incluyendo descarga del modelo).
  - Factor `sentiment_daily`: **12 filas (ticker, día)**.
- **Atribución temporal PIT**: `effective_ts = max(published_at, ingested_at)` y corte
  15:30 ET. El harvest se ejecutó en **viernes 2026-06-13**, así que el factor se atribuye
  a la **siguiente sesión, lunes 2026-06-15** — confirmado en el lago: `sentiment_daily`
  va todo a `2026-06-15`. Correcto: no se opera una noticia antes de poseerla.

**Sesgo crítico de cobertura** (idéntico al origen): yfinance solo expone el stream
reciente por ticker; la historia del factor **empieza el día del harvest**. Re-ejecutar a
diario hace que la serie se acumule.

---

## 4. Estado del lago tras esta validación

| Tabla / vista | Contenido | Filas (este run) | Origen (referencia) |
|---|---|---|---|
| `universe_membership` | Membresía PIT NASDAQ | 3.327 distintos | 3.674 (incl. seed SP500) |
| `ohlcv_daily` | Precios sin ajustar (desde 2015) | 34.536 | 6.133.675 |
| `ohlcv_daily_adj` | Ajustados al leer | (derivada) | (derivada) |
| `corporate_actions` | Splits y dividendos | 307 | 28.543 |
| `validation_flags` | Anomalías (no mutan precio) | 57 | 28.985 |
| `news_ticker_sentiment` | Bind ticker↔artículo + scores | 600 puntuados | 90.137 |
| `sentiment_daily` | Factor diario | 12 | 3.049 |
| `fundamentals_snapshot` | Snapshot estático | **vacío** (sin CSV) | 5.304 |

**Informe IC** (`qt research sentiment-ic`): `n_días=0` / `nan` en los tres horizontes
(1/5/20) — **esperado y caveateado**: con un solo día de factor no hay retornos forward
que correlacionar. Igual que en el origen; gana sentido al acumular sesiones.

**Capa agente — validada en vivo contra la suscripción**: `qt agent screener` con
`QT_AGENT_USE_SUBSCRIPTION=true` resolvió un mandato de momentum a 6 meses sobre
`ohlcv_daily_adj`: 5 llamadas reales a `api.anthropic.com` (todas HTTP 200, OAuth),
3 rondas SQL de solo lectura, propuesta verificada (AMD +142,7%, NVDA +17,4%, GOOGL
+16,4%), revisor PASS, verificación determinista OK, coste contable ~$0,30 facturado a la
suscripción. El agente puso en los caveats la ausencia de tabla de sectores y descartó
artefacto de split en AMD vía `adj_factor=1.0` — anclaje real al dato, no narrativa.

---

## 5. Runbook del NASDAQ COMPLETO (pendiente de lanzar en este equipo)

Este es el montaje completo, a relanzar cuando la máquina esté libre (~1,5–2,5 h, casi
todo red + CPU). Todas las etapas son idempotentes/reanudables; DuckDB es single-writer
(no lanzar dos comandos de escritura a la vez).

```bash
source venv/bin/activate
qt init
qt universe refresh                                        # → ~3.327 acciones NASDAQ (PIT)
qt ingest --universe NASDAQ                                # → ~6,1 M filas OHLCV (~35 min)
qt curate                                                  # valida + promueve (~7 min)
qt validate                                                # barrido de anomalías (opcional)
qt news ingest --provider yfinance_news --universe NASDAQ  # harvest (~50 min)
qt news curate
qt news score --limit 20000   # repetir hasta "scored 0"   # FinBERT (~60 min)
qt news build-factor                                       # factor sentiment_daily
# qt fundamentals ingest Webinar_Agentes/screener_us.csv   # SOLO si se dispone del CSV
qt research sentiment-ic                                   # informe IC en data/reports/
QT_AGENT_USE_SUBSCRIPTION=true qt agent screener "..."     # agente vía suscripción
```

Recortes deliberados de esta instalación frente al runbook completo:
- **Subconjunto de 12 tickers** en lugar de `--universe NASDAQ` (validación, no producción).
- **Sin Alpha Vantage** (sin clave): firehose de noticias omitido; solo FinBERT.
- **Sin fundamentals**: `Webinar_Agentes/screener_us.csv` (export de stockanalysis.com,
  gitignorado, no reproducible por código) no está en esta máquina; `fundamentals_snapshot`
  queda vacío. Todo lo demás funciona sin él.

---

## 6. Diferencia clave de esta instalación: agente vía suscripción

A diferencia del equipo de origen (que dejó `qt agent` pendiente de clave Anthropic), aquí
la capa agente **factura contra una suscripción Claude Pro/Max** mediante el camino OAuth
añadido en `src/qtdata/agents/claude_subscription.py`:

```bash
claude auth login --claudeai      # una vez: autentica la CLI oficial de Claude Code
# en .env:
QT_AGENT_USE_SUBSCRIPTION=true
```

Con eso `QT_ANTHROPIC_API_KEY` se ignora y el agente reutiliza el token OAuth de
`~/.claude/.credentials.json`. Ver la sección "Agent billing via Claude subscription" del
README para los detalles del bypass. Sin esa variable, el comportamiento por defecto
(API key) se mantiene intacto.

---

## 7. Implicaciones para agentes y consultas (idénticas al origen)

- `sentiment_daily`/`sentiment_daily_decayed`: la historia **empieza el día del harvest**;
  ausencia previa = falta de cobertura, no señal neutra.
- `score_av` **vacío** en este lago (no se usó AV): usar `score_finbert`.
- `fundamentals_snapshot` **vacío** aquí; cuando exista, es estático y sesgado por
  supervivencia — research/agente, nunca factor PIT.
- Universo NASDAQ **forward-PIT desde 2026-06-13**: no proyectar el roster actual al pasado.
