# Contexto operativo — universo NASDAQ y binds de noticias (nuestro montaje)

Este documento captura la **idiosincrasia de NUESTRO proyecto frente al repositorio
público**: el repo solo lleva código; el lago de datos (`data/`) está gitignorado, así
que **la compilación concreta de tickers del NASDAQ y los binds de noticias que
ejecutamos no quedan registrados en GitHub**. Aquí se documentan, con sus cifras, su
procedencia y sus sesgos, para el trabajo cotidiano y para que cualquier agente que
consulte el lago sepa qué está mirando.

> Snapshot del run de referencia: **2026-06-12** (sin clave de Alpha Vantage, máquina
> Windows/CPU). Reproducible con el runbook de la sección 4.

---

## 1. Compilación del universo NASDAQ

- **Fuente**: directorio de símbolos de NASDAQ Trader (`nasdaqlisted.txt`), descargado en
  vivo por `qt universe refresh` (`src/qtdata/nasdaq_directory.py`).
- **Filtrado a acción común**: se excluyen ETFs, test issues, derivados por nombre
  (warrant/right/unit/preferred/notes/debenture/bond), estatus financiero malo y símbolos
  que no casan `^[A-Z]{1,5}$`.
- **Resultado del run**: **5.506 símbolos → 3.325 acciones comunes** (`+3325 / -0`,
  snapshot inicial).
- **Point-in-time SOLO hacia delante**: la membresía (`universe_membership`) abre
  intervalos a partir de `effective_from = 2026-06-12`. **No hay prehistoria**: cualquier
  backtest sobre este universo antes de esa fecha está sesgado por supervivencia. Para
  historia real de constituyentes haría falta Norgate/Sharadar. Cada fila de la primera
  carga lleva la nota `INITIAL SNAPSHOT`.
- **Coexistencia con SP500**: la tabla `universe_membership` retiene además un seed previo
  de SP500 (de ahí que el total de tickers distintos —3.674— supere las 3.325 del NASDAQ).
  Los índices conviven por `index_name`; filtra por `index_name='NASDAQ'`.

**Cómo consultarlo:**
```sql
-- miembros NASDAQ vigentes a una fecha
SELECT ticker FROM universe_membership
WHERE index_name='NASDAQ' AND effective_from <= DATE '2026-06-12'
  AND (effective_to IS NULL OR effective_to > DATE '2026-06-12');
```

---

## 2. Binds de noticias (ticker ↔ artículo ↔ sentimiento)

El "bind" es el enlace entre cada artículo y los tickers que menciona, y de ahí al factor
diario. Cadena: harvest → curación (artículos + binds ticker) → scoring FinBERT → factor.

- **Proveedor usado**: `yfinance_news` (keyless), harvest por ticker sobre el universo
  NASDAQ. **El firehose de Alpha Vantage NO se ejecutó** (sin `QT_ALPHA_VANTAGE_API_KEY`):
  por tanto **no hay `score_av` ni relevancia de vendor**; el sentimiento proviene
  íntegramente de FinBERT.
- **Volumen del run**:
  - Harvest: `ok=3049` tickers con noticias, `empty=276`; **90.144 filas raw**.
  - Curación → `news_articles` (artículos deduplicados, first-capture-wins) +
    `news_ticker_sentiment` (el bind ticker↔artículo): **164.232 filas curadas**.
  - Scoring FinBERT (revisión pineada `4556d130…`): **90.137 titulares puntuados**;
    media **+0,205**, rango [−0,967, +0,94].
  - Factor `sentiment_daily`: **3.049 filas (ticker, día)**.
- **Atribución temporal PIT**: `effective_ts = max(published_at, ingested_at)` y corte
  15:30 ET → como el harvest se hizo hoy, casi todo el sentimiento se atribuye a la
  **siguiente sesión (lunes 2026-06-15)**. Esto es correcto: no se puede operar una
  noticia antes de poseerla.

**Sesgo crítico de cobertura**: yfinance solo expone el **stream reciente** por ticker
(~50 ítems, sin archivo). Por eso la historia del factor **empieza efectivamente en la
fecha del harvest**: no hay sentimiento "hacia atrás". Re-ejecutar el harvest a diario
hace que la serie se acumule.

---

## 3. Estado del lago tras el run (qué hay ahora mismo)

| Tabla / vista | Contenido | Filas |
|---|---|---|
| `universe_membership` | Membresía PIT (NASDAQ + seed SP500) | 3.674 tickers distintos |
| `ohlcv_daily` | Precios diarios sin ajustar (desde 2015) | 6.133.675 |
| `ohlcv_daily_adj` | Ajustados al leer (split/dividendo) | (derivada) |
| `corporate_actions` | Splits y dividendos | 28.543 |
| `validation_flags` | Anomalías (no mutan el precio) | 28.985 |
| `news_articles` | Artículos deduplicados | (de 164.232 binds) |
| `news_ticker_sentiment` | Bind ticker↔artículo + scores | 90.137 |
| `sentiment_daily` | Factor diario de sentimiento | 3.049 |
| `fundamentals_snapshot` | Snapshot estático stockanalysis.com | 5.304 |

**Flags de validación (desglose)**: `stale_price` (info 8.645 / warn 4.715),
`zero_volume_run` 8.481, `unexplained_gap` (error 4.088), `return_outlier_mad` (warn
3.057), `missing_session` 1. Son materia prima de investigación (micro-caps, halts), no
errores del pipeline.

**Informe IC** (`qt research sentiment-ic`): con un solo día de factor, `n_days=0` en todos
los horizontes — **esperado y caveateado**. El IC gana sentido según se acumulen sesiones.

---

## 4. Runbook que produjo este estado

```powershell
qt init
qt universe refresh                                        # → 3.325 acciones comunes NASDAQ (PIT)
qt ingest --universe NASDAQ                                # → 6,1 M filas OHLCV (~35 min)
qt curate                                                  # → 6.132.201 filas, 17 cuarentena, 28.985 flags
qt news ingest --provider yfinance_news --universe NASDAQ  # → 90.144 filas raw (~50 min)
qt news curate                                             # → 164.232 binds curados
qt news score --limit 20000   (×5, hasta "scored 0")       # → 90.137 titulares FinBERT (~60 min)
qt news build-factor                                       # → 3.049 filas sentiment_daily
qt fundamentals ingest Webinar_Agentes/screener_us.csv     # → 5.304 fundamentales
qt research sentiment-ic                                   # → informe en data/reports/
```

Sin firehose AV (no había clave) y sin `qt agent screener` (pendiente de clave Anthropic).

---

## 5. Implicaciones para agentes y consultas

- Al leer `sentiment_daily`/`sentiment_daily_decayed`: la **historia empieza el día del
  harvest**; tratar la ausencia de sentimiento previo como falta de cobertura, no como
  señal neutra.
- `sent_av` está **vacío** en este lago (no se usó AV): usar `sent_finbert`.
- `fundamentals_snapshot` es **estático y sesgado por supervivencia** — research/agente,
  nunca fuente de factor PIT (cada fila lo dice en `note`).
- El universo NASDAQ es **forward-PIT desde 2026-06-12**: no proyectar el roster actual
  hacia el pasado.
