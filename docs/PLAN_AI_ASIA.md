# Plan AI-Asia — sentimiento asiático (1) + cesta temática (2)

> Plan de arquitectura para análisis de acciones asiáticas (SEHK/KRX) con foco en
> el tema "IA China/Asia". Producido por un loop de autointerrogación (skill
> grilling) sobre el plan inicial, afinado contra recon EN VIVO del pipeline.
> NO ejecutado: es diseño revisable. El paso 3 (motor intradía 5-min) queda FUERA,
> a comentar aparte.
>
> Principio rector (igual que con TimesFM): workstream SEPARADO y AISLADO que NO
> contamina el core NASDAQ PIT. Honra los invariantes (flag-never-mutate,
> no-look-ahead, single-writer, invariante 6 = nunca señal compra/venta).

## 0. Hallazgos del recon que CAMBIARON el plan inicial
Interrogar antes de construir tumbó dos premisas mías:

- **FALSO que necesitemos un scorer multilingüe primero.** yfinance_news SÍ devuelve
  noticias para 2513.HK / 005930.KS / 000660.KS, en INGLÉS, relevantes y con ~3,5
  meses de histórico (Bloomberg, Reuters, Fortune, Quartz, Simply Wall St.).
  FinBERT-inglés las puntúa con señal coherente — verificado en vivo:
  "Zhipu accelerates pivot to domestic chips amid AI boom" = +0,889;
  "SK Hynix Major Warning for Micron" = −0,499; "Anthropic Ban Forces Rethink" = −0,476.
- **El provider de news YA acepta .HK/.KS** sin cambios (llama a `yf.Ticker(t).get_news()`).

→ Conclusión: el verdadero gap de sentimiento NO es el idioma para estos nombres
globalmente notables. Es (a) la ATRIBUCIÓN temporal (corte 15:30 ET + calendario
XNYS hardcodeados en news/aggregate.py → una noticia de Zhipu a las 10:00 HKT se
bucketea contra sesiones de Nueva York, día de trading mal asignado) y (b) la
COBERTURA fina (solo ~10 ítems recientes/ticker, sin archivo). El multilingüe pasa
a ser refinamiento de Fase 2 (small-caps SEHK con solo prensa local en chino).

---

## 1. Árbol de diseño resuelto (autointerrogación)

**Q1 — ¿Lago separado o el mismo para las asiáticas?**
R: MISMO lago para precios (ya están vía Tier A; OHLCV es [ticker,date], los sufijos
.HK/.KS conviven sin problema), pero SEPARAR la pertenencia con su propio
`index_name` ("AIASIA") para NO contaminar el roster PIT del NASDAQ. El análisis es
un módulo READ-ONLY aparte. Encaja con la guía Tier B del skill qtdata-pipeline.

**Q2 — ¿Arreglar el calendario global XNYS ahora?**
R: En la capa de ANÁLISIS, resolver calendario por-ticker explícito (XHKG para .HK,
XKRX para .KS/.KQ) — barato y aislado. El desfase del watermark de ingest se mitiga
como ya hicimos: `qt ingest --tickers ... --end YYYY-MM-DD`. El calendario per-market
en config (ingest end-resolution) es Tier B propio; se aplaza salvo que la cesta se
vuelva un frente serio.

**Q3 — El bug real de atribución de sentimiento (corte 15:30 ET + XNYS).**
R: Parametrizar `news/aggregate.trading_day_for()` con `tz` y `calendar` por mercado.
CAMBIO DELICADO porque toca la cadena que usa NASDAQ → debe ser
RETROCOMPATIBLE: defaults `NY_TZ`/`XNYS` intactos, el comportamiento NASDAQ NO
cambia; las asiáticas pasan su tz/calendario propios. Con tests de regresión que
fijen que NASDAQ sigue idéntico.

**Q4 — ¿Scorer multilingüe?**
R: Fase 2, no Fase 1. Fase 1 reutiliza FinBERT sobre noticias inglesas (validado).
Documentar el límite: small-caps SEHK con SOLO prensa local en chino quedan
infra-cubiertas. Cuando llegue, opciones: FinBERT multilingüe o scorer vía la propia
capa Claude (que ya tenemos integrada).

**Q5 — Poco recorrido (2513 = 107 sesiones, Gap D).**
R: "modo histórico-corto": calcula SOLO las métricas definidas (SMA20/50, RSI(14),
MACD, vol corta, fuerza relativa) y devuelve N/A EXPLÍCITO (no basura) para ventana
larga (SMA200, ret 12M, rango 52-sem, max-DD histórico). Guarda dura: NUNCA computar
SMA200 sobre <200 barras y presentarlo como válido.

**Q6 — ¿Quién decide la composición de la cesta?**
R: Lista semilla CURADA por el usuario (como el seed SP500), forward-PIT desde su
fecha de semilla, `index_name="AIASIA"`. Semilla inicial propuesta: 2513.HK (Zhipu),
0100.HK (MiniMax), 005930.KS (Samsung), 000660.KS (SK Hynix), 035720.KS (Kakao),
247540.KQ (Ecopro) + los AI/semis del SEHK que añadas. Caveat de sesgo de
supervivencia documentado.

NOTA sobre niveles de histórico en la semilla (verificado en vivo):
- 0100.HK MiniMax Group (IPO ~18-jun-2026): ~1-2 sesiones. NINGÚN indicador técnico
  calculable (SMA/RSI/MACD/vol/fuerza-relativa todos N/A). Solo precio + sentimiento.
  Caso extremo del Gap D. Entra en universo y en seguimiento de sentimiento; los
  indicadores se "encienden" solos según acumule barras. Identidad: MiniMax Group Inc,
  Technology/Software-Infrastructure, China, HKG, HKD, mktcap ~156B HKD. La del
  modelo MiniMax/Hailuo. yfinance solo acepta period 1d/5d → firma de IPO ultra-reciente.
- 2513.HK Zhipu (~107 sesiones): momentum corto SÍ calculable.
- Coreanas (Samsung/SK Hynix/Kakao/Ecopro): histórico largo, todo calculable.

**Q7 — Correlación/pares con pocos nombres: ¿válido?**
R: Con ~6 nombres y poco histórico, cointegración/pares es EXPLORATORIO, no factor.
Calcular fuerza relativa (ratio a la mediana de la cesta + ranking) y correlación
rodante como CARACTERIZACIÓN descriptiva, etiquetada non-PIT/non-signal. Honra
invariante 6.

**Q8 — "Capturar las subidas" vs lag T-1.**
R: Honestidad: un sistema EOD NO caza gaps overnight (el +21% de 2513 ya había
ocurrido al tener la barra cerrada). Reformular el objetivo: DETECCIÓN TEMPRANA DE
RÉGIMEN (tendencia + sentimiento creciente) para posicionarse antes de la siguiente
pierna, asumiendo entrada tras cierre. No es cazar el gap; es leer el régimen pronto.

---

## 2. ITEM 1 — Overlay de sentimiento asiático (detallado)

Objetivo: que `sentiment_daily` para tickers asiáticos quede bien fechado y se pueda
cruzar con su precio, reutilizando FinBERT.

Cambios (mínimos, retrocompatibles):
1. `news/aggregate.py::trading_day_for()` — añadir parámetros `tz` y `calendar`
   (default NY_TZ/XNYS). Para .HK usar `Asia/Hong_Kong`+XHKG; .KS/.KQ
   `Asia/Seoul`+XKRX. El corte horario pasa a ser relativo al cierre del mercado
   local (p.ej. 16:00 HKT), no 15:30 ET.
2. Mapa ticker→(tz,calendar,cutoff) — derivado del sufijo (.HK/.KS/.KQ); resto = NASDAQ.
3. `qt news ingest --tickers "2513.HK,005930.KS,..."` (el provider ya los acepta) →
   `qt news curate` → `qt news score` (FinBERT, sin cambios) → `qt news build-factor`
   con la atribución corregida.
4. Tests de regresión: (a) NASDAQ sin cambios (mismo trading_day para un caso ET
   conocido); (b) noticia HK 10:00 HKT → sesión XHKG correcta, no XNYS.

Límites a documentar:
- Cobertura ~10 ítems/ticker, sin archivo → la serie de sentimiento EMPIEZA en el
  harvest (igual que NASDAQ); re-ejecutar a diario para acumular.
- IC del factor necesita acumular sesiones; con días sueltos = nan (esperado).
- Small-caps con solo prensa china: infra-cubiertas hasta Fase 2.

## 3. ITEM 2 — Cesta temática "AI-Asia" como módulo de research aislado

Objetivo: caracterizar tendencia/fuerza relativa de la cesta + overlay de sentimiento
del Item 1, sin tocar el core NASDAQ.

Componentes:
1. Universo propio `index_name="AIASIA"`: seed curado + forward-PIT (mirror del patrón
   nasdaq_directory/seed_universe). NO mezclar con NASDAQ.
2. Precios: Tier A ya cubre la ingesta (`qt ingest --tickers`), con `--end` explícito
   para barras asiáticas frescas; calendario por-ticker en el análisis.
3. Módulo `research/aiasia` (read-only, abre catálogo en read-only como el agente):
   - métricas corto-plazo con guarda de histórico-corto (Q5);
   - fuerza relativa: ratio de cada nombre a la mediana de la cesta + ranking diario;
   - correlación rodante 20/60 sesiones (descriptiva);
   - overlay: sentiment_daily del Item 1 alineado por sesión local.
4. Salida: tabla en terminal (headless) + opcional gráfico ASCII (como el spike
   TimesFM). Caracterización de datos, NUNCA señal (invariante 6).

## 4. Flecos / riesgos (lo que el alto nivel actual deja escapar)
- R1 [atribución] noticia asiática mal fechada por tz/calendario → Item 1 §1. MITIGADO.
- R2 [contaminación PIT] meter asiáticas en el roster NASDAQ → index_name separado. MITIGADO.
- R3 [histórico corto] SMA200/12M sobre 107 barras = basura presentada como válida →
  guarda N/A explícita (Q5). MITIGADO.
- R4 [look-ahead] creer que cazamos el gap overnight → reformulado a régimen (Q8). MITIGADO.
- R5 [cobertura escasa] ~10 ítems/ticker, sin archivo → factor empieza en harvest;
  documentado; Fase 2 para multilingüe/otra fuente. ACEPTADO.
- R6 [moneda] precios .HK en HKD, .KS en KRW, NASDAQ en USD → NO comparar niveles
  absolutos entre mercados; usar retornos/ratios. Añadir nota de divisa por ticker.
- R7 [festivos divergentes] watermark XNYS deja barra asiática 1 sesión atrás →
  `--end` explícito; Tier B (calendario per-market en ingest) si se vuelve serio.
- R8 [retrocompatibilidad] el cambio de trading_day_for NO debe alterar NASDAQ →
  defaults intactos + test de regresión. CRÍTICO, MITIGADO por diseño.
- R9 [sesgo supervivencia] seed AIASIA forward-PIT; no proyectar roster al pasado. DOCUMENTADO.

## 5. Fases (validar en pequeño antes de comprometer)
- FASE 0 (hecha): recon + spike FinBERT sobre titulares reales → señal útil confirmada.
- FASE 1: Item 1 sobre 2513 + Samsung + SK Hynix (3 nombres): parametrizar
  trading_day_for + tests regresión + harvest/score/factor + verificar atribución.
- FASE 2: Item 2 cesta completa (seed AIASIA + módulo research aiasia) sobre Fase 1.
- FASE 3 (FUERA, a discutir): motor intradía 5-min. Bajo retorno hoy (sin histórico
  para backtest, ruido para el horizonte, sin ejecución fina en la cuenta).

## 6. Criterios de éxito
- Item 1: una noticia asiática conocida queda atribuida a la sesión LOCAL correcta
  (test); sentiment_daily de 2513 no-vacío y con signo coherente vs el evento.
- Item 2: tabla de la cesta con métricas corto-plazo + fuerza relativa + overlay
  sentimiento, con N/A explícito donde el histórico no alcanza; NASDAQ intacto
  (suite verde, mismas cifras).
