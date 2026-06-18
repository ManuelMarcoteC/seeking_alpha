# Notas de trabajo — doble ajuste de splits (yfinance) + universo KRX

> Documento de handoff. Conversación del 16/06/2026 (madrugada). Retomar desde aquí.
> El historial completo de la conversación está en la session DB de Hermes
> (recuperable con session_search "doble ajuste split yfinance Samsung").

---

## 1. Estado del trabajo (qué está hecho y qué falta)

Plan de 4 tareas acordado ("sí a todo"):

1. [HECHO ✓] Fix dedup de splits duplicados (workstream curation).
   - Rama `fix/corporate-action-adjustments` (creada desde master), commit
     `1310325`, **pusheada** a origin. PR pendiente de abrir por el usuario.
   - `_dedup_splits()` en `src/qtdata/curation/adjustments.py`: colapsa splits de
     ratio idéntico dentro de 20 días (ruido de vendor), conserva el ex-date más
     temprano. Raw intacto.
   - 2 tests nuevos en `tests/test_adjustments.py` (caso Samsung 50:1 duplicado →
     50x no 2500x; control de dos splits legítimos lejanos que SÍ aplican ambos).
   - Suite: 231 verde, ruff limpio.

2. [PENDIENTE — bloqueado por decisión del usuario] Fix sistémico del DOBLE AJUSTE.
   - Investigación COMPLETA (ver sección 3). Falta SOLO elegir enfoque A o B
     (sección 4) e implementar. NO se ha tocado código de este fix todavía.

3. [PENDIENTE] Universo KRX propio (Opción B de integración coreana):
   - seed/refresh para índice "KRX" + calendario por mercado (XKRX) en vez del
     global XNYS. Toca config + ingest (end via calendario) + news/aggregate
     (corte 15:30 ET es específico de NY). Con tests.

4. [PENDIENTE] Commits finales por workstream + push + actualizar skill
   `qtdata-pipeline` con los hallazgos.

### Estado del lago (lo ya ejecutado y verificado)
- Precios actualizados de 12/06 → **15/06/2026** (lunes, T-1). NASDAQ full:
  ingest run `56bc935429f2` (ok=3240), curate (3235 filas, 1 cuarentena).
- Tickers coreanos ingeridos y curados (Opción A, tickers sueltos):
  - `005930.KS` Samsung Electronics — 2015-01-02 → 2026-06-15 (2804 sesiones)
  - `000660.KS` SK Hynix — 2015-01-02 → 2026-06-15 (2802 sesiones)
  - `035720.KS` Kakao — 2015-01-02 → 2026-06-15 (2805 sesiones)
  - `247540.KQ` Ecopro BM — 2019-03-05 → 2026-06-15 (1781 sesiones, IPO)
  - ingest run `500b48579e47` (ok=8, 10.287 filas), curate (10.192 OHLCV).
- Factores de Samsung y Kakao recalculados a mano (vía script puntual que llama a
  `compute_adjustment_factors` + upsert) tras el fix #1. El dedup actuó (log:
  "Duplicate split ratio 50.0000 for 005930.KS on 2018-05-16 (prev 2018-05-04)").

---

## 2. Análisis de entrada realizado (cierre 15/06/2026, sobre close_raw)

> Se usó `close_raw` porque la columna "close" ajustada está corrupta (ver bug).
> close_raw de yfinance YA es split-adjusted y continuo → sirve para análisis.

| Métrica               | Samsung (005930.KS) | SK Hynix (000660.KS) |
|-----------------------|---------------------|----------------------|
| Último cierre 15/06   | 337.000 KRW         | 2.288.000 KRW        |
| Régimen               | Alcista (SMA50>200) | Alcista (SMA50>200)  |
| vs SMA50 / SMA200     | +29,9% / +114,0%    | +42,9% / +164,7%     |
| Retorno 3M / 12M      | +79,4% / +500,7%    | +146,0% / +979,2%    |
| YTD 2026              | +162,3%             | +237,9%              |
| RSI(14)               | 60,4                | 62,5                 |
| Pos. rango 52-sem     | 92,3% (techo)       | 96,5% (techo)        |
| DD desde máx hist     | −6,5%               | −3,2%                |
| Vol. anualizada       | 57,1%               | 71,6%                |
| SMA50 / SMA200        | 259.467 / 157.462   | 1.601.120 / 864.512  |

Lectura: ambos en tendencia alcista intacta pero MUY extendidos (techo del rango
anual, lejísimos de SMA50). SK Hynix el más parabólico. Entrada a mercado =
perseguir fuerza; referencias de mejor R/R serían retrocesos a SMA50 o SMA200.
Caracterización de datos, NO recomendación (regla 6 del proyecto).

---

## 3. EL BUG DEL DOBLE AJUSTE — explicación completa

### Qué es un split y por qué se ajusta
Samsung hizo split 50:1 en mayo-2018: 1 acción → 50, precio ÷50. Nadie gana/pierde
(1 acción a 2.650.000 = 50 acciones a 53.000). Pero el gráfico "tal cual" muestra
una caída del 98% falsa. El AJUSTE reescribe la historia previa ÷50 para que la
serie sea continua.

Dos conceptos clave:
- **AS-TRADED**: el precio real de pantalla de cada día (Samsung 2015 ≈ 26.000).
- **SPLIT-ADJUSTED**: esa historia reescrita en unidades de hoy (Samsung 2015 ≈ 520).

### Lo que el pipeline ASUME (diseño CRSP, "flag never mutate")
Cadena de diseño:
1. Guardar el crudo AS-TRADED (el código dice literal: "raw layer stores prices AS TRADED").
2. Calcular `adj_factor` por (ticker, fecha) desde la tabla de splits/dividendos.
3. Aplicar al leer, en la vista `ohlcv_daily_adj`:
   - `close_ajustado = close_crudo × adj_factor`
   - `volumen_ajustado = volumen_crudo / split_factor`
   - guarda `close_raw` aparte.

TODO descansa en que el Paso 1 recibe AS-TRADED. **Esa suposición es falsa.**

### Lo que yfinance REALMENTE da (verificado en vivo)
`auto_adjust=False` NO significa "crudo sin tocar". Esa opción solo controla los
DIVIDENDOS. yfinance **YA entrega el precio (y el volumen) ajustados por SPLITS**
de fábrica.

PRUEBA — AAPL, split 4:1 del 31-ago-2020:
- Precio real as-traded 27-ago-2020 ≈ **499 USD**.
- yfinance (auto_adjust=False) dio **125,01** para ese día. 125,01 × 4 = 500 ≈ 499.
  → ya estaba dividido por 4.
- Volumen real 27-ago ≈ **38,8M**. yfinance dio **155,5M**. 38,8 × 4 = 155.
  → el volumen también viene split-adjusted.

### El doble ajuste
- El dato ENTRA ya split-adjusted (Yahoo lo hizo).
- El pipeline, creyéndolo as-traded, lo ajusta OTRA VEZ por splits.
- → el split se aplica DOS VECES.

Evidencia en el lago (AAPL):
| Fecha        | close_raw | close_adj (vista) |
|--------------|-----------|-------------------|
| 27-ago-2020  | 125,01    | 30,31             |
| 31-ago-2020  | 129,04    | 125,17            |
close_adj salta ×4 (de 30 a 125) de un día a otro = el split aplicado 2 veces.
Debería ser una serie continua sin salto.

Volumen aún peor: la vista DIVIDE por split_factor → 155M / (1/4) = 620M, cuando
el real era 38,8M (inflado ×16).

Samsung era más extremo porque yfinance ADEMÁS duplicaba el split (50×50=2500) —
eso es el fix #1, ya resuelto. Al arreglarlo se destapó este, que es el de fondo.

### Alcance
- Afecta a **CUALQUIER ticker con splits de TODO el universo NASDAQ**, no solo KRX.
- La columna "close" (y el volumen) de `ohlcv_daily_adj` está corrupta para todos ellos.
- `close_raw` está BIEN (es split-adjusted continuo) → por eso los análisis se hicieron con él.
- Tickers sin splits (SK Hynix) están limpios.

---

## 4. DECISIÓN PENDIENTE para el fix #2 (elegir A o B)

**Opción A — devolver el dato a as-traded (arreglar en el provider).**
Des-ajustar multiplicando de vuelta por los splits futuros en `yfinance_provider`,
para reconstruir el precio real de cada día. El Paso 1 vuelve a cumplir su promesa;
factores/vista/231 tests intactos.
- A favor: respeta el diseño original; cambio quirúrgico en un sitio.
- En contra: frágil en incremental (cada split nuevo hace que yfinance reescriba
  el pasado del ticker; hay que recalcular el des-ajuste).

**Opción B — aceptar que la verdad es split-adjusted (cambiar la convención).**
`adj_factor` deja de aplicar splits (ya aplicados) y solo aplica dividendos; el
volumen deja de dividirse por split; se renombra el invariante
("raw = split-adjusted, no as-traded"). Toca vista + factores + varios tests.
- A favor: honesto con el vendor; robusto en incremental; menos sorpresas.
- En contra: toca más archivos hoy; pierdes el as-traded puro salvo reconstruirlo aparte.

Ambas correctas. Tercera vía ofrecida: preparar el DIFF de ambas lado a lado antes
de decidir sobre código concreto.

→ **PRÓXIMO PASO mañana: el usuario elige A o B (o pide ver los diffs).**
