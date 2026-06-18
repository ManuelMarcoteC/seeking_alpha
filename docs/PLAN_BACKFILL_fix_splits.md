# Plan de backfill — fix #2 doble-ajuste de splits (Opción A)

> Documento de PLAN. NO ejecutado todavía. Redactado en fase B (nivel 2, papel).
> La ejecución es la fase C y requiere OK humano explícito (escribe el lago).
> Contexto completo del bug: skills qtdata-pipeline → references/yfinance-split-adjustment.md
> y PROGRESS.md (raíz del repo).

## 0. Resumen en una frase
El código `reconstruct_as_traded` ya entrega AS-TRADED correcto, pero el raw YA
almacenado en el lago se bajó ANTES del fix (está split-adjusted del vendor, mal).
Hay que re-ingerir los ~1122 tickers con splits para reemplazar ese raw, recomputar
factores y verificar que `ohlcv_daily_adj.close` deja de tener el salto ×split.

## 1. Precondiciones (verificar ANTES de tocar nada)
- [ ] Rama correcta: `fix/corporate-action-adjustments`, working tree con el fix del
      provider + los 5 tests nuevos (sin commitear aún).
- [ ] Suite verde offline + ruff (ya confirmado en fase B: 12 passed en el archivo
      del provider, suite completa verde, ruff ok).
- [ ] Single-writer: NINGÚN otro `qt` escribiendo el lago durante el backfill
      (invariante 1). Cerrar cualquier proceso/agente que abra el catálogo en write.
- [ ] Backup ligero del estado actual del lago para rollback (ver §6).

## 2. Validación en pequeño PRIMERO (subset AAPL) — gate antes del backfill masivo
Esto es el "validar subset antes de run largo" (estilo del usuario). NO seguir al
paso 3 si esto no pasa.

2.1. Crear el probe `scripts/verify_as_traded_unadjust.py` (citado en el skill pero
     AÚN NO existe en el repo). Debe, en vivo contra yfinance:
     - bajar AAPL alrededor del split 4:1 (2020-08-31) con auto_adjust=False,
     - aplicar reconstruct_as_traded,
     - ASSERT: close 27-ago-2020 ≈ 500,04 (real as-traded), volumen ≈ 38,8M,
     - ASSERT: la fila del ex-date (31-ago) queda IGUAL al vendor (no ×4).
2.2. Re-ingerir SOLO AAPL con full-refresh a un lago de PRUEBA o con backup hecho:
     `qt ingest --tickers AAPL --full-refresh` → `qt curate`.
2.3. Query de verificación (el síntoma original del bug):
     ```sql
     SELECT date, close_raw, close AS close_adj
     FROM ohlcv_daily_adj
     WHERE ticker='AAPL' AND date IN ('2020-08-28','2020-08-31');
     ```
     ESPERADO tras el fix: NO debe haber salto ×4 entre 28-ago y 31-ago en close_adj
     (serie continua). Antes del fix: close_adj saltaba 30,26 → 125,17.
     GATE: si el salto sigue, PARAR y diagnosticar. No hacer el backfill masivo.

## 3. Backfill masivo (los ~1122 tickers con splits)
3.1. Obtener la lista exacta:
     ```sql
     SELECT DISTINCT ticker FROM corporate_actions WHERE action_type='split';
     ```
     (~1122 de ~3331). Guardar la lista a fichero para reproducibilidad.
3.2. Re-ingest aditivo con full-refresh (latest-wins por [ticker,date] reemplaza el
     raw viejo). Lanzar como BACKGROUND TRACKED (notify_on_complete), single-writer:
     `qt ingest --tickers "<lista>" --full-refresh`
     - Sólo estos tickers (no todo el universo): minimiza riesgo y tiempo.
     - Ojo tickers foráneos (.KS/.KQ/.HK) con calendario divergente: el `--end` usa
       XNYS global; si algún coreano/HK necesita su última barra, forzar `--end`.
3.3. Re-curate. PUNTO A VALIDAR en C (sutileza del skill): `qt curate` es incremental
     por watermark de archivos curados y podría NO reprocesar tickers ya curados aun
     con raw nuevo. Dos caminos, decidir en C según se observe:
       (a) si curate detecta el raw reescrito y reprocesa → usar `qt curate` normal;
       (b) si NO → script one-off que replica el bloque de factores de curate.py
           (compute_adjustment_factors sobre el slice afectado → parquet_store.upsert
           con FACTORS_KEY, partition_col="year" → Catalog.refresh_views()).
           Snippet de referencia en el skill (sección "Forcing a factor recompute").

## 4. Verificación post-backfill (objetiva, no narrativa)
- [ ] Muestreo de 5-10 tickers US con splits conocidos (AAPL, TSLA, NVDA, etc.):
      query close_raw vs close_adj alrededor de su ex-date → SIN salto ×split.
- [ ] Conteo de filas OHLCV antes/después: debe ser ~igual (re-ingest aditivo, no
      duplica por la clave [ticker,date]); ninguna pérdida masiva.
- [ ] Re-correr `qt validate` y comparar nº de flags: no debe dispararse de forma
      anómala (un cambio de escala de precios mal hecho generaría outliers nuevos).
- [ ] Spot-check de un ticker SIN splits (control): debe quedar idéntico (no-op).
- [ ] 2513.HK / z.ai (0 splits): confirmar que NO cambió (control foráneo limpio).

## 5. Cierre (commit + PR) — sólo tras verificación verde
- [ ] Commit del provider + tests en `fix/corporate-action-adjustments`
      (mensaje: fix(provider): reconstruct as-traded to kill split double-adjust).
- [ ] Actualizar references/yfinance-split-adjustment.md si algo cambió en C.
- [ ] Abrir PR contra master. Incluir en la descripción: el síntoma (salto ×4 en
      AAPL), la causa, Opción A vs B, y la evidencia de verificación del §4.
- [ ] NO mergear sin revisión humana.

## 6. Rollback (si algo sale mal en C)
- El raw viejo se reemplaza por [ticker,date]; para revertir, restaurar el backup
  de `data/raw/.../ohlcv_daily/` y `data/curated/` tomado en §1, y re-curate.
- Hacer el backup ANTES de §2.2 (p.ej. copia de data/ o snapshot de las particiones
  afectadas). Sin backup, el rollback exige re-ingerir desde el vendor (posible pero
  lento). DECISIÓN para C: ¿copia completa de data/ (24 GB libres de sobra) o sólo
  las particiones de los 1122 tickers?

## 7. Riesgos y mitigaciones
- R1: curate no reprocesa raw nuevo → §3.3(b) script de recompute forzado. MITIGADO.
- R2: ticker foráneo con barra faltante por calendario → forzar --end. MITIGADO.
- R3: yfinance falla/throttla en 1122 tickers → ingest aísla fallos por ticker
  (no aborta); re-lanzar recoge los que faltaron (idempotente). MITIGADO.
- R4: escritura concurrente rompe el catálogo → single-writer estricto. MITIGADO.
- R5: pérdida de datos → backup §6 antes de escribir. MITIGADO si se hace el backup.

## 8. Estimación
- Probe + validación AAPL: minutos.
- Re-ingest 1122 tickers full-refresh (histórico 2015→hoy): decenas de minutos
  (más que un incremental, menos que el universo completo ~35-45 min).
- Re-curate + verificación: minutos.
- Total realista: ~1 hora de reloj, la mayor parte desatendida (background tracked).
