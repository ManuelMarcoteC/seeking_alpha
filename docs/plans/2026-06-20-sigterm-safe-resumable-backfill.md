# SIGTERM-Safe Resumable Backfill Implementation Plan

> **For Hermes:** Use subagent-driven-development to implement task-by-task. Autonomía nivel 2: NO commitear ni tocar `data/` sin OK explícito del usuario. El usuario prefiere validar con subset pequeño y crear cronjobs EN PAUSA antes de soltar nada unattended.

**Goal:** Hacer que el backfill/ingesta de los ~1122 tickers sobreviva a una parada de contenedor (`docker stop` / `kubectl delete pod` / `timeout` / systemd stop), confirmando el progreso por-ticker en vez de perder el trabajo en vuelo, y exponiendo el progreso parcial con un exit code distintivo y un summary fiel.

**Architecture:** El TIL de SOFA describe el fallo: en CPython, `Ctrl+C` (SIGINT) desenrolla la pila y ejecuta `finally`, pero los orquestadores envían **SIGTERM**, cuya disposición por defecto **mata el proceso sin levantar nada** — ni unwind, ni `finally`, ni flush — y tras el grace period llega SIGKILL (incatchable). qtdata YA está parcialmente blindado (DuckDB autocommitea cada `set_watermark`, y los raw parquet llevan `run_id` en el nombre → re-fetch idempotente), así que NO perdemos todo como en el caso del post. El gap real es: (a) un SIGTERM a mitad del bucle de tickers no produce summary parcial ni cierra limpio la conexión DuckDB (riesgo de WAL/lock colgado); (b) no hay frontera de parada cooperativa; (c) no hay forma de distinguir "terminó" de "lo pararon a la mitad". La solución: convertir SIGTERM en un `SystemExit` ordinario mediante un context manager reutilizable, comprobar una bandera de parada entre tickers/batches para drenar limpio, y devolver exit code 143 con summary parcial.

**Tech Stack:** Python 3.12, signal (stdlib), DuckDB, typer, pytest. Sin dependencias nuevas.

**Procedencia:** TIL "S3 resume-state object only appears when the process is killed: Python finally-based flush runs on Ctrl+C but never on SIGTERM container stops" (agents.stackoverflow.com, Martin Eve, 2026-06-11). Refuerzo: TIL "Resuming a multi-step agent loop must replay the attempt log, not reset session counters".

---

## Contexto del código (leer antes de empezar)

- `src/qtdata/ingestion/ingest.py` → `ingest()`: bucle de planificación + `_ingest_batched` / `_ingest_per_ticker`. Cada ticker exitoso llama `_record_result` → `set_watermark` (commit DuckDB inmediato) + `record_fetch` (manifest). El aislamiento de fallos por-ticker ya existe (`except Exception` por ticker). **No hay punto de parada cooperativa ni manejo de SIGTERM.**
- `src/qtdata/ingestion/watermarks.py` → `set_watermark`: `INSERT OR REPLACE ... current_timestamp`. DuckDB en modo autocommit: cada watermark es durable al instante. **Esta es la "committed log / source of truth" del TIL de resume.**
- `src/qtdata/storage/catalog.py` → `Catalog`: `__exit__` → `close()`. Si SIGTERM mata el proceso, `__exit__` NO corre → la conexión DuckDB no cierra limpio.
- `src/qtdata/cli.py:137-162` → comando `ingest`: abre `Catalog` como context manager y llama `run_ingest` (alias de `ingest`).
- Tests espejo: `tests/test_ingest.py`, `tests/test_ingest_batch.py`. Nuevo: `tests/test_graceful_shutdown.py`.

**Invariante (del TIL de resume):** el watermark commiteado es la única fuente de verdad; reanudar = `watermark + 1 sesión` (ya implementado en `_effective_start`). El backfill es naturalmente resumible: basta re-lanzar `qt ingest` y los tickers ya completados se saltan (`eff is None`). Lo que falta es PARAR limpio y REPORTAR lo hecho.

---

### Task 1: Context manager `terminable` que convierte SIGTERM en SystemExit

**Objective:** Primitiva reutilizable: instala un handler de SIGTERM que lanza `SystemExit(143)` (128+15), restaura el handler previo al salir (no lo filtra), y expone una bandera `should_stop` para parada cooperativa.

**Files:**
- Create: `src/qtdata/ingestion/shutdown.py`
- Test: `tests/test_graceful_shutdown.py`

**Step 1: Write failing test**

```python
# tests/test_graceful_shutdown.py
"""SIGTERM -> ordinary unwind, so finally/cleanup runs on container stop.

CPython default disposition for SIGTERM terminates without raising; Ctrl+C
(SIGINT) raises KeyboardInterrupt. Manual testing with Ctrl+C therefore
validates a path that never runs under docker/k8s. We convert SIGTERM into a
catchable SystemExit and verify cleanup fires.

Ported from the SOFA TIL on SIGTERM/finally asymmetry.
"""
import os
import signal
import time

import pytest

from qtdata.ingestion import shutdown


def test_sigterm_raises_system_exit_and_runs_finally():
    cleaned = {"ran": False}
    # Sentinela: si el código aún no instaló su handler, que el SIGTERM se vea
    # como fallo de test y NO mate el runner.
    def _sentinel(signum, frame):
        raise AssertionError("SIGTERM not handled by terminable yet")
    prev = signal.signal(signal.SIGTERM, _sentinel)
    try:
        with pytest.raises(SystemExit) as exc:
            with shutdown.terminable():
                try:
                    os.kill(os.getpid(), signal.SIGTERM)
                    time.sleep(0.5)  # delivery determinista; el handler interrumpe
                finally:
                    cleaned["ran"] = True
        assert exc.value.code == 143
        assert cleaned["ran"] is True
    finally:
        signal.signal(signal.SIGTERM, prev)


def test_terminable_restores_previous_handler():
    before = signal.getsignal(signal.SIGTERM)
    with shutdown.terminable():
        assert signal.getsignal(signal.SIGTERM) is not before
    assert signal.getsignal(signal.SIGTERM) is before  # no se filtra


def test_should_stop_flag_flips_on_signal():
    with shutdown.terminable() as guard:
        assert guard.should_stop is False
        guard._request_stop(signal.SIGTERM, None)  # simula entrega
        assert guard.should_stop is True
```

**Step 2: Run test to verify failure**

Run: `pytest tests/test_graceful_shutdown.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qtdata.ingestion.shutdown'`

**Step 3: Write minimal implementation**

```python
# src/qtdata/ingestion/shutdown.py
"""Make SIGTERM a catchable unwind so finally/cleanup runs on container stop.

Two modes, composable:
- As a context manager, SIGTERM raises SystemExit(143) (128 + SIGTERM). The
  stack unwinds, `finally` blocks run, DuckDB connections close, partial
  summaries surface. The previous handler is restored on exit (never leaked).
- For cooperative draining, `guard.should_stop` flips True on first signal so a
  loop can finish the current unit and break BEFORE the SystemExit fires (set
  `raise_on_signal=False` to only flip the flag and not raise).

Ported from the SOFA TIL "S3 resume-state object only appears when the process
is killed" — CPython's SIGINT/SIGTERM asymmetry, not specific to any framework.
"""
from __future__ import annotations

import logging
import signal
from contextlib import contextmanager
from types import FrameType

logger = logging.getLogger(__name__)

_EXIT_SIGTERM = 143  # 128 + 15


class _Guard:
    def __init__(self, raise_on_signal: bool) -> None:
        self.should_stop = False
        self._raise = raise_on_signal

    def _request_stop(self, signum: int, frame: FrameType | None) -> None:
        self.should_stop = True
        logger.warning("SIGTERM received: draining; will exit %d after cleanup", _EXIT_SIGTERM)
        if self._raise:
            raise SystemExit(_EXIT_SIGTERM)


@contextmanager
def terminable(raise_on_signal: bool = True):
    """Install a SIGTERM->SystemExit(143) handler for the duration of the block.

    Restores the previous handler on exit. Yields a guard whose `should_stop`
    becomes True when SIGTERM is delivered, enabling cooperative loop draining.
    """
    guard = _Guard(raise_on_signal=raise_on_signal)
    previous = signal.signal(signal.SIGTERM, guard._request_stop)
    try:
        yield guard
    finally:
        signal.signal(signal.SIGTERM, previous)  # don't leak the handler
```

**Step 4: Run test to verify pass**

Run: `pytest tests/test_graceful_shutdown.py -v`
Expected: PASS — 3 passed.

**Step 5: Commit** (tras OK)

```bash
git add src/qtdata/ingestion/shutdown.py tests/test_graceful_shutdown.py
git commit -m "feat(ingestion): terminable() context manager — SIGTERM-safe unwind"
```

---

### Task 2: Parada cooperativa entre grupos/tickers en el orquestador

**Objective:** Que `ingest()` compruebe `should_stop` entre unidades de trabajo (grupos batched y tickers per-ticker) y drene limpio: deja de empezar nuevas unidades, conserva todo lo ya commiteado, y marca el summary como interrumpido. Re-lanzar `qt ingest` reanuda exacto desde los watermarks (ya implementado).

**Files:**
- Modify: `src/qtdata/ingestion/ingest.py` (firmas internas + `ingest()` + `IngestSummary`)
- Test: `tests/test_graceful_shutdown.py`

**Step 1: Write failing test**

```python
# añadir a tests/test_graceful_shutdown.py
from datetime import date
from qtdata.ingestion.ingest import ingest as run_ingest


def test_ingest_drains_on_stop_and_preserves_committed(settings, catalog, monkeypatch):
    """Si se solicita parada tras el primer ticker, los siguientes no se fetchan,
    pero el watermark del primero queda commiteado (resumible)."""
    catalog.init_schema()
    from qtdata.ingestion import shutdown

    # provider sintético determinista; dispara stop después del 1er record
    calls = {"n": 0}
    real_record = None

    import qtdata.ingestion.ingest as ing
    orig = ing._record_result
    def _counting_record(*args, **kwargs):
        orig(*args, **kwargs)
        calls["n"] += 1
        # tras el primer ticker registrado, pedir parada
        if calls["n"] == 1 and "guard" in kwargs:
            kwargs["guard"].should_stop = True
    monkeypatch.setattr(ing, "_record_result", _counting_record)

    summary = run_ingest(
        settings, catalog, ["AAA", "BBB", "CCC"],
        provider_name="synthetic", full_refresh=True,
    )
    # al menos un ticker commiteado, y el resto NO procesado por la parada
    assert summary.ok >= 1
    assert summary.interrupted is True
    assert summary.ok < 3
```

(Nota para el implementador: ajustar el mecanismo exacto de inyección del `guard` al firmar las funciones internas en el Step 3; el test verifica el comportamiento observable, no la firma.)

**Step 2: Run test to verify failure**

Run: `pytest tests/test_graceful_shutdown.py::test_ingest_drains_on_stop_and_preserves_committed -v`
Expected: FAIL — `AttributeError: 'IngestSummary' object has no attribute 'interrupted'`

**Step 3: Write minimal implementation**

En `src/qtdata/ingestion/ingest.py`:

1. Añadir campo al summary:

```python
@dataclass
class IngestSummary:
    run_id: str
    ok: int = 0
    empty: int = 0
    skipped: int = 0
    failed: int = 0
    rows: int = 0
    interrupted: bool = False  # True si SIGTERM drenó el run antes de acabar
    failures: list[tuple[str, str, str]] = field(default_factory=list)
```

2. Pasar un `guard` opcional por la cadena. Firmar `_ingest_per_ticker` y `_ingest_batched` con `guard=None` y comprobar al inicio de cada iteración de unidad de trabajo:

```python
def _ingest_per_ticker(settings, catalog, provider, plan, end, run_id, summary, guard=None):
    for ticker, per_dataset in plan.items():
        if guard is not None and guard.should_stop:
            summary.interrupted = True
            return
        for dataset, eff_start in per_dataset.items():
            ...  # cuerpo existente sin cambios
```

```python
def _ingest_batched(settings, catalog, provider, plan, end, run_id, summary, guard=None):
    groups: dict[tuple, list[str]] = defaultdict(list)
    for ticker, per_dataset in plan.items():
        signature = tuple(sorted((str(ds), es) for ds, es in per_dataset.items()))
        groups[signature].append(ticker)
    for tickers in groups.values():
        if guard is not None and guard.should_stop:
            summary.interrupted = True
            return
        ...  # cuerpo existente; pasar guard al fallback per-ticker:
        # _ingest_per_ticker(..., summary, guard=guard)
```

3. En `ingest()`, envolver el despacho con `terminable()` y pasar el guard:

```python
    from qtdata.ingestion.shutdown import terminable
    ...
    if not plan:
        return summary

    with terminable() as guard:
        if isinstance(provider, BatchFetchProtocol) and len(plan) > 1:
            _ingest_batched(settings, catalog, provider, plan, end, run_id, summary, guard=guard)
        else:
            _ingest_per_ticker(settings, catalog, provider, plan, end, run_id, summary, guard=guard)
    return summary
```

(Para que el test del Step 1 pueda inyectar el stop vía `_record_result`, pasar `guard` también a `_record_result` como kwarg keyword-only opcional `guard=None`, o —más limpio— exponer el guard en el summary durante el run. Elegir la opción que mantenga las firmas públicas estables; documentar la decisión en el commit.)

**Step 4: Run test to verify pass**

Run: `pytest tests/test_graceful_shutdown.py -v && pytest tests/test_ingest.py tests/test_ingest_batch.py -v`
Expected: PASS — el nuevo test pasa y TODOS los de ingesta legacy siguen verdes (el guard es opcional; sin SIGTERM el comportamiento es idéntico).

**Step 5: Commit** (tras OK)

```bash
git add src/qtdata/ingestion/ingest.py tests/test_graceful_shutdown.py
git commit -m "feat(ingestion): cooperative SIGTERM draining between tickers/batches"
```

---

### Task 3: Exit code 143 + summary parcial en la CLI

**Objective:** Que `qt ingest` salga con código 143 cuando fue interrumpido (no 0 — "exit 0 no significa que la tarea terminó", otro TIL de SOFA), imprimiendo el summary parcial para que un operador o cron sepa que debe re-lanzar.

**Files:**
- Modify: `src/qtdata/cli.py:137-162` (comando `ingest`)
- Test: `tests/test_cli.py`

**Step 1: Write failing test**

```python
# añadir a tests/test_cli.py
def test_ingest_exits_143_when_interrupted(monkeypatch):
    from typer.testing import CliRunner
    from qtdata.cli import app
    import qtdata.cli as cli

    class _FakeSummary:
        run_id = "deadbeef"; ok = 5; empty = 0; skipped = 0
        failed = 0; rows = 100; interrupted = True; failures = []

    monkeypatch.setattr(cli, "run_ingest", lambda *a, **k: _FakeSummary())
    monkeypatch.setattr(cli, "_resolve_tickers", lambda *a, **k: ["AAA"])
    result = CliRunner().invoke(app, ["ingest", "--tickers", "AAA"])
    assert result.exit_code == 143
```

**Step 2: Run test to verify failure**

Run: `pytest tests/test_cli.py::test_ingest_exits_143_when_interrupted -v`
Expected: FAIL — `assert 0 == 143`.

**Step 3: Write minimal implementation**

Al final del comando `ingest` en `src/qtdata/cli.py`, tras `_print_ingest_summary(summary)`:

```python
    _print_ingest_summary(summary)
    if getattr(summary, "interrupted", False):
        console.print(
            "[yellow]Ingesta interrumpida por SIGTERM: progreso commiteado por "
            "watermark. Re-lanza `qt ingest` para reanudar.[/yellow]"
        )
        raise typer.Exit(143)
```

**Step 4: Run test to verify pass**

Run: `pytest tests/test_cli.py -v`
Expected: PASS — el nuevo test y los existentes (que devuelven `interrupted=False` → exit 0).

**Step 5: Commit** (tras OK)

```bash
git add src/qtdata/cli.py tests/test_cli.py
git commit -m "feat(cli): ingest exits 143 with partial summary on SIGTERM interrupt"
```

---

### Task 4: Test de señal real end-to-end (la prueba que el TIL exige)

**Objective:** No confiar: enviar un SIGTERM real a mitad de proceso y asertar que (a) sale por SystemExit/143, (b) lo ya commiteado sobrevive, (c) lo en vuelo queda para reintento. El TIL es explícito: "the SIGTERM path is testable with a real signal rather than trust".

**Files:**
- Modify: `tests/test_graceful_shutdown.py`

**Step 1: Write the test**

```python
# añadir a tests/test_graceful_shutdown.py
def test_real_sigterm_midrun_preserves_committed_watermarks(settings, catalog, monkeypatch):
    """Envía un SIGTERM real durante el procesamiento del 2º ticker; asegura que
    el watermark del 1º quedó commiteado y el run sale por la vía catchable."""
    import os, signal, time
    catalog.init_schema()
    import qtdata.ingestion.ingest as ing
    from qtdata.ingestion.watermarks import get_watermark
    from qtdata.models import Dataset

    orig = ing._record_result
    seen = {"n": 0}
    def _record_then_signal(*args, **kwargs):
        orig(*args, **kwargs)
        seen["n"] += 1
        if seen["n"] == 1:
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(0.3)  # entrega determinista
    monkeypatch.setattr(ing, "_record_result", _record_then_signal)

    with pytest.raises(SystemExit) as exc:
        ing.ingest(settings, catalog, ["AAA", "BBB", "CCC"],
                   provider_name="synthetic", full_refresh=True)
    assert exc.value.code == 143
    # el primer ticker commiteado sobrevive (resumible)
    wm = get_watermark(catalog.conn, "synthetic", Dataset.OHLCV_DAILY, "AAA")
    assert wm is not None
```

**Step 2: Run test to verify**

Run: `pytest tests/test_graceful_shutdown.py::test_real_sigterm_midrun_preserves_committed_watermarks -v`
Expected: PASS. (Si el provider sintético no soporta `full_refresh`/orden determinista, ajustar el fixture; el assert clave es watermark persistido + SystemExit 143.)

**Trap del TIL a respetar:** registrar SIEMPRE el sentinela de SIGTERM en el test ANTES de invocar el código bajo prueba (ya cubierto por `terminable`, pero en tests que tocan señales fuera del context manager, instalar un handler que levante `AssertionError` para que un SIGTERM mal entregado falle el test en vez de matar el runner).

**Step 3: Commit** (tras OK)

```bash
git add tests/test_graceful_shutdown.py
git commit -m "test(ingestion): real-SIGTERM mid-run preserves committed watermarks"
```

---

### Task 5: Cierre durable de DuckDB ante interrupción

**Objective:** Garantizar que la conexión DuckDB cierra limpio (checkpoint del WAL) aunque llegue SIGTERM, evitando un `.wal` colgado o lock en el siguiente arranque. Como `terminable` convierte SIGTERM en `SystemExit`, el `with Catalog(...)` del CLI ya ejecuta `__exit__` → `close()`; este task lo verifica explícitamente y añade un checkpoint defensivo.

**Files:**
- Modify: `src/qtdata/storage/catalog.py` (`close()`)
- Test: `tests/test_graceful_shutdown.py`

**Step 1: Write failing test**

```python
# añadir a tests/test_graceful_shutdown.py
def test_catalog_close_checkpoints_wal(settings):
    from qtdata.storage.catalog import Catalog
    cat = Catalog(settings)
    cat.init_schema()
    cat.conn.execute(
        "INSERT OR REPLACE INTO watermarks VALUES "
        "('synthetic','ohlcv_daily','AAA','2026-01-02','run1', current_timestamp)"
    )
    cat.close()
    # reabrir: el dato persiste y no hay lock colgado
    cat2 = Catalog(settings, read_only=True)
    row = cat2.conn.execute(
        "SELECT high_water_date FROM watermarks WHERE ticker='AAA'"
    ).fetchone()
    cat2.close()
    assert row is not None
```

**Step 2: Run test to verify**

Run: `pytest tests/test_graceful_shutdown.py::test_catalog_close_checkpoints_wal -v`
Expected: probablemente PASS ya (DuckDB autocommit + checkpoint al cerrar). Si pasa sin cambios, el test queda como regresión. Si falla por WAL, aplicar Step 3.

**Step 3 (solo si el test falla): checkpoint defensivo en `close()`**

```python
    def close(self) -> None:
        try:
            try:
                self.conn.execute("CHECKPOINT")  # flush WAL into the main file
            except Exception:  # noqa: BLE001 — read-only conn o ya cerrada
                pass
            self.conn.close()
        except Exception:  # noqa: BLE001 — double-close is a no-op
            pass
```

**Step 4: Run test to verify pass**

Run: `pytest tests/test_graceful_shutdown.py -v`
Expected: PASS.

**Step 5: Commit** (tras OK)

```bash
git add src/qtdata/storage/catalog.py tests/test_graceful_shutdown.py
git commit -m "feat(storage): defensive WAL checkpoint on Catalog.close()"
```

---

### Task 6: Suite completa + ensayo controlado (con OK)

**Objective:** Confirmar no-regresión y ensayar una parada real antes de cualquier uso unattended.

**Step 1: Run full suite**

Run: `pytest -q`
Expected: todo verde. La ingesta sin señales se comporta idéntica (guard opcional, default no-interrumpido).

**Step 2 (requiere OK — usa un `data_dir` de prueba, NO producción):** ensayo de parada real

```bash
# terminal A: lanzar ingesta de un subset y pararla a mano a los pocos segundos
QT_DATA_DIR=/tmp/qt_drill qt ingest --universe NASDAQ --provider synthetic &
PID=$!
sleep 5
kill -TERM $PID            # SIGTERM, como docker stop
echo "exit: $?"           # esperar 143
# re-lanzar: debe reanudar desde watermarks, saltando lo ya hecho
QT_DATA_DIR=/tmp/qt_drill qt ingest --universe NASDAQ --provider synthetic
```

**Criterios de aceptación del ensayo:**
- Primer run sale con 143 y un summary parcial coherente (`ok < total`, `interrupted=True`).
- No queda `.wal` colgado ni lock; el segundo run abre el catálogo sin error.
- El segundo run salta los tickers ya commiteados (`skipped` ≈ los completados antes) y termina con exit 0.
- Cero duplicados en el raw layer (cada run usa su propio `run_id`; el re-fetch es idempotente).

**Step 3: Cronjob unattended (EN PAUSA primero)**

Cuando se programe el backfill como cron, crearlo **pausado**, dispararlo manual una vez para validar coste/exit code, y solo entonces activarlo. El exit 143 permite que el wrapper/cron distinga "me pararon, re-lánzame" de "fallé de verdad".

---

## Notas de diseño y mantenimiento

- **Por qué qtdata ya parte con ventaja sobre el TIL:** el caso del post perdía TODO porque su estado vivía en un objeto S3 reescrito cada 25 records y solo se flushaba en `finally`. Aquí cada watermark se autocommitea en DuckDB al instante y los raw files son inmutables con `run_id` — el "checkpoint interval" efectivo es 1 ticker. Este plan cierra el resto: parada limpia, reporte fiel y exit code distintivo.
- **Idempotencia primero (regla del TIL):** "none of it is safe unless re-processing a record is idempotent". Re-fetchar un ticker ya commiteado es seguro: `_effective_start` arranca en `watermark + 1 sesión`, y un re-fetch con la misma ventana escribe un raw nuevo con `run_id` distinto sin corromper la verdad curada (la curación es latest-wins en OHLCV / first-wins en news). Verificar esta invariante si algún día se cambia el naming de raw.
- **SIGKILL sigue siendo incatchable:** si el grace period del orquestador es muy corto, un batch yfinance largo podría comerse el SIGKILL a mitad. Mitigación: mantener `yfinance_batch_size` moderado para que cada unidad de trabajo (grupo) sea corta, de modo que `should_stop` se chequee con frecuencia. No requiere código nuevo; es operativo.
- **No materializar estado derivado:** igual que el TIL de resume ("replay the log, not reset counters"), aquí NO se introduce un fichero de progreso paralelo que pueda divergir del watermark. El watermark DuckDB ES el log committeado; cualquier contador de resumen se deriva de él en cada arranque.
