# Syndicated Headline Dedup (pre-FinBERT) Implementation Plan

> **For Hermes:** Use subagent-driven-development to implement task-by-task. Autonomía nivel 2: NO commitear ni tocar `data/` sin OK explícito del usuario. Validar con subset pequeño antes de cualquier rebuild real.

**Goal:** Colapsar noticias sindicadas (misma historia, distintos medios → distintos `article_id`) en un único evento por (ticker, día) ANTES de promediar el sentiment, para que `n_articles` y la media ponderada dejen de inflarse cuando 8 outlets publican el mismo titular.

**Architecture:** Un módulo determinista nuevo `news/dedup.py` (Jaccard sobre conjunto de tokens + mapa de sinónimos canónicos + sentinela `_skip`), portado del TIL de SOFA (token-set Jaccard, sin embeddings: cero latencia, cero coste API, explicable). El colapso se aplica DENTRO de `build_sentiment_daily` (`news/aggregate.py`), una vez que las filas ya están bucketizadas por (ticker, `date`) — esa bucketización ES la guarda de ventana temporal que el TIL recomienda contra falsos merges, y el ticker es la guarda de diversidad. La capa curated (`news_articles`, `news_ticker_sentiment`) NO se toca: la integridad PIT y el first-capture-wins se conservan; el dedup es una proyección de solo lectura sobre el factor gold. Opt-in por settings con default conservador.

**Tech Stack:** Python 3.12, pandas, pytest. Sin dependencias nuevas.

**Procedencia:** TIL "Collapsing syndicated duplicate news headlines without embeddings: token-set Jaccard plus a synonym map" (agents.stackoverflow.com, Anton Yakutovich, 2026-06-14).

---

## Contexto del código (leer antes de empezar)

- `src/qtdata/news/aggregate.py` → `build_sentiment_daily()`: lee `news_ticker_sentiment`, calcula `effective_ts = max(published_at, ingested_at)`, atribuye cada fila a una sesión de su propio mercado (`date`), filtra por `news_relevance_floor`, y agrupa por `[ticker, date]` con `_aggregate()` que hace `np.average(score, weights=weight)` y `n_articles = group["article_id"].nunique()`. **Aquí es donde los duplicados sindicados inflan el factor.**
- `src/qtdata/news/curate.py`: append-only, first-capture-wins por `article_id`. NO modificar — el dedup es downstream.
- `src/qtdata/config.py` → `Settings`: bloque `# news / sentiment`. Aquí van los flags nuevos.
- `src/qtdata/news/__init__.py`: paquete; el módulo nuevo cuelga aquí.
- Tests espejo: `tests/test_news_aggregate.py` (estilo a calcar), nuevo `tests/test_news_dedup.py`.

**Decisión de diseño clave (no-look-ahead):** el dedup colapsa solo entre filas que YA comparten (ticker, día de atribución). Dos copias del mismo evento que caen en días distintos NO se fusionan (correcto: son observables en momentos distintos). Esto respeta la disciplina PIT del repo y hace el merge explicable y acotado.

---

### Task 1: Crear el módulo de tokenización determinista

**Objective:** Implementar `words()` (tokenizador con stopwords + sinónimos + sentinela `_skip`) y `similarity()` (Jaccard de conjuntos con manejo explícito de vacíos), portados del TIL.

**Files:**
- Create: `src/qtdata/news/dedup.py`
- Test: `tests/test_news_dedup.py`

**Step 1: Write failing test**

```python
# tests/test_news_dedup.py
"""Token-set Jaccard dedup of syndicated headlines (no embeddings).

Portado del TIL de SOFA: el trabajo real lo hace el tokenizador (stopwords +
mapa de sinónimos canónicos + sentinela _skip), no el Jaccard.
"""
import pandas as pd
import pytest

from qtdata.news import dedup


class TestWords:
    def test_lowercases_strips_punct_and_drops_stopwords(self):
        # "the"/"a"/"of"/"near" caen; tokens de 1 char caen
        toks = dedup.words("The drone, a UAV, of Iran near Hormuz")
        assert "the" not in toks and "a" not in toks and "of" not in toks
        assert all(len(t) > 1 for t in toks)

    def test_synonyms_map_to_canonical(self):
        # drone/drones/uav/uavs -> uav ; intercept/engage -> engage
        assert "uav" in dedup.words("drones")
        assert "uav" in dedup.words("UAV")
        assert "engage" in dedup.words("intercepts")
        assert "drone" not in dedup.words("drone")  # canonicalizado a uav

    def test_skip_sentinel_drops_filler_verbs(self):
        # says/said/report -> _skip -> dropped (no token de identidad espurio)
        assert "_skip" not in dedup.words("Apple says revenue up")
        assert "says" not in dedup.words("Apple says revenue up")
        assert "said" not in dedup.words("Tesla said output rose")


class TestSimilarity:
    def test_two_empty_sets_are_identical(self):
        assert dedup.similarity("the a of", "in on at") == 1.0  # ambos -> vacíos

    def test_one_empty_one_not_is_disjoint(self):
        assert dedup.similarity("the", "Iran drone Hormuz") == 0.0

    def test_syndicated_pair_exceeds_threshold(self):
        a = "Iran intercepts drone near Hormuz"
        b = "Iranian forces engage UAV over Strait of Hormuz"
        assert dedup.similarity(a, b) >= 0.5

    def test_unrelated_sharing_topic_word_stays_below(self):
        a = "Iran intercepts drone near Hormuz"
        b = "Hormuz shipping insurance rates climb sharply"
        assert dedup.similarity(a, b) < 0.5
```

**Step 2: Run test to verify failure**

Run: `pytest tests/test_news_dedup.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qtdata.news.dedup'`

**Step 3: Write minimal implementation**

```python
# src/qtdata/news/dedup.py
"""Deterministic syndicated-headline dedup: token-set Jaccard + synonym map.

No embeddings: zero latency, zero API cost, fully explainable ("these merged
because both reduced to {uav, engage, hormuz}"). The tradeoff is a hand-curated,
domain-specific synonym table — cheap for a bounded finance-news vocabulary.

Ported from the SOFA TIL "Collapsing syndicated duplicate news headlines without
embeddings" (token-set Jaccard plus a synonym map).
"""
from __future__ import annotations

import re

# Filler verbs that survive stopword removal but carry no identity. Mapping them
# to _skip (which is itself a stopword) prevents "X says ..." / "Y says ..."
# from sharing an inflated `says` token and over-merging.
_SKIP = "_skip"

# Domain synonym map: surface form -> canonical token. Keep this small and
# finance-news specific. Order does not matter (applied per-token after split).
_SYNONYMS: dict[str, str] = {
    # filler verbs -> _skip
    "says": _SKIP, "said": _SKIP, "report": _SKIP, "reports": _SKIP,
    "reported": _SKIP, "according": _SKIP,
    # corporate-action / market vocabulary canonicalization (extend as needed)
    "shares": "stock", "share": "stock", "equity": "stock", "equities": "stock",
    "rises": "rise", "rose": "rise", "gains": "rise", "gained": "rise",
    "jumps": "rise", "jumped": "rise", "climbs": "rise", "climbed": "rise",
    "falls": "fall", "fell": "fall", "drops": "fall", "dropped": "fall",
    "slumps": "fall", "slumped": "fall", "slides": "fall", "slid": "fall",
    "beats": "beat", "tops": "beat", "misses": "miss", "missed": "miss",
    "acquires": "acquire", "acquired": "acquire", "buys": "acquire",
    "merger": "acquire", "merges": "acquire",
    "lawsuit": "sue", "sues": "sue", "sued": "sue",
    "earnings": "earnings", "profit": "earnings", "profits": "earnings",
    "revenue": "revenue", "sales": "revenue",
    "upgrade": "upgrade", "upgrades": "upgrade", "raised": "upgrade",
    "downgrade": "downgrade", "downgrades": "downgrade", "cut": "downgrade",
}

# Stopwords: structural words plus the _skip sentinel.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for",
        "by", "with", "from", "as", "is", "are", "be", "its", "it", "this",
        "that", "after", "over", "near", "amid", "into", "up", "down", "new",
        _SKIP,
    }
)

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def words(text: str) -> set[str]:
    """Tokenize a headline into a canonical content-word set.

    Lowercase, strip non-alphanumerics, split on whitespace, map synonyms onto
    canonical tokens, drop stopwords and any single-character token.
    """
    if not text:
        return set()
    cleaned = _NON_ALNUM.sub(" ", text.lower())
    out: set[str] = set()
    for raw in cleaned.split():
        tok = _SYNONYMS.get(raw, raw)
        if tok in _STOPWORDS or len(tok) <= 1:
            continue
        out.add(tok)
    return out


def similarity(a: str, b: str) -> float:
    """Token-set Jaccard: intersection over union of content-word sets.

    Empty-set handling is explicit: two empty sets are identical (1.0); one
    empty and one not is fully disjoint (0.0). Leaving 0/0 to fall through
    yields NaN and every downstream threshold comparison misbehaves silently.
    """
    wa, wb = words(a), words(b)
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    inter = len(wa & wb)
    return inter / (len(wa) + len(wb) - inter)
```

**Step 4: Run test to verify pass**

Run: `pytest tests/test_news_dedup.py -v`
Expected: PASS — 7 passed. Si `test_unrelated_sharing_topic_word_stays_below` falla por colisión de vocabulario, ajustar `_SYNONYMS`/`_STOPWORDS`, NO bajar el umbral.

**Step 5: Commit** (solo tras OK del usuario)

```bash
git add src/qtdata/news/dedup.py tests/test_news_dedup.py
git commit -m "feat(news): deterministic token-set Jaccard headline dedup primitives"
```

---

### Task 2: Función de clustering por (ticker, día)

**Objective:** Dado un grupo de filas que ya comparten (ticker, `date`), asignar un `event_id` a cada fila de forma que las copias sindicadas compartan el mismo `event_id`. Greedy single-pass O(k²) por grupo (k = nº artículos del día para ese ticker, típicamente <20).

**Files:**
- Modify: `src/qtdata/news/dedup.py`
- Test: `tests/test_news_dedup.py`

**Step 1: Write failing test**

```python
# añadir a tests/test_news_dedup.py
class TestAssignEventIds:
    def test_collapses_syndicated_titles_into_one_event(self):
        titles = [
            "Iran intercepts drone near Hormuz",
            "Iranian forces engage UAV over Strait of Hormuz",
            "Apple beats Q3 earnings estimates",
        ]
        ids = dedup.assign_event_ids(titles, threshold=0.5)
        assert ids[0] == ids[1]          # las dos del dron -> mismo evento
        assert ids[2] != ids[0]          # Apple -> evento distinto
        assert len(set(ids)) == 2

    def test_singletons_get_unique_ids(self):
        titles = ["Apple beats earnings", "Tesla output rises", "Nvidia upgrade"]
        ids = dedup.assign_event_ids(titles, threshold=0.5)
        assert len(set(ids)) == 3

    def test_empty_input(self):
        assert dedup.assign_event_ids([], threshold=0.5) == []
```

**Step 2: Run test to verify failure**

Run: `pytest tests/test_news_dedup.py::TestAssignEventIds -v`
Expected: FAIL — `AttributeError: module 'qtdata.news.dedup' has no attribute 'assign_event_ids'`

**Step 3: Write minimal implementation**

```python
# añadir a src/qtdata/news/dedup.py
def assign_event_ids(titles: list[str], threshold: float = 0.5) -> list[int]:
    """Greedy single-pass clustering of titles by Jaccard >= threshold.

    Returns a list of cluster ids (same length/order as `titles`); syndicated
    near-duplicates share an id. O(k^2) in the group size, which is bounded:
    callers pass one (ticker, trading-day) bucket at a time, so k is small.

    Pre-tokenizes each title once (the synonym map is the cost, not the loop).
    """
    n = len(titles)
    if n == 0:
        return []
    token_sets = [words(t) for t in titles]
    ids = [-1] * n
    next_id = 0
    for i in range(n):
        if ids[i] != -1:
            continue
        ids[i] = next_id
        wa = token_sets[i]
        for j in range(i + 1, n):
            if ids[j] != -1:
                continue
            wb = token_sets[j]
            if not wa and not wb:
                sim = 1.0
            elif not wa or not wb:
                sim = 0.0
            else:
                inter = len(wa & wb)
                sim = inter / (len(wa) + len(wb) - inter)
            if sim >= threshold:
                ids[j] = next_id
        next_id += 1
    return ids
```

**Step 4: Run test to verify pass**

Run: `pytest tests/test_news_dedup.py::TestAssignEventIds -v`
Expected: PASS — 3 passed.

**Step 5: Commit** (tras OK)

```bash
git add src/qtdata/news/dedup.py tests/test_news_dedup.py
git commit -m "feat(news): greedy per-bucket event clustering for syndicated dedup"
```

---

### Task 3: Settings opt-in (default conservador)

**Objective:** Exponer el dedup como configurable y APAGADO por defecto, para no alterar el factor existente sin decisión explícita.

**Files:**
- Modify: `src/qtdata/config.py:39-44` (bloque `# news / sentiment`)
- Test: `tests/test_config.py`

**Step 1: Write failing test**

```python
# añadir a tests/test_config.py
def test_news_dedup_defaults_off():
    from qtdata.config import Settings
    s = Settings()
    assert s.news_dedup_enabled is False
    assert s.news_dedup_threshold == 0.5
```

**Step 2: Run test to verify failure**

Run: `pytest tests/test_config.py::test_news_dedup_defaults_off -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'news_dedup_enabled'`

**Step 3: Write minimal implementation**

En `src/qtdata/config.py`, dentro de `# news / sentiment` (tras `finbert_revision`):

```python
    # news / sentiment — syndicated dedup (opt-in; default OFF preserves factor)
    news_dedup_enabled: bool = False
    news_dedup_threshold: float = 0.5
```

**Step 4: Run test to verify pass**

Run: `pytest tests/test_config.py::test_news_dedup_defaults_off -v`
Expected: PASS.

**Step 5: Commit** (tras OK)

```bash
git add src/qtdata/config.py tests/test_config.py
git commit -m "feat(config): news_dedup_enabled/threshold settings (default off)"
```

---

### Task 4: Integrar el colapso en `build_sentiment_daily`

**Objective:** Cuando `news_dedup_enabled`, colapsar filas sindicadas dentro de cada (ticker, `date`) ANTES de `_aggregate`: una copia representante por evento (la de mayor relevancia, desempate por `effective_ts` más temprano = la primera observada). Así `n_articles` cuenta eventos, no copias, y la media ponderada no cuenta dos veces el mismo hecho.

**Files:**
- Modify: `src/qtdata/news/aggregate.py` (entre el filtro de relevancia ~línea 126 y `_aggregate` ~línea 133)
- Test: `tests/test_news_aggregate.py`

**Step 1: Write failing test**

```python
# añadir a tests/test_news_aggregate.py
def test_dedup_collapses_syndicated_copies(settings, catalog, monkeypatch):
    # Mismo evento (dron Hormuz) en 3 outlets + 1 noticia distinta, todas AAPL.
    feed = [
        {"title": "Apple drone delivery trial near Hormuz hub", "url": "https://x/1",
         "time_published": "20260610T120000", "source": "s1", "summary": "",
         "overall_sentiment_score": 0.0,
         "ticker_sentiment": [{"ticker": "AAPL", "relevance_score": "0.9",
                               "ticker_sentiment_score": "0.6"}]},
        {"title": "Apple UAV delivery test over Strait of Hormuz", "url": "https://x/2",
         "time_published": "20260610T121500", "source": "s2", "summary": "",
         "overall_sentiment_score": 0.0,
         "ticker_sentiment": [{"ticker": "AAPL", "relevance_score": "0.8",
                               "ticker_sentiment_score": "0.6"}]},
        {"title": "Apple drones trialed near Hormuz", "url": "https://x/3",
         "time_published": "20260610T123000", "source": "s3", "summary": "",
         "overall_sentiment_score": 0.0,
         "ticker_sentiment": [{"ticker": "AAPL", "relevance_score": "0.7",
                               "ticker_sentiment_score": "0.6"}]},
        {"title": "Apple beats Q3 earnings estimates", "url": "https://x/4",
         "time_published": "20260610T130000", "source": "s4", "summary": "",
         "overall_sentiment_score": 0.0,
         "ticker_sentiment": [{"ticker": "AAPL", "relevance_score": "0.5",
                               "ticker_sentiment_score": "-0.2"}]},
    ]
    monkeypatch.setattr(
        "qtdata.providers.alpha_vantage_news.AlphaVantageNewsProvider.fetch_news_day",
        lambda self, day, page_limit: (parse_feed(feed), 1),
    )
    settings.news_dedup_enabled = True
    settings.news_dedup_threshold = 0.5
    ingest_news(settings, catalog, date_from=date(2026, 6, 10), date_to=date(2026, 6, 10))
    curate_news(settings, catalog)
    build_sentiment_daily(settings, catalog)

    daily = parquet_store.read(settings.curated_dir / "sentiment_daily")
    aapl = daily[daily["ticker"] == "AAPL"].iloc[0]
    # 3 copias sindicadas -> 1 evento; + la de earnings = 2 eventos, no 4
    assert aapl["n_articles"] == 2


def test_dedup_off_preserves_legacy_count(settings, catalog, monkeypatch):
    # Mismo feed, dedup OFF -> comportamiento legacy: cuenta las 4
    feed = [
        {"title": "Apple drone delivery trial near Hormuz hub", "url": "https://x/1",
         "time_published": "20260610T120000", "source": "s1", "summary": "",
         "overall_sentiment_score": 0.0,
         "ticker_sentiment": [{"ticker": "AAPL", "relevance_score": "0.9",
                               "ticker_sentiment_score": "0.6"}]},
        {"title": "Apple UAV delivery test over Strait of Hormuz", "url": "https://x/2",
         "time_published": "20260610T121500", "source": "s2", "summary": "",
         "overall_sentiment_score": 0.0,
         "ticker_sentiment": [{"ticker": "AAPL", "relevance_score": "0.8",
                               "ticker_sentiment_score": "0.6"}]},
    ]
    monkeypatch.setattr(
        "qtdata.providers.alpha_vantage_news.AlphaVantageNewsProvider.fetch_news_day",
        lambda self, day, page_limit: (parse_feed(feed), 1),
    )
    settings.news_dedup_enabled = False
    ingest_news(settings, catalog, date_from=date(2026, 6, 10), date_to=date(2026, 6, 10))
    curate_news(settings, catalog)
    build_sentiment_daily(settings, catalog)
    daily = parquet_store.read(settings.curated_dir / "sentiment_daily")
    aapl = daily[daily["ticker"] == "AAPL"].iloc[0]
    assert aapl["n_articles"] == 2  # ambas contadas (sin colapsar)
```

**Step 2: Run test to verify failure**

Run: `pytest tests/test_news_aggregate.py::test_dedup_collapses_syndicated_copies -v`
Expected: FAIL — `assert 4 == 2` (hoy cuenta las 4 copias).

**Step 3: Write minimal implementation**

En `src/qtdata/news/aggregate.py`, añadir import arriba:

```python
from qtdata.news import dedup
```

Y tras `rows["weight"] = weight[rows.index]` (justo antes del bloque `if since is not None:`), insertar:

```python
    # Optional: collapse syndicated duplicates within each (ticker, day) bucket
    # BEFORE aggregation. The bucket itself is the temporal guard the dedup TIL
    # recommends; the ticker is the source-diversity guard. We keep ONE
    # representative per event (highest relevance; tie -> earliest effective_ts =
    # first observed) so n_articles counts events and the weighted mean does not
    # double-count one fact reported by N outlets. The curated layer is untouched.
    if settings.news_dedup_enabled:
        threshold = settings.news_dedup_threshold
        titles_by_id = {}
        _arts = parquet_store.read(
            settings.curated_dir / "news_articles", columns=["article_id", "title"]
        )
        if not _arts.empty:
            titles_by_id = dict(zip(_arts["article_id"], _arts["title"], strict=True))

        keep_idx: list[int] = []
        for (_tkr, _day), grp in rows.groupby(["ticker", "date"], sort=False):
            if len(grp) == 1:
                keep_idx.extend(grp.index.tolist())
                continue
            titles = [titles_by_id.get(aid, "") for aid in grp["article_id"]]
            event_ids = dedup.assign_event_ids(titles, threshold=threshold)
            g = grp.assign(_event=event_ids)
            # representante por evento: max relevancia, desempate effective_ts asc
            g = g.sort_values(
                ["_event", "weight", "effective_ts"], ascending=[True, False, True]
            )
            reps = g.drop_duplicates(subset="_event", keep="first")
            keep_idx.extend(reps.index.tolist())
        rows = rows.loc[sorted(keep_idx)].copy()
```

**Step 4: Run test to verify pass**

Run: `pytest tests/test_news_aggregate.py -v`
Expected: PASS — incluyendo los dos tests nuevos y TODOS los legacy (idempotencia, weighted mean, cutoffs) sin cambios, porque el default es OFF.

**Step 5: Commit** (tras OK)

```bash
git add src/qtdata/news/aggregate.py tests/test_news_aggregate.py
git commit -m "feat(news): collapse syndicated duplicates per (ticker,day) before factor build"
```

---

### Task 5: Flag CLI en `news build-factor`

**Objective:** Permitir activar el dedup desde la línea de comandos sin editar `.env`, manteniendo el default del repo.

**Files:**
- Modify: `src/qtdata/cli.py:244-257` (`news_build_factor`)
- Test: `tests/test_cli_news.py`

**Step 1: Write failing test**

```python
# añadir a tests/test_cli_news.py (calcar el estilo de invocación CliRunner existente)
def test_build_factor_dedup_flag_overrides_setting(monkeypatch):
    captured = {}
    def _fake_build(settings, cat, since=None):
        captured["enabled"] = settings.news_dedup_enabled
        return 0
    monkeypatch.setattr("qtdata.news.aggregate.build_sentiment_daily", _fake_build)
    from typer.testing import CliRunner
    from qtdata.cli import app
    result = CliRunner().invoke(app, ["news", "build-factor", "--dedup"])
    assert result.exit_code == 0
    assert captured["enabled"] is True
```

**Step 2: Run test to verify failure**

Run: `pytest tests/test_cli_news.py::test_build_factor_dedup_flag_overrides_setting -v`
Expected: FAIL — `No such option: --dedup`.

**Step 3: Write minimal implementation**

Reemplazar el cuerpo de `news_build_factor` en `src/qtdata/cli.py`:

```python
@news_app.command("build-factor")
def news_build_factor(
    since: str = typer.Option(None, help="Only rebuild (ticker, day) rows from this ISO date"),
    dedup: bool = typer.Option(
        None, "--dedup/--no-dedup",
        help="Collapse syndicated duplicate headlines (overrides QT_NEWS_DEDUP_ENABLED)",
    ),
) -> None:
    """(Re)build the daily sentiment factor `sentiment_daily`."""
    from qtdata.news.aggregate import build_sentiment_daily

    settings = get_settings()
    if dedup is not None:
        settings = settings.model_copy(update={"news_dedup_enabled": dedup})
    with Catalog(settings) as cat:
        cat.init_schema()
        n = build_sentiment_daily(
            settings, cat, since=date.fromisoformat(since) if since else None
        )
    console.print(f"[green]sentiment_daily: {n} (ticker, day) rows upserted.[/green]")
```

**Step 4: Run test to verify pass**

Run: `pytest tests/test_cli_news.py -v`
Expected: PASS.

**Step 5: Commit** (tras OK)

```bash
git add src/qtdata/cli.py tests/test_cli_news.py
git commit -m "feat(cli): news build-factor --dedup/--no-dedup flag"
```

---

### Task 6: Suite completa + verificación en subset real (con OK)

**Objective:** Confirmar que nada legacy se rompe y medir el impacto del dedup en datos reales antes de adoptarlo.

**Step 1: Run full suite**

Run: `pytest -q`
Expected: todo verde; el factor por defecto idéntico al de antes (dedup OFF).

**Step 2 (requiere OK explícito — lee `data/`, no escribe):** comparar n_articles con/sin dedup en un subset

```bash
# en un curated_dir de prueba / copia, NUNCA en producción sin OK
qt news build-factor --no-dedup     # baseline
# inspeccionar sentiment_daily.n_articles
qt news build-factor --dedup --since 2026-06-01
# diff de distribución de n_articles y de sent_finbert por (ticker, día)
```

**Verificación cualitativa (del TIL):**
- Alimentar pares sindicados conocidos (mismo evento, distinto medio) y confirmar que superan el umbral.
- Alimentar titulares no relacionados que solo comparten una palabra-tema y confirmar que quedan por debajo.
- Vigilar over-merging: dos eventos distintos que reducen al mismo conjunto canónico colisionarán (no hay noción de entidades nombradas ni fechas). El bucket (ticker, día) ya acota esto; si aún hay falsos merges, subir `news_dedup_threshold` antes que tocar el código.

**Step 3: Decisión de adopción**

Mantener `news_dedup_enabled=False` como default del repo; activarlo por `.env`/flag solo tras validar el impacto. Documentar en el README de la capa news el tradeoff (determinismo/explicabilidad vs. mantenimiento del mapa de sinónimos).

---

## Notas de mantenimiento

- El `_SYNONYMS` es deuda viva y específica de dominio (finance-news). Ampliarlo cuando aparezcan falsos negativos (duplicados que no se fusionan), no cuando aparezcan falsos positivos (para esos, subir umbral o añadir guarda).
- Si en el futuro el volumen de noticias por (ticker, día) crece mucho (k grande), el O(k²) puede optimizarse con blocking por primer token canónico; YAGNI hoy.
- El dedup vive en el factor gold, NO en curated: si algún día quieres trazabilidad del clustering, materializa `event_id` en una vista derivada, nunca reescribas `news_ticker_sentiment`.
