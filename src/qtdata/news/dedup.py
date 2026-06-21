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
    # demonym -> country canonicalization (entity identity across syndication)
    "iranian": "iran",
    # corporate-action / market vocabulary canonicalization (extend as needed)
    "drone": "uav", "drones": "uav", "uavs": "uav",
    "intercept": "engage", "intercepts": "engage", "intercepted": "engage",
    "engages": "engage", "engaged": "engage",
    "shares": "stock", "share": "stock", "equity": "stock", "equities": "stock",
    "rises": "rise", "rose": "rise", "gains": "rise", "gained": "rise",
    "jumps": "rise", "jumped": "rise", "climbs": "rise", "climbed": "rise",
    "falls": "fall", "fell": "fall", "drops": "fall", "dropped": "fall",
    "slumps": "fall", "slumped": "fall", "slides": "fall", "slid": "fall",
    "beats": "beat", "tops": "beat", "misses": "miss", "missed": "miss",
    "acquires": "acquire", "acquired": "acquire", "buys": "acquire",
    "merger": "acquire", "merges": "acquire",
    "lawsuit": "sue", "sues": "sue", "sued": "sue",
    "profit": "earnings", "profits": "earnings",
    "sales": "revenue",
    # NOTE: deliberately NOT canonicalizing polysemous verbs like "cut"/"raised"
    # — in finance news they appear in non-rating senses ("Fed raised rates",
    # "OPEC cuts output") and would inject misleading canonical tokens.
    "upgrades": "upgrade",
    "downgrades": "downgrade",
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


def _jaccard(wa: set[str], wb: set[str]) -> float:
    """Jaccard over two pre-tokenized sets, with explicit empty-set handling.

    Two empty sets are identical (1.0); one empty and one not is fully disjoint
    (0.0). Leaving 0/0 to fall through yields NaN and every downstream threshold
    comparison misbehaves silently. Single source of truth for both the
    pairwise `similarity()` and the per-bucket `assign_event_ids()` clustering.
    """
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    inter = len(wa & wb)
    return inter / (len(wa) + len(wb) - inter)


def similarity(a: str, b: str) -> float:
    """Token-set Jaccard: intersection over union of content-word sets."""
    return _jaccard(words(a), words(b))


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
            if _jaccard(wa, token_sets[j]) >= threshold:
                ids[j] = next_id
        next_id += 1
    return ids
