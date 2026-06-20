"""AI-Asia thematic basket analytics (research-only, read-only).

Characterizes a small curated basket of Asian AI/semiconductor names
(SEHK/KRX) that the NASDAQ-first pipeline does not serve through its usual
factor machinery. This is DESCRIPTIVE research, never a signal (invariant 6):
it reports short-window momentum, intra-basket relative strength, and a
sentiment overlay, with EXPLICIT N/A where a name's history is too short
(e.g. a recent IPO like 0100.HK MiniMax with a handful of sessions).

Hard rules honoured:
- Short-history guard: a metric whose lookback exceeds a name's observed
  sessions returns None, never a number computed on too few bars (no SMA200
  on 107 bars presented as valid).
- Cross-currency: price LEVELS are not comparable across markets (HKD vs KRW
  vs USD); everything reported is a return, ratio, or oscillator — never a
  raw level comparison.
- Read-only: callers pass a Catalog opened read-only; this module never writes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from qtdata.storage.catalog import Catalog

# Default seed basket (curated). Suffixes drive market resolution downstream.
DEFAULT_BASKET: tuple[str, ...] = (
    "2513.HK",    # Zhipu / Knowledge Atlas (China AI LLM)
    "0100.HK",    # MiniMax Group (China AI LLM, recent IPO)
    "005930.KS",  # Samsung Electronics (semis/memory)
    "000660.KS",  # SK Hynix (HBM memory for AI)
    "035720.KS",  # Kakao (Korea tech/AI)
    "247540.KQ",  # Ecopro BM (batteries — non-AI control)
)


@dataclass
class TickerMetrics:
    """Short-window characterization for one basket member. None = N/A (too short)."""

    ticker: str
    sessions: int
    last_close: float
    ret_5d: float | None
    ret_20d: float | None
    vs_sma20: float | None
    vs_sma50: float | None
    rsi_14: float | None
    rel_strength: float | None  # last close / basket-median-normalized level
    sentiment: float | None
    n_news: int


def load_basket_closes(catalog: Catalog, basket: tuple[str, ...]) -> pd.DataFrame:
    """[ticker, date, close_raw] for the basket, sorted. close_raw = the clean
    split-adjusted vendor series (see fix #2 rationale). Read-only query."""
    in_list = ", ".join(f"'{t}'" for t in basket)
    sql = (
        f"SELECT ticker, date, close_raw FROM ohlcv_daily_adj "  # noqa: S608 - fixed basket
        f"WHERE ticker IN ({in_list}) ORDER BY ticker, date"
    )
    df = catalog.query(sql)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def load_basket_sentiment(catalog: Catalog, basket: tuple[str, ...]) -> pd.DataFrame:
    """Latest sentiment per ticker from sentiment_daily, if present."""
    in_list = ", ".join(f"'{t}'" for t in basket)
    sql = (
        f"SELECT ticker, date, sent_finbert, n_articles FROM sentiment_daily "  # noqa: S608
        f"WHERE ticker IN ({in_list}) ORDER BY ticker, date"
    )
    try:
        df = catalog.query(sql)
    except Exception:  # noqa: BLE001 - sentiment table may not exist yet
        return pd.DataFrame(columns=["ticker", "date", "sent_finbert", "n_articles"])
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _rsi(closes: np.ndarray, period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    d = np.diff(closes)
    up = np.clip(d, 0, None)[-period:].mean()
    dn = -np.clip(d, None, 0)[-period:].mean()
    if dn == 0:
        return 100.0
    return float(100.0 - 100.0 / (1.0 + up / dn))


def _ret(closes: np.ndarray, k: int) -> float | None:
    if len(closes) <= k:
        return None
    return float(closes[-1] / closes[-1 - k] - 1.0)


def _vs_sma(closes: np.ndarray, window: int) -> float | None:
    if len(closes) < window:
        return None
    return float(closes[-1] / closes[-window:].mean() - 1.0)


def compute_ticker_metrics(
    ticker: str,
    closes: np.ndarray,
    sentiment: float | None,
    n_news: int,
    rel_strength: float | None,
) -> TickerMetrics:
    """Pure metric computation for one ticker with the short-history guard."""
    return TickerMetrics(
        ticker=ticker,
        sessions=len(closes),
        last_close=float(closes[-1]) if len(closes) else float("nan"),
        ret_5d=_ret(closes, 5),
        ret_20d=_ret(closes, 20),
        vs_sma20=_vs_sma(closes, 20),
        vs_sma50=_vs_sma(closes, 50),
        rsi_14=_rsi(closes, 14),
        rel_strength=rel_strength,
        sentiment=sentiment,
        n_news=n_news,
    )


@dataclass
class BasketReport:
    metrics: list[TickerMetrics] = field(default_factory=list)
    as_of: pd.Timestamp | None = None


def _relative_strength(per_ticker_last_ret20: dict[str, float | None]) -> dict[str, float | None]:
    """Rank-free relative strength: each name's 20d return minus the basket median
    20d return (positive = outperforming the basket). None where 20d undefined.

    Using returns (not price levels) keeps it currency-agnostic across HKD/KRW.
    """
    vals = [v for v in per_ticker_last_ret20.values() if v is not None]
    if not vals:
        return dict.fromkeys(per_ticker_last_ret20, None)
    median = float(np.median(vals))
    return {
        t: (None if v is None else float(v - median))
        for t, v in per_ticker_last_ret20.items()
    }


def build_basket_report(
    catalog: Catalog, basket: tuple[str, ...] = DEFAULT_BASKET
) -> BasketReport:
    """Assemble the full descriptive report for the basket. Read-only."""
    closes_df = load_basket_closes(catalog, basket)
    sent_df = load_basket_sentiment(catalog, basket)

    # latest sentiment per ticker
    sent_latest: dict[str, tuple[float | None, int]] = {}
    if not sent_df.empty:
        for tk, g in sent_df.groupby("ticker"):
            row = g.sort_values("date").iloc[-1]
            val = row["sent_finbert"]
            sent_latest[str(tk)] = (
                None if pd.isna(val) else float(val),
                int(row["n_articles"]),
            )

    # per-ticker close arrays + 20d returns (for relative strength)
    closes_by_ticker: dict[str, np.ndarray] = {}
    ret20_by_ticker: dict[str, float | None] = {}
    for tk in basket:
        c = closes_df.loc[closes_df["ticker"] == tk, "close_raw"].to_numpy(dtype=float)
        closes_by_ticker[tk] = c
        ret20_by_ticker[tk] = _ret(c, 20)

    rel = _relative_strength(ret20_by_ticker)

    metrics = []
    for tk in basket:
        c = closes_by_ticker[tk]
        if len(c) == 0:
            continue  # not in lake
        sent, n_news = sent_latest.get(tk, (None, 0))
        metrics.append(
            compute_ticker_metrics(tk, c, sent, n_news, rel[tk])
        )

    as_of = closes_df["date"].max() if not closes_df.empty else None
    return BasketReport(metrics=metrics, as_of=as_of)


def _fmt_pct(v: float | None) -> str:
    return "   N/A" if v is None else f"{v * 100:+.1f}%"


def _fmt_num(v: float | None, digits: int = 0) -> str:
    return "N/A" if v is None else f"{v:.{digits}f}"


def render_basket_table(report: BasketReport) -> str:
    """Terminal-friendly table (headless). Descriptive only, not a signal."""
    lines = []
    as_of = report.as_of.date() if report.as_of is not None else "?"
    lines.append(f"=== AI-Asia basket | as of {as_of} | close_raw (clean split-adj) ===")
    lines.append(
        f"{'ticker':10} {'ses':>4} {'last':>10} {'ret5':>7} {'ret20':>7} "
        f"{'vSMA20':>7} {'vSMA50':>7} {'RSI':>4} {'relStr':>7} {'sent':>6} {'news':>4}"
    )
    for m in report.metrics:
        lines.append(
            f"{m.ticker:10} {m.sessions:>4} {m.last_close:>10.1f} "
            f"{_fmt_pct(m.ret_5d):>7} {_fmt_pct(m.ret_20d):>7} "
            f"{_fmt_pct(m.vs_sma20):>7} {_fmt_pct(m.vs_sma50):>7} "
            f"{_fmt_num(m.rsi_14):>4} {_fmt_pct(m.rel_strength):>7} "
            f"{_fmt_num(m.sentiment, 2):>6} {m.n_news:>4}"
        )
    lines.append("")
    lines.append("N/A = insufficient history. Levels NOT comparable across currencies")
    lines.append("(HKD/KRW); use returns/ratios. relStr = 20d return minus basket median.")
    lines.append("Descriptive characterization, NOT a buy/sell signal (invariant 6).")
    return "\n".join(lines)
