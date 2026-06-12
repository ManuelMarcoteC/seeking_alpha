"""Cross-sectional Spearman information coefficient of a daily factor.

Daily IC = Spearman rank correlation between the factor score and the forward
return across the names scored that day (>= min_breadth, else the day is
skipped — a 3-name "cross-section" is noise, not a signal estimate). The
t-stat assumes i.i.d. daily ICs; with overlapping forward windows (h > 1) it
is over-optimistic — reported as-is and flagged in the report caveats.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

NAN = float("nan")


@dataclass
class ICSummary:
    horizon: int
    n_days: int
    mean_ic: float
    std_ic: float
    t_stat: float
    icir: float
    hit_rate: float


def daily_ic(
    factor: pd.DataFrame,
    fwd: pd.DataFrame,
    *,
    score_col: str,
    fwd_col: str,
    min_breadth: int = 10,
) -> pd.DataFrame:
    """One row per qualifying day: [date, n, ic]."""
    merged = factor.merge(fwd, on=["ticker", "date"], how="inner").dropna(
        subset=[score_col, fwd_col]
    )
    rows = []
    for day, group in merged.groupby("date"):
        if len(group) < min_breadth:
            continue
        if group[score_col].nunique() <= 1 or group[fwd_col].nunique() <= 1:
            continue  # constant score or constant return: correlation undefined
        # Spearman = Pearson over average ranks (avoids the scipy dependency)
        ic = group[score_col].rank().corr(group[fwd_col].rank())
        if pd.isna(ic):
            continue
        rows.append({"date": day, "n": len(group), "ic": float(ic)})
    return pd.DataFrame(rows, columns=["date", "n", "ic"])


def summarize_ic(daily: pd.DataFrame, horizon: int) -> ICSummary:
    n = len(daily)
    if n == 0:
        return ICSummary(horizon, 0, NAN, NAN, NAN, NAN, NAN)
    mean = float(daily["ic"].mean())
    std = float(daily["ic"].std(ddof=1)) if n > 1 else NAN
    has_std = not math.isnan(std) and std > 0
    t_stat = mean / std * math.sqrt(n) if has_std else NAN
    icir = mean / std if has_std else NAN
    hit_rate = float((daily["ic"] > 0).mean())
    return ICSummary(horizon, n, mean, std, t_stat, icir, hit_rate)


def decay_profile(
    factor: pd.DataFrame,
    fwd: pd.DataFrame,
    *,
    score_col: str,
    horizons: tuple[int, ...],
    min_breadth: int = 10,
) -> list[ICSummary]:
    """Mean IC per horizon — the signal decay curve."""
    return [
        summarize_ic(
            daily_ic(
                factor, fwd, score_col=score_col, fwd_col=f"fwd_{h}d",
                min_breadth=min_breadth,
            ),
            h,
        )
        for h in horizons
    ]
