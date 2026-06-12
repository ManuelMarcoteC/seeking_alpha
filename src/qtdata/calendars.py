"""Trading-calendar helpers (single import point for exchange_calendars).

Everything downstream asks this module about sessions so a future swap to
pandas_market_calendars touches one file.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd

_CAL_START = "1990-01-01"

DateLike = str | date | pd.Timestamp


@lru_cache(maxsize=8)
def _calendar(name: str) -> xcals.ExchangeCalendar:
    return xcals.get_calendar(name, start=_CAL_START)


def _ts(d: DateLike) -> pd.Timestamp:
    return pd.Timestamp(d).normalize()


def sessions_between(start: DateLike, end: DateLike, calendar: str = "XNYS") -> pd.DatetimeIndex:
    """All exchange sessions in [start, end], clamped to the calendar's bounds."""
    cal = _calendar(calendar)
    s = max(_ts(start), cal.first_session)
    e = min(_ts(end), cal.last_session)
    if s > e:
        return pd.DatetimeIndex([])
    return cal.sessions_in_range(s, e)


def is_session(d: DateLike, calendar: str = "XNYS") -> bool:
    cal = _calendar(calendar)
    t = _ts(d)
    if t < cal.first_session or t > cal.last_session:
        return False
    return bool(cal.is_session(t))


def next_session(d: DateLike, calendar: str = "XNYS") -> pd.Timestamp:
    """First session strictly after d. Raises if beyond the calendar's horizon."""
    cal = _calendar(calendar)
    return cal.date_to_session(_ts(d) + pd.Timedelta(days=1), direction="next")


def last_completed_session(today: DateLike, calendar: str = "XNYS") -> pd.Timestamp:
    """Most recent session strictly before `today`.

    Used as the default ingestion end: today's bar is partial while the market
    is open, and ingesting it would freeze a half-day print under the watermark.
    """
    cal = _calendar(calendar)
    return cal.date_to_session(_ts(today) - pd.Timedelta(days=1), direction="previous")
