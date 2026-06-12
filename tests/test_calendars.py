import pandas as pd

from qtdata import calendars


def test_july_4th_is_not_a_session():
    assert not calendars.is_session("2024-07-04")
    assert calendars.is_session("2024-07-03")


def test_weekend_is_not_a_session():
    assert not calendars.is_session("2024-07-06")  # Saturday


def test_sessions_between_known_week():
    # Week of 2024-07-01: Mon, Tue, Wed (half day), Fri — Thu is July 4th
    sessions = calendars.sessions_between("2024-07-01", "2024-07-07")
    assert list(sessions) == [
        pd.Timestamp("2024-07-01"),
        pd.Timestamp("2024-07-02"),
        pd.Timestamp("2024-07-03"),
        pd.Timestamp("2024-07-05"),
    ]


def test_next_session_skips_weekend_and_holiday():
    assert calendars.next_session("2024-07-03") == pd.Timestamp("2024-07-05")
    assert calendars.next_session("2024-07-05") == pd.Timestamp("2024-07-08")


def test_sessions_between_empty_when_inverted():
    assert len(calendars.sessions_between("2024-07-07", "2024-07-01")) == 0


def test_last_completed_session_excludes_today():
    # Friday 2024-07-05 was a session — but "today" itself is never complete
    assert calendars.last_completed_session("2024-07-05") == pd.Timestamp("2024-07-03")
    # Monday after a holiday weekend reaches back to Friday
    assert calendars.last_completed_session("2024-07-08") == pd.Timestamp("2024-07-05")
