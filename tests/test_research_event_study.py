import pandas as pd
import pytest

from qtdata.research.event_study import find_events, run_event_study

DATES = pd.date_range("2026-01-05", periods=15, freq="B")


def _flat_universe_with_jump() -> pd.DataFrame:
    """6 flat tickers; F jumps +5% the session AFTER the event (offset +1)."""
    frames = []
    for name in ("A", "B", "C", "D", "E"):
        frames.append(pd.DataFrame({"ticker": name, "date": DATES, "close": 100.0}))
    f_close = [100.0] * 8 + [105.0] * 7  # event at index 7 -> jump at index 8
    frames.append(pd.DataFrame({"ticker": "F", "date": DATES, "close": f_close}))
    return pd.concat(frames, ignore_index=True)


def test_find_events_thresholds():
    factor = pd.DataFrame(
        {
            "ticker": ["F", "G", "H", "I"],
            "date": [DATES[7]] * 4,
            "score": [0.8, -0.9, 0.3, 0.9],  # H below threshold
            "n_articles": [5, 5, 5, 1],  # I lacks article support
        }
    )
    events = find_events(factor, score_col="score", threshold=0.5, min_articles=3)
    assert set(events["ticker"]) == {"F", "G"}
    assert events.set_index("ticker")["sign"].to_dict() == {"F": 1, "G": -1}


def test_event_study_car_shape_and_magnitude():
    closes = _flat_universe_with_jump()
    events = pd.DataFrame({"ticker": ["F"], "date": [DATES[7]], "sign": [1]})
    result = run_event_study(closes, events, window=(-2, 3))

    assert result.n_pos == 1
    assert result.n_neg == 0
    assert list(result.car_pos.index) == [-2, -1, 0, 1, 2, 3]
    # flat before the event: CAR 0 through offset 0
    assert result.car_pos.loc[0] == pytest.approx(0.0)
    # +5% raw at offset +1, market-adjusted by the equal-weighted mean (6 names)
    expected = 0.05 - 0.05 / 6
    assert result.car_pos.loc[1] == pytest.approx(expected)
    # nothing after: CAR flat at the jump level... almost — the 5 flat names get
    # -mkt at offset +1 only for themselves; F's own CAR carries unchanged
    assert result.car_pos.loc[3] == pytest.approx(expected)


def test_incomplete_windows_are_dropped():
    closes = _flat_universe_with_jump()
    # event too close to the start for a (-5, +3) window
    events = pd.DataFrame({"ticker": ["F"], "date": [DATES[2]], "sign": [1]})
    result = run_event_study(closes, events, window=(-5, 3))
    assert result.n_pos == 0
    assert result.car_pos.empty
