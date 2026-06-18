"""Verify reconstruct_as_traded recovers AAPL's true as-traded prices.

Re-runnable probe (cited in the qtdata-pipeline skill). Hits yfinance live, does
NOT write the lake. Reproduces AAPL's 4:1 split (2020-08-31) and asserts the
provider's un-adjust gives back the real tape price.

Run: ./venv/bin/python scripts/verify_as_traded_unadjust.py
"""
from __future__ import annotations

import sys

import pandas as pd
import yfinance as yf

from qtdata.providers.yfinance_provider import (
    normalize_history_actions,
    normalize_history_ohlcv,
    reconstruct_as_traded,
)

# Known truth (as-traded, from the historical tape):
#   AAPL 2020-08-27 close ~499 USD, volume ~38.8M  (pre 4:1 split)
#   AAPL 2020-08-31 (ex-date) is already post-split: close ~129, must NOT change
EXPECTED_2708_CLOSE = 500.04
EXPECTED_2708_VOL = 38_888_100
TOL_CLOSE = 1.0
TOL_VOL_FRAC = 0.001


def main() -> int:
    hist = yf.Ticker("AAPL").history(
        start="2020-08-20", end="2020-09-05", interval="1d",
        auto_adjust=False, actions=True,
    )
    ohlcv = normalize_history_ohlcv(hist, "AAPL")
    actions = normalize_history_actions(hist, "AAPL")
    out = reconstruct_as_traded(ohlcv, actions).set_index("date")

    d2708 = pd.Timestamp("2020-08-27")
    d3108 = pd.Timestamp("2020-08-31")
    vendor = ohlcv.set_index("date")

    close_2708 = float(out.loc[d2708, "close"])
    vol_2708 = int(out.loc[d2708, "volume"])
    close_3108 = float(out.loc[d3108, "close"])
    vendor_3108 = float(vendor.loc[d3108, "close"])

    split_rows = actions[actions["action_type"] == "split"]
    listed = [
        (d.date(), float(v))
        for d, v in zip(split_rows["ex_date"], split_rows["value"], strict=True)
    ]
    print(f"split rows in frame: {listed}")
    print(f"2020-08-27 close: {close_2708:.2f}  (expect ~{EXPECTED_2708_CLOSE})")
    print(f"2020-08-27 vol:   {vol_2708:,}  (expect ~{EXPECTED_2708_VOL:,})")
    print(
        f"2020-08-31 close: {close_3108:.2f}  vendor {vendor_3108:.2f} "
        f"(must be EQUAL: ex-date untouched)"
    )

    ok_close = abs(close_2708 - EXPECTED_2708_CLOSE) < TOL_CLOSE
    ok_vol = abs(vol_2708 - EXPECTED_2708_VOL) / EXPECTED_2708_VOL < TOL_VOL_FRAC
    ok_exdate = abs(close_3108 - vendor_3108) < 1e-6

    print()
    print(f"[{'PASS' if ok_close else 'FAIL'}] pre-split close recovered to as-traded")
    print(f"[{'PASS' if ok_vol else 'FAIL'}] pre-split volume recovered to as-traded")
    print(f"[{'PASS' if ok_exdate else 'FAIL'}] ex-date row untouched (no spurious x4)")

    if ok_close and ok_vol and ok_exdate:
        print("\nALL CHECKS PASS — reconstruct_as_traded is correct on AAPL 4:1")
        return 0
    print("\nFAILED — do NOT proceed with the lake backfill")
    return 1


if __name__ == "__main__":
    sys.exit(main())
