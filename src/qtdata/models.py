"""Canonical datasets, column contracts and shared enums.

The curated layer stores UNADJUSTED prices plus a corporate-actions table;
adjusted series are always derived (see curation/adjustments.py). Validation
never mutates data: anomalies land in `validation_flags`, keyed back to rows.
"""

from __future__ import annotations

from enum import StrEnum


class Dataset(StrEnum):
    OHLCV_DAILY = "ohlcv_daily"
    CORPORATE_ACTIONS = "corporate_actions"


class ActionType(StrEnum):
    SPLIT = "split"
    DIVIDEND = "dividend"


class Severity(StrEnum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


# Primary keys of the curated tables
OHLCV_KEY = ["ticker", "date"]
ACTIONS_KEY = ["ticker", "ex_date", "action_type"]
FLAGS_KEY = ["ticker", "date", "flag_type"]
UNIVERSE_KEY = ["index_name", "ticker", "effective_from"]
FACTORS_KEY = ["ticker", "date"]

# Canonical column order
OHLCV_COLUMNS = [
    "ticker", "date", "open", "high", "low", "close", "volume",
    "source", "run_id", "ingested_at",
]
ACTIONS_COLUMNS = [
    "ticker", "ex_date", "action_type", "value",
    "source", "run_id", "ingested_at",
]
FLAGS_COLUMNS = ["ticker", "date", "flag_type", "severity", "detail", "run_id", "flagged_at"]

LINEAGE_COLUMNS = ["source", "run_id", "ingested_at"]


class ProviderNotConfiguredError(RuntimeError):
    """Raised when a provider is used without its required credentials."""
