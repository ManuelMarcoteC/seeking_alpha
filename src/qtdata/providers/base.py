"""Provider contract: FetchResult, DataProviderProtocol, rate limiting and retries."""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Protocol, runtime_checkable

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from qtdata.models import Dataset


def payload_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FetchResult:
    df: pd.DataFrame
    provider: str
    dataset: Dataset
    ticker: str
    requested_start: date
    requested_end: date
    fetched_at: datetime
    payload_sha256: str


def make_fetch_result(
    df: pd.DataFrame, provider: str, dataset: Dataset, ticker: str, start: date, end: date
) -> FetchResult:
    return FetchResult(
        df=df,
        provider=provider,
        dataset=dataset,
        ticker=ticker,
        requested_start=start,
        requested_end=end,
        fetched_at=datetime.now(UTC),
        payload_sha256=payload_hash(df),
    )


@runtime_checkable
class DataProviderProtocol(Protocol):
    name: str

    def fetch_ohlcv(self, ticker: str, start: date, end: date) -> FetchResult: ...

    def fetch_corporate_actions(self, ticker: str, start: date, end: date) -> FetchResult: ...

    def supported_datasets(self) -> frozenset[Dataset]: ...


class RateLimiter:
    """Enforces a minimum spacing between calls (token-interval limiter)."""

    def __init__(self, calls_per_min: int):
        self._interval = 60.0 / max(calls_per_min, 1)
        self._lock = threading.Lock()
        self._last = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._last + self._interval - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._last = now


TRANSIENT_ERRORS = (requests.ConnectionError, requests.Timeout, ConnectionError, TimeoutError)

# Retry transient network failures only; logical errors (bad ticker, bad key) surface immediately.
retry_transient = retry(
    retry=retry_if_exception_type(TRANSIENT_ERRORS),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
