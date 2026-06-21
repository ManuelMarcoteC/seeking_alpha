"""Make SIGTERM a catchable unwind so finally/cleanup runs on container stop.

Two modes, composable:
- As a context manager, SIGTERM raises SystemExit(143) (128 + SIGTERM). The
  stack unwinds, `finally` blocks run, DuckDB connections close, partial
  summaries surface. The previous handler is restored on exit (never leaked).
- For cooperative draining, `guard.should_stop` flips True on first signal so a
  loop can finish the current unit and break BEFORE the SystemExit fires (set
  `raise_on_signal=False` to only flip the flag and not raise).

Ported from the SOFA TIL "S3 resume-state object only appears when the process
is killed" — CPython's SIGINT/SIGTERM asymmetry, not specific to any framework.
Ctrl+C (SIGINT) raises KeyboardInterrupt and runs `finally`; the default SIGTERM
disposition terminates the process without raising, so manual Ctrl+C testing
validates a path that never runs under docker/k8s container stops.
"""
from __future__ import annotations

import logging
import signal
from contextlib import contextmanager
from types import FrameType

logger = logging.getLogger(__name__)

EXIT_SIGTERM = 143  # 128 + 15


class _Guard:
    def __init__(self, raise_on_signal: bool) -> None:
        self.should_stop = False
        self._raise = raise_on_signal

    def _request_stop(self, signum: int, frame: FrameType | None) -> None:
        self.should_stop = True
        logger.warning(
            "SIGTERM received: draining; will exit %d after cleanup", EXIT_SIGTERM
        )
        if self._raise:
            raise SystemExit(EXIT_SIGTERM)


@contextmanager
def terminable(raise_on_signal: bool = True):
    """Install a SIGTERM->SystemExit(143) handler for the duration of the block.

    Restores the previous handler on exit (never leaked). Yields a guard whose
    `should_stop` becomes True when SIGTERM is delivered, enabling cooperative
    loop draining. With `raise_on_signal=False` the handler only flips the flag
    and does not raise, so the loop drains on its own terms.
    """
    guard = _Guard(raise_on_signal=raise_on_signal)
    previous = signal.signal(signal.SIGTERM, guard._request_stop)
    try:
        yield guard
    finally:
        signal.signal(signal.SIGTERM, previous)  # don't leak the handler
