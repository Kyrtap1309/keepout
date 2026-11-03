"""The pacing of Space-Track requests below the published per-minute and
per-hour limits."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from typing import Final

from keepout.units import HOUR, MINUTE

__all__ = ["PER_HOUR", "PER_MINUTE", "Throttle"]

PER_MINUTE: Final = 20
"""Most requests sent in any 60 s; Space-Track allows fewer than 30."""

PER_HOUR: Final = 200
"""Most requests sent in any hour; Space-Track allows fewer than 300."""


class Throttle:
    """A pacing of requests: at most ``per_minute`` in any 60 s and
    ``per_hour`` in any hour, with a wait before a request that would make
    more.

    Parameters
    ----------
    per_minute, per_hour
        The limits.
    clock
        Gives the time [s] (a monotonic clock).
    sleep
        Waits the given seconds.

    Raises
    ------
    ValueError
        If a limit is below 1.
    """

    def __init__(
        self,
        *,
        per_minute: int = PER_MINUTE,
        per_hour: int = PER_HOUR,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], object] = time.sleep,
    ) -> None:
        if per_minute < 1 or per_hour < 1:
            raise ValueError(
                f"limits must be at least 1, got {per_minute}/min and {per_hour}/h"
            )
        self._limits = ((MINUTE, per_minute), (HOUR, per_hour))
        self._clock = clock
        self._sleep = sleep
        self._sent: deque[float] = deque(maxlen=max(per_minute, per_hour))

    def wait(self) -> None:
        """Wait until a request may be sent; it then counts as sent."""
        now = self._clock()
        due = now
        for window, limit in self._limits:
            if len(self._sent) >= limit:
                due = max(due, self._sent[-limit] + window)
        if due > now:
            self._sleep(due - now)
        self._sent.append(due)
