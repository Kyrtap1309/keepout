"""Unit constants shared across keepout."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from astropy.time import Time

__all__ = ["DAY", "KM", "MINUTE", "seconds_between"]

KM: Final = 1e3
"""Metres in a kilometre."""

MINUTE: Final = 60.0
"""Seconds in a minute."""

DAY: Final = 86400.0
"""Seconds in a day."""


def seconds_between(start: Time, end: Time) -> float:
    """Seconds from ``start`` to ``end``, from their UTC Julian days."""
    return float((end.jd1 - start.jd1) * DAY + (end.jd2 - start.jd2) * DAY)
