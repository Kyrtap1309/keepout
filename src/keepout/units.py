"""Unit constants: keepout works in metres and seconds, and these convert."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from astropy.time import Time

__all__ = ["DAY", "HOUR", "KM", "MINUTE", "elapsed_seconds", "julian_seconds_between"]

KM: Final = 1e3
"""Metres in a kilometre."""

MINUTE: Final = 60.0
"""Seconds in a minute."""

HOUR: Final = 3600.0
"""Seconds in an hour."""

DAY: Final = 86400.0
"""Seconds in a day."""


def elapsed_seconds(start: Time, end: Time) -> float:
    """SI seconds elapsed from ``start`` to ``end``, leap seconds included."""
    return float((end - start).to_value("s"))


def julian_seconds_between(start: Time, end: Time) -> float:
    """Seconds from ``start`` to ``end`` counted as 86400 s per day of their
    two-part Julian dates, in the Times' own scale.

    Fast and exact for Times in one scale, but across a UTC leap second it
    differs from the elapsed SI seconds (:func:`elapsed_seconds`) by that
    second.
    """
    return float((end.jd1 - start.jd1) * DAY + (end.jd2 - start.jd2) * DAY)
