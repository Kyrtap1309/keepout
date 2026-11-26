"""Words for messages and notes, shared by the library, the command line
and the dashboard."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from keepout.units import DAY, HOUR, MINUTE

if TYPE_CHECKING:
    from astropy.time import Time

    from keepout.risk import PcObservation

__all__ = [
    "ASSUMED_COVARIANCES",
    "PC_FAILED",
    "before_tca",
    "days_text",
    "duration_words",
    "isot",
    "minute",
    "pc_cell",
    "pc_text",
    "plural",
    "utc",
]

ASSUMED_COVARIANCES: Final = (
    "Covariances are assumed (keepout's model of GP element set errors), "
    "not orbit determination."
)
"""The line under a Pc on the synthetic covariance model."""


def before_tca(seconds: float) -> str:
    """A time before TCA in hours: ``6.0 h before TCA``."""
    return f"{seconds / HOUR:.1f} h before TCA"


def days_text(seconds: float) -> str:
    """A duration in days, as keepout prints it: ``1.50 d``."""
    return f"{seconds / DAY:.2f} d"


def duration_words(seconds: float) -> str:
    """A whole number of the largest unit of days, hours and minutes that
    ``seconds`` holds, in words: ``a day``, ``an hour``, ``10 minutes``
    (else seconds, ``90 s``)."""
    for unit, noun, one in ((DAY, "day", "a day"), (HOUR, "hour", "an hour")):
        count = seconds / unit
        if count >= 1.0 and count.is_integer():
            return one if count == 1.0 else f"{count:g} {noun}s"
    count = seconds / MINUTE
    if count >= 1.0 and count.is_integer():
        return "a minute" if count == 1.0 else f"{count:g} minutes"
    return f"{seconds:g} s"


def pc_text(pc: float | None) -> str:
    """A Pc for tables, ``1.23e-05``, or ``-`` if there is none."""
    return "-" if pc is None else f"{pc:.2e}"


PC_FAILED: Final = "failed"
"""A Pc that could not be computed, in a table."""


def pc_cell(observation: PcObservation | None, *, missing: str) -> str:
    """The Pc of ``observation`` for a table: ``missing`` without one,
    :data:`PC_FAILED` if it could not be computed."""
    if observation is None:
        return missing
    if observation.pc is None:
        return PC_FAILED
    return pc_text(observation.pc)


def plural(count: int, noun: str, nouns: str | None = None) -> str:
    """``count`` and ``noun``, or its plural ``nouns`` (default: ``noun`` and
    ``s``) unless ``count`` is one: ``1 event``, ``2 events``."""
    if count == 1:
        return f"{count} {noun}"
    return f"{count} {nouns or noun + 's'}"


def isot(time: Time) -> str:
    """``time`` in UTC, ISO 8601 to the millisecond, for JSON:
    ``2026-10-06T19:36:42.123``."""
    return str(time.utc.isot)


def utc(time: Time) -> str:
    """``time`` in UTC to the second, ISO 8601: ``2026-10-06T19:36:42``."""
    return isot(time)[:19]


def minute(time: Time) -> str:
    """``time`` in UTC to the minute, for tables: ``2026-10-06 19:36``."""
    return utc(time)[:16].replace("T", " ")
