"""Words for messages and notes, shared by the library and the command line."""

from __future__ import annotations

from keepout.units import DAY, HOUR

__all__ = ["before_tca", "days_text", "pc_text", "plural"]


def before_tca(seconds: float) -> str:
    """A time before TCA in hours: ``6.0 h before TCA``."""
    return f"{seconds / HOUR:.1f} h before TCA"


def days_text(seconds: float) -> str:
    """A duration in days, as keepout prints it: ``1.50 d``."""
    return f"{seconds / DAY:.2f} d"


def pc_text(pc: float | None) -> str:
    """A Pc for tables, ``1.23e-05``, or ``-`` if there is none."""
    return "-" if pc is None else f"{pc:.2e}"


def plural(count: int, noun: str, nouns: str | None = None) -> str:
    """``count`` and ``noun``, or its plural ``nouns`` (default: ``noun`` and
    ``s``) unless ``count`` is one: ``1 event``, ``2 events``."""
    if count == 1:
        return f"{count} {noun}"
    return f"{count} {nouns or noun + 's'}"
