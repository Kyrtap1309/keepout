"""Words for messages and notes, shared by the library and the command line."""

from __future__ import annotations

from keepout.units import HOUR

__all__ = ["before_tca", "plural"]


def before_tca(seconds: float) -> str:
    """A time before TCA in hours: ``6.0 h before TCA``."""
    return f"{seconds / HOUR:.1f} h before TCA"


def plural(count: int, noun: str, nouns: str | None = None) -> str:
    """``count`` and ``noun``, or its plural ``nouns`` (default: ``noun`` and
    ``s``) unless ``count`` is one: ``1 event``, ``2 events``."""
    if count == 1:
        return f"{count} {noun}"
    return f"{count} {nouns or noun + 's'}"
