"""Words for messages and notes, shared by the library and the command line."""

from __future__ import annotations

__all__ = ["plural"]


def plural(count: int, noun: str, nouns: str | None = None) -> str:
    """``count`` and ``noun``, or its plural ``nouns`` (default: ``noun`` and
    ``s``) unless ``count`` is one: ``1 event``, ``2 events``."""
    if count == 1:
        return f"{count} {noun}"
    return f"{count} {nouns or noun + 's'}"
