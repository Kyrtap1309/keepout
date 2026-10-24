"""How the risk assessment writes its numbers in reasons and notes."""

from __future__ import annotations

from keepout.units import DAY

__all__ = ["days_text"]


def days_text(seconds: float) -> str:
    """A duration in days, as keepout prints it: ``1.50 d``."""
    return f"{seconds / DAY:.2f} d"
