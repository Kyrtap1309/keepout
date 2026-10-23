"""Argument checks shared by the package, without NumPy.

Each ``require_*`` raises :class:`ValueError` naming the argument;
:func:`positive_float` and :func:`non_negative_float` also return the value
as a float, for a dataclass to store.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astropy.time import Time

__all__ = [
    "is_probability",
    "is_single_line",
    "non_negative_float",
    "positive_float",
    "require_at_least",
    "require_non_negative",
    "require_positive",
    "require_probability",
    "require_single_time",
]


def require_positive(value: float, name: str) -> None:
    """A check that ``value`` is finite and positive (``ValueError`` if not).

    Raises
    ------
    ValueError
        If ``value`` is not finite or not greater than zero.
    """
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be finite and > 0, got {value}")


def require_non_negative(value: float, name: str) -> None:
    """A check that ``value`` is finite and non-negative (``ValueError`` if not).

    Raises
    ------
    ValueError
        If ``value`` is not finite or is negative.
    """
    if not (math.isfinite(value) and value >= 0.0):
        raise ValueError(f"{name} must be finite and >= 0, got {value}")


def is_probability(value: float) -> bool:
    """Whether ``value`` is a probability: finite and in [0, 1]."""
    return math.isfinite(value) and 0.0 <= value <= 1.0


def require_probability(value: float, name: str) -> None:
    """A check that ``value`` is a probability (``ValueError`` if not).

    Raises
    ------
    ValueError
        If ``value`` is not finite or not in [0, 1].
    """
    if not is_probability(value):
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def require_at_least(value: int, minimum: int, name: str) -> None:
    """A check that the count ``value`` is at least ``minimum`` (``ValueError`` if not).

    Raises
    ------
    ValueError
        If ``value`` is below ``minimum``.
    """
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")


def require_single_time(time: Time, name: str) -> None:
    """A check that ``time`` is a single time, not an array (``ValueError`` if not).

    Raises
    ------
    ValueError
        If ``time`` is not scalar.
    """
    if not time.isscalar:
        raise ValueError(f"{name} must be a single time, got shape {time.shape}")


def positive_float(value: float, name: str) -> float:
    """``value`` as a float, checked to be finite and positive.

    Raises
    ------
    ValueError
        If it is not finite or not greater than zero.
    """
    number = float(value)
    require_positive(number, name)
    return number


def non_negative_float(value: float, name: str) -> float:
    """``value`` as a float, checked to be finite and non-negative.

    Raises
    ------
    ValueError
        If it is not finite or is negative.
    """
    number = float(value)
    require_non_negative(number, name)
    return number


def is_single_line(text: str) -> bool:
    """Whether ``text`` is at most one line, without leading or trailing
    blanks (it then survives a KVN and an XML round-trip)."""
    return len(text.splitlines()) <= 1 and text == text.strip()
