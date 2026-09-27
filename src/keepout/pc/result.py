"""The result of a probability-of-collision computation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, TypeAlias

Method: TypeAlias = Literal["foster", "elrod"]
"""Pc methods implemented so far."""


@dataclass(frozen=True, eq=False)
class PcResult:
    """Probability of collision with the method used and what to watch out for.

    Parameters
    ----------
    value
        Probability of collision, in ``[0, 1]``.
    method
        The method that produced ``value``.
    diagnostics
        Human-readable warnings: covariance repairs, questionable 2-D
        assumptions, quadrature problems. Empty when there is nothing to report.
    details
        Method-specific numbers, e.g. ``"order"`` (Elrod) or
        ``"error_estimate"`` (Foster); stored read-only.

    Raises
    ------
    ValueError
        If ``value`` is not in ``[0, 1]``.
    """

    value: float
    method: Method
    diagnostics: tuple[str, ...] = ()
    details: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (math.isfinite(self.value) and 0.0 <= self.value <= 1.0):
            raise ValueError(f"Pc must be in [0, 1], got {self.value}")
        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))
