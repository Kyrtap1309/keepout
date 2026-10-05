"""The result of a probability-of-collision computation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, TypeAlias

__all__ = ["Method", "PcResult"]

Method: TypeAlias = Literal[
    "foster", "elrod", "hall3d", "montecarlo", "frisbee", "dilution", "alfano"
]
"""Pc methods: 2-D (``foster``, ``elrod``), 3-D (``hall3d``), Monte Carlo
(``montecarlo``) and maximum Pc (``frisbee``, ``dilution``, ``alfano``)."""


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
        Method-specific numbers, e.g. ``"order"`` (Elrod),
        ``"error_estimate"`` (Foster), ``"nc"`` (Hall) or ``"ci_low"`` and
        ``"ci_high"`` (Monte Carlo); stored read-only.

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

    @classmethod
    def clamped(
        cls,
        value: float,
        method: Method,
        diagnostics: tuple[str, ...] = (),
        details: Mapping[str, float] | None = None,
    ) -> PcResult:
        """A result with ``value`` clipped to ``[0, 1]`` (quadrature round-off)."""
        return cls(min(max(value, 0.0), 1.0), method, diagnostics, details or {})
