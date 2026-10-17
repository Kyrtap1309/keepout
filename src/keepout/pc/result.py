"""The result of a probability-of-collision computation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Literal, Self, TypeAlias, get_args

from keepout._methods import Method
from keepout._types import override, require_probability

__all__ = ["RESULT_KINDS", "Method", "PcResult", "ResultKind"]

ResultKind: TypeAlias = Literal["estimate", "upper_bound"]
"""What a Pc value is: an ``"estimate"`` of the probability, or an
``"upper_bound"`` on it (a maximum Pc)."""

RESULT_KINDS: Final[tuple[ResultKind, ...]] = get_args(ResultKind)
"""Every :data:`ResultKind`."""


@dataclass(frozen=True, eq=False, slots=True)
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
        Method-specific numbers for display, e.g. ``"order"`` (Elrod),
        ``"error_estimate"`` (Foster), ``"nc"`` (Hall) or ``"samples"``
        (Monte Carlo); stored read-only.
    kind
        Whether ``value`` estimates Pc or bounds it from above (the
        maximum-Pc methods).
    interval
        Confidence interval ``(low, high)`` on Pc, if the method gives one
        (Monte Carlo; its level is ``details["confidence"]``).
    diluted
        Whether the conjunction is in the dilution region: the maximum Pc
        over covariance scaling needs a smaller covariance than the nominal
        one (:func:`~keepout.pc.maxpc.dilution` only; ``None`` otherwise).

    Raises
    ------
    ValueError
        If ``value`` is not in ``[0, 1]``, ``kind`` is unknown, or
        ``interval`` is not two probabilities with ``low <= high``.
    """

    value: float
    method: Method
    diagnostics: tuple[str, ...] = ()
    details: Mapping[str, float] = field(default_factory=dict)
    kind: ResultKind = "estimate"
    interval: tuple[float, float] | None = None
    diluted: bool | None = None

    def __post_init__(self) -> None:
        require_probability(self.value, "Pc")
        if self.kind not in RESULT_KINDS:
            raise ValueError(f"kind must be one of {RESULT_KINDS}, got {self.kind!r}")
        if self.interval is not None:
            low, high = self.interval
            require_probability(low, "interval low")
            require_probability(high, "interval high")
            if low > high:
                raise ValueError(f"interval must have low <= high, got {self.interval}")
            object.__setattr__(self, "interval", (float(low), float(high)))
        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))

    @override
    def __reduce__(self) -> tuple[type[Self], tuple[object, ...]]:
        # A mapping proxy does not pickle: pickle the plain mapping.
        return type(self), (
            self.value,
            self.method,
            self.diagnostics,
            dict(self.details),
            self.kind,
            self.interval,
            self.diluted,
        )

    @classmethod
    def clamped(
        cls,
        value: float,
        method: Method,
        diagnostics: tuple[str, ...] = (),
        details: Mapping[str, float] | None = None,
        *,
        kind: ResultKind = "estimate",
        diluted: bool | None = None,
    ) -> PcResult:
        """A result with ``value`` clipped to ``[0, 1]`` (quadrature round-off)."""
        return cls(
            min(max(value, 0.0), 1.0),
            method,
            diagnostics,
            details or {},
            kind=kind,
            diluted=diluted,
        )
