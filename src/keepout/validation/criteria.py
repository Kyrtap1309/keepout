"""Pass/fail criteria of the validation cases (spec section 5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, NamedTuple, Protocol

from keepout._types import override
from keepout._validation import require_non_negative

__all__ = [
    "HALL",
    "HALL_PC_SET",
    "MONTE_CARLO",
    "REPRODUCIBILITY_FRACTION",
    "TWO_D",
    "BoundCriterion",
    "Criterion",
    "IntervalCriterion",
    "Verdict",
]


class Verdict(NamedTuple):
    """How far a value is from a case's reference, and whether that passes."""

    error: float
    """Relative error if ``relative``, else absolute error."""
    relative: bool
    passed: bool


REPRODUCIBILITY_FRACTION: Final = 0.25
"""Share of a criterion's margin within which two runs give the same result.
Runs on different CPUs differ by up to ~1/20 of it (BLAS kernels, adaptive
quadrature paths, Monte Carlo hits at the sphere boundary)."""


class Criterion(Protocol):
    """How a method's value is compared with a case's reference:
    :class:`BoundCriterion` or :class:`IntervalCriterion`."""

    __slots__ = ()

    def judge(
        self,
        value: float,
        reference: float,
        interval: tuple[float, float] | None = None,
    ) -> Verdict:
        """The verdict on ``value`` against ``reference``; ``interval`` is
        the result's confidence interval, if it has one."""
        ...

    def margin(
        self, reference: float, interval: tuple[float, float] | None = None
    ) -> float:
        """Absolute difference from ``reference`` the criterion allows."""
        ...

    def describe(self) -> str:
        """The criterion in words, for the report."""
        ...

    def reproduces(
        self,
        previous: float,
        current: float,
        reference: float,
        interval: tuple[float, float] | None = None,
    ) -> bool:
        """Whether ``current`` repeats ``previous`` within
        :data:`REPRODUCIBILITY_FRACTION` of the :meth:`margin` at ``reference``.
        """
        allowed = REPRODUCIBILITY_FRACTION * self.margin(reference, interval)
        return abs(current - previous) <= allowed


@dataclass(frozen=True, slots=True)
class BoundCriterion(Criterion):
    """A bound on the error of a method's value.

    Parameters
    ----------
    relative
        Largest relative error, used where the reference is at least
        ``small``.
    absolute
        Largest absolute error, used where the reference is below ``small``
        or no relative bound is given.
    small
        Reference below which the absolute bound applies.

    Raises
    ------
    ValueError
        If no bound is given, a bound is negative or not finite, or ``small``
        is given without ``absolute``.
    """

    relative: float | None = None
    absolute: float | None = None
    small: float = 0.0

    def __post_init__(self) -> None:
        if self.relative is None and self.absolute is None:
            raise ValueError("a criterion needs a relative or an absolute bound")
        for name, bound in (
            ("relative", self.relative),
            ("absolute", self.absolute),
            ("small", self.small),
        ):
            if bound is not None:
                require_non_negative(bound, name)
        if self.small > 0.0 and self.absolute is None:
            raise ValueError("small needs the absolute bound that applies below it")

    @override
    def judge(
        self,
        value: float,
        reference: float,
        interval: tuple[float, float] | None = None,
    ) -> Verdict:
        """The verdict on ``value`` against ``reference`` (``interval`` is not
        used).

        The relative bound falls back to the absolute bound (0 if none)
        where the reference is below ``small`` or is 0.
        """
        del interval  # a bound does not depend on it
        difference = abs(value - reference)
        relative = self._relative_at(reference)
        if relative is not None:
            error = difference / reference
            return Verdict(error, relative=True, passed=error <= relative)
        return Verdict(
            difference, relative=False, passed=difference <= self.margin(reference)
        )

    @override
    def margin(
        self, reference: float, interval: tuple[float, float] | None = None
    ) -> float:
        """The bound that applies at ``reference``, as an absolute difference."""
        del interval  # a bound does not depend on it
        relative = self._relative_at(reference)
        if relative is not None:
            return relative * reference
        return self.absolute if self.absolute is not None else 0.0

    def _relative_at(self, reference: float) -> float | None:
        """The relative bound if it applies at ``reference``."""
        if reference >= self.small and reference > 0.0:
            return self.relative
        return None

    @override
    def describe(self) -> str:
        """The criterion in words, for the report."""
        if self.relative is None:
            return f"absolute error at most {self.absolute:g}"
        text = f"relative error at most {self.relative:g}"
        if self.absolute is not None:
            text += (
                f" (absolute error at most {self.absolute:g} where the reference"
                f" is below {self.small:g})"
            )
        return text


@dataclass(frozen=True, slots=True)
class IntervalCriterion(Criterion):
    """Pass if the reference lies within the result's confidence interval
    (:attr:`~keepout.pc.PcResult.interval`).

    Each method raises :class:`ValueError` if the interval is not given.
    """

    @override
    def judge(
        self,
        value: float,
        reference: float,
        interval: tuple[float, float] | None = None,
    ) -> Verdict:
        """The verdict on ``value`` against ``reference``: passed if
        ``reference`` lies in ``interval``; the error is relative unless the
        reference is 0."""
        difference = abs(value - reference)
        low, high = _required(interval)
        error = difference / reference if reference > 0.0 else difference
        return Verdict(error, reference > 0.0, low <= reference <= high)

    @override
    def margin(
        self, reference: float, interval: tuple[float, float] | None = None
    ) -> float:
        """The half-width of the result's ``interval``."""
        del reference  # the interval does not depend on it
        low, high = _required(interval)
        return 0.5 * (high - low)

    @override
    def describe(self) -> str:
        """The criterion in words, for the report."""
        return "CARA's value within keepout's confidence interval"


def _required(interval: tuple[float, float] | None) -> tuple[float, float]:
    if interval is None:
        raise ValueError("an interval criterion needs the result's interval")
    return interval


TWO_D: Final = BoundCriterion(relative=1e-6, absolute=1e-15, small=1e-10)
"""Spec section 5: 2-D and maximum Pc methods."""

HALL: Final = BoundCriterion(relative=1e-3)
"""Hall 3D-Nc against CARA's Alfano and unit-test values: the tolerance of
CARA's own ``Pc3D_Hall_UnitTest.m``."""

HALL_PC_SET: Final = BoundCriterion(relative=3e-3)
"""Hall 3D-Nc against the 53-CDM set's ``Nc3D``: CARA's values there sit
systematically above its own ``Nc2D`` on nearly straight-line encounters. On
the 18 cases where keepout's 3D-Nc and 2-D Pc agree within 1e-3, CARA's
``Nc3D / Nc2D - 1`` averages +8.2e-4, while keepout's Hall / Foster - 1 spreads
from -7.7e-4 to +7.0e-4 around a mean of -3e-5 (see docs/pc-3d.md)."""

MONTE_CARLO: Final = IntervalCriterion()
"""Spec section 5: the reference lies within the Monte Carlo interval."""
