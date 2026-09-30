"""Pass/fail criteria of the validation cases (spec section 5)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple


class Verdict(NamedTuple):
    """How far a value is from a case's reference, and whether that passes."""

    error: float
    """Relative error if ``relative``, else absolute error."""
    relative: bool
    passed: bool


@dataclass(frozen=True)
class Criterion:
    """How a method's value is compared with a case's reference.

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
    interval
        Pass if the reference lies within the result's confidence interval
        (``details["ci_low"]``, ``details["ci_high"]``) instead.

    Raises
    ------
    ValueError
        If no bound is given, bounds are given with ``interval``, a bound is
        negative or not finite, or ``small`` is given without ``absolute``.
    """

    relative: float | None = None
    absolute: float | None = None
    small: float = 0.0
    interval: bool = False

    def __post_init__(self) -> None:
        bounds = [b for b in (self.relative, self.absolute) if b is not None]
        if self.interval and bounds:
            raise ValueError("an interval criterion takes no error bounds")
        if not self.interval and not bounds:
            raise ValueError("a criterion needs a relative or an absolute bound")
        if not all(math.isfinite(b) and b >= 0.0 for b in [*bounds, self.small]):
            raise ValueError("criterion bounds must be finite and >= 0")
        if self.small > 0.0 and self.absolute is None:
            raise ValueError("small needs the absolute bound that applies below it")

    def judge(
        self,
        value: float,
        reference: float,
        interval: tuple[float, float] | None = None,
    ) -> Verdict:
        """Compare ``value`` with ``reference``.

        ``interval`` is the result's confidence interval, required by an
        interval criterion. A relative criterion falls back to the absolute
        bound (0 if none) where the reference is below ``small`` or is 0.
        """
        difference = abs(value - reference)
        if self.interval:
            if interval is None:
                raise ValueError("an interval criterion needs the result's interval")
            low, high = interval
            error = difference / reference if reference > 0.0 else difference
            return Verdict(error, reference > 0.0, low <= reference <= high)
        if self.relative is not None and reference >= self.small and reference > 0.0:
            error = difference / reference
            return Verdict(error, True, error <= self.relative)
        bound = self.absolute if self.absolute is not None else 0.0
        return Verdict(difference, False, difference <= bound)

    def describe(self) -> str:
        """The criterion in words, for the report."""
        if self.interval:
            return "CARA's value within keepout's confidence interval"
        if self.relative is None:
            return f"absolute error at most {self.absolute:g}"
        text = f"relative error at most {self.relative:g}"
        if self.absolute is not None:
            text += (
                f" (absolute error at most {self.absolute:g} where the reference"
                f" is below {self.small:g})"
            )
        return text


TWO_D = Criterion(relative=1e-6, absolute=1e-15, small=1e-10)
"""Spec section 5: 2-D and maximum Pc methods."""

HALL = Criterion(relative=1e-3)
"""Hall 3D-Nc against CARA's Alfano and unit-test values: the tolerance of
CARA's own ``Pc3D_Hall_UnitTest.m``."""

HALL_PC_SET = Criterion(relative=3e-3)
"""Hall 3D-Nc against the 53-CDM set's ``Nc3D``: CARA's values there sit
systematically above its own ``Nc2D`` on nearly straight-line encounters. On
the 18 cases where keepout's 3D-Nc and 2-D Pc agree within 1e-3, CARA's
``Nc3D / Nc2D - 1`` averages +8.2e-4, while keepout's Hall / Foster - 1 spreads
from -7.7e-4 to +7.0e-4 around a mean of -3e-5 (see docs/pc-3d.md)."""

MONTE_CARLO = Criterion(interval=True)
"""Spec section 5: the reference lies within the Monte Carlo interval."""
