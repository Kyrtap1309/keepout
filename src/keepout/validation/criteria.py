"""Pass/fail criteria of the validation cases (spec section 5)."""

from __future__ import annotations

from dataclasses import dataclass


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
    """

    relative: float | None = None
    absolute: float | None = None
    small: float = 0.0
    interval: bool = False

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
