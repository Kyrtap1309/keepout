"""Run validation cases and judge them against the spec tolerances."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import NamedTuple, get_args

from keepout.pc import compute_pc
from keepout.pc.result import Method, PcResult
from keepout.validation.cases import ValidationCase

RELATIVE_TOLERANCE = 1e-6
"""Spec section 5: 2-D methods, relative error where the reference >= 1e-10."""

ABSOLUTE_TOLERANCE = 1e-15
"""Spec section 5: 2-D methods, absolute error where the reference < 1e-10."""

SMALL_PC = 1e-10
"""Reference Pc below which the absolute criterion applies."""

METHODS: tuple[Method, ...] = get_args(Method)


class Verdict(NamedTuple):
    """How far a value is from a case's reference, and whether that passes."""

    error: float
    """Relative error if ``relative``, else absolute error."""
    relative: bool
    passed: bool


def judge(case: ValidationCase, value: float) -> Verdict:
    """Compare one value with the case's reference under its criterion."""
    difference = abs(value - case.reference)
    if case.absolute_tolerance is not None:
        return Verdict(difference, False, difference <= case.absolute_tolerance)
    if case.reference >= SMALL_PC:
        error = difference / case.reference
        return Verdict(error, True, error <= RELATIVE_TOLERANCE)
    return Verdict(difference, False, difference <= ABSOLUTE_TOLERANCE)


@dataclass(frozen=True, eq=False)
class Outcome:
    """One method on one case.

    Parameters
    ----------
    case
        The case.
    result
        keepout's Pc, with its method and diagnostics.
    """

    case: ValidationCase
    result: PcResult

    @property
    def method(self) -> Method:
        """The Pc method."""
        return self.result.method

    @property
    def value(self) -> float:
        """keepout's Pc."""
        return self.result.value

    @property
    def verdict(self) -> Verdict:
        """Error against the reference and pass/fail."""
        return judge(self.case, self.value)

    @property
    def passed(self) -> bool:
        """Whether the case's criterion is met."""
        return self.verdict.passed


def evaluate(cases: Iterable[ValidationCase]) -> list[Outcome]:
    """Run every method on every case."""
    return [
        Outcome(case, compute_pc(case.conjunction, method))
        for case in cases
        for method in METHODS
    ]
