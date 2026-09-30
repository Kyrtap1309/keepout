"""Run validation cases and judge them against their criteria."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import NamedTuple

from keepout.pc import compute_pc
from keepout.pc.result import Method, PcResult
from keepout.validation.cases import ValidationCase


class Verdict(NamedTuple):
    """How far a value is from a case's reference, and whether that passes."""

    error: float
    """Relative error if ``relative``, else absolute error."""
    relative: bool
    passed: bool


def judge(case: ValidationCase, result: PcResult) -> Verdict:
    """Compare a result with the case's reference under the case's criterion."""
    criterion, reference = case.criterion, case.reference
    difference = abs(result.value - reference)
    if criterion.interval:
        low, high = result.details["ci_low"], result.details["ci_high"]
        error = difference / reference if reference > 0.0 else difference
        return Verdict(error, reference > 0.0, low <= reference <= high)
    if (
        criterion.relative is not None
        and reference >= criterion.small
        and reference > 0
    ):
        error = difference / reference
        return Verdict(error, True, error <= criterion.relative)
    bound = criterion.absolute if criterion.absolute is not None else 0.0
    return Verdict(difference, False, difference <= bound)


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
        return judge(self.case, self.result)

    @property
    def passed(self) -> bool:
        """Whether the case's criterion is met."""
        return self.verdict.passed


def evaluate(cases: Iterable[ValidationCase]) -> list[Outcome]:
    """Run each case's methods on it."""
    return [
        Outcome(case, compute_pc(case.conjunction, method, **case.options))
        for case in cases
        for method in case.methods
    ]
