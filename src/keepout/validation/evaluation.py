"""Run validation cases and judge them against their criteria."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property

from keepout.pc import compute_pc
from keepout.pc.result import Method, PcResult
from keepout.validation.cases import ValidationCase
from keepout.validation.criteria import Verdict

__all__ = ["Outcome", "Verdict", "evaluate", "judge"]


def judge(case: ValidationCase, result: PcResult) -> Verdict:
    """Compare a result with the case's reference under the case's criterion."""
    interval = None
    if case.criterion.interval:
        interval = (result.details["ci_low"], result.details["ci_high"])
    return case.criterion.judge(result.value, case.reference, interval)


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

    @cached_property
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
