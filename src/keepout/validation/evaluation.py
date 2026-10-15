"""Run validation cases and judge them against their criteria."""

from __future__ import annotations

import multiprocessing
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

from keepout.pc import compute_pc
from keepout.pc.result import Method, PcResult
from keepout.validation.cases import ValidationCase
from keepout.validation.criteria import Verdict

__all__ = [
    "Outcome",
    "Verdict",
    "confidence_interval",
    "evaluate",
    "judge",
    "run",
]


def confidence_interval(result: PcResult) -> tuple[float, float] | None:
    """The result's confidence interval, if its method gives one (Monte Carlo)."""
    details = result.details
    if "ci_low" not in details:
        return None
    return details["ci_low"], details["ci_high"]


def judge(case: ValidationCase, result: PcResult) -> Verdict:
    """Compare a result with the case's reference under the case's criterion."""
    return case.criterion.judge(
        result.value, case.reference, confidence_interval(result)
    )


@dataclass(frozen=True, eq=False, slots=True)
class Outcome:
    """One method on one case.

    Parameters
    ----------
    case
        The case.
    result
        keepout's Pc, with its method and diagnostics.

    Attributes
    ----------
    ci
        keepout's confidence interval, if the method gives one.
    verdict
        Error against the reference and pass/fail.
    """

    case: ValidationCase
    result: PcResult
    ci: tuple[float, float] | None = field(init=False)
    verdict: Verdict = field(init=False)

    def __post_init__(self) -> None:
        ci = confidence_interval(self.result)
        case = self.case
        verdict = case.criterion.judge(self.result.value, case.reference, ci)
        object.__setattr__(self, "ci", ci)
        object.__setattr__(self, "verdict", verdict)

    @property
    def passed(self) -> bool:
        """Whether the case's criterion is met."""
        return self.verdict.passed


def run(case: ValidationCase, method: Method) -> PcResult:
    """Keepout's Pc of a case by one of its methods, with the case's options.

    :class:`ValidationCase` checks that its methods take its options, so
    each method is given all of them.
    """
    conjunction, options = case.conjunction, case.options
    if method == "montecarlo":
        return compute_pc(conjunction, method, **options)
    if method == "hall3d":
        return compute_pc(conjunction, method, window=options.get("window"))
    return compute_pc(conjunction, method)


def evaluate(cases: Iterable[ValidationCase], *, workers: int = 1) -> list[Outcome]:
    """Run each case's methods on it, in order.

    Parameters
    ----------
    cases
        The cases.
    workers
        Processes to run them in; 1 runs them in this one. The results do
        not depend on it (Monte Carlo cases have their own seeds).
    """
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    tasks = [(case, method) for case in cases for method in case.methods]
    if workers == 1 or len(tasks) < 2:
        return [_outcome(task) for task in tasks]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        return list(pool.map(_outcome, tasks))


def _outcome(task: tuple[ValidationCase, Method]) -> Outcome:
    case, method = task
    return Outcome(case, run(case, method))
