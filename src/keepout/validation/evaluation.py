"""Validation cases run and judged against their criteria."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from keepout._workers import process_pool, resolve_workers
from keepout.pc import PcComputation
from keepout.validation.criteria import Verdict

if TYPE_CHECKING:
    from collections.abc import Iterable

    from keepout.pc.result import Method, PcResult
    from keepout.validation.cases import Suite, ValidationCase

__all__ = [
    "MethodSummary",
    "Outcome",
    "Verdict",
    "evaluate",
    "judge",
    "run",
    "summarise",
]


def judge(case: ValidationCase, result: PcResult) -> Verdict:
    """The verdict on a result against the case's reference, by the case's criterion."""
    return case.criterion.judge(result.value, case.reference, result.interval)


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
    verdict
        Error against the reference and pass/fail.
    """

    case: ValidationCase
    result: PcResult
    verdict: Verdict = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "verdict", judge(self.case, self.result))

    @property
    def passed(self) -> bool:
        """Whether the case's criterion is met."""
        return self.verdict.passed


def run(case: ValidationCase, method: Method) -> PcResult:
    """The Pc keepout computes for a case by one of its methods, with its options.

    :class:`ValidationCase` checks that its methods take its options, so
    each method is given all of them.
    """
    return PcComputation(method, case.options)(case.conjunction)


def evaluate(
    cases: Iterable[ValidationCase], *, workers: int | None = 1
) -> list[Outcome]:
    """The outcomes of each case's methods run on it, in order.

    Parameters
    ----------
    cases
        The cases.
    workers
        Processes to run them in; 1 runs them in this one, ``None`` one per
        CPU, at most 4. The results do not depend on it (Monte Carlo cases
        have their own seeds).

    Raises
    ------
    ValueError
        If ``workers`` is below 1.
    """
    count = resolve_workers(workers)
    tasks: list[tuple[ValidationCase, Method]] = [
        (case, method) for case in cases for method in case.methods
    ]
    if count == 1 or len(tasks) < 2:
        return [_outcome(task) for task in tasks]
    with process_pool(count) as pool:
        return list(pool.map(_outcome, tasks))


def _outcome(task: tuple[ValidationCase, Method]) -> Outcome:
    case, method = task
    return Outcome(case, run(case, method))


@dataclass(frozen=True, slots=True)
class MethodSummary:
    """One method on the cases of one suite: how many passed, the worst error.

    Parameters
    ----------
    suite
        The suite (:attr:`~keepout.validation.cases.ValidationCase.suite`).
    method
        The keepout method.
    passed
        Cases whose criterion the method meets.
    total
        Cases the method ran on.
    worst_relative_error
        Largest relative error among the cases judged by a relative bound;
        ``None`` if none is.
    """

    suite: Suite
    method: Method
    passed: int
    total: int
    worst_relative_error: float | None


def summarise(outcomes: Iterable[Outcome]) -> tuple[MethodSummary, ...]:
    """Per suite and method, the cases passed and the worst relative error.

    Suites are in the order they first appear, and so are the methods
    within a suite (the order of the summary in ``docs/validation.md``).
    """
    suites: dict[Suite, dict[Method, list[Outcome]]] = {}
    for outcome in outcomes:
        methods = suites.setdefault(outcome.case.suite, {})
        methods.setdefault(outcome.result.method, []).append(outcome)
    return tuple(
        MethodSummary(
            suite,
            method,
            sum(o.passed for o in members),
            len(members),
            max(
                (o.verdict.error for o in members if o.verdict.relative),
                default=None,
            ),
        )
        for suite, methods in suites.items()
        for method, members in methods.items()
    )
