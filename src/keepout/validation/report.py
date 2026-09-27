"""Run validation cases and render the Markdown report."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import get_args

from keepout.pc import compute_pc
from keepout.pc.result import Method
from keepout.sources.cara import CARA_COMMIT
from keepout.validation.cases import ValidationCase

RELATIVE_TOLERANCE = 1e-6
"""Spec section 5: 2-D methods, relative error where the reference >= 1e-10."""

ABSOLUTE_TOLERANCE = 1e-15
"""Spec section 5: 2-D methods, absolute error where the reference < 1e-10."""

SMALL_PC = 1e-10
"""Reference Pc below which the absolute criterion applies."""

PAPER_TOLERANCE = 1e-3
"""Relative agreement with Alfano's published values (as in CARA's own tests)."""

METHODS: tuple[Method, ...] = get_args(Method)

SUMMARY_START = "<!-- validation-summary:start -->"
SUMMARY_END = "<!-- validation-summary:end -->"


@dataclass(frozen=True)
class Outcome:
    """One method on one case.

    Parameters
    ----------
    case
        The case.
    method
        The Pc method.
    value
        keepout's Pc.
    error
        Relative error (``relative`` true) or absolute error.
    relative
        Whether ``error`` is relative.
    passed
        Whether the case's criterion is met.
    """

    case: ValidationCase
    method: Method
    value: float
    error: float
    relative: bool
    passed: bool


def judge(case: ValidationCase, value: float) -> tuple[float, bool, bool]:
    """Error, whether it is relative, and pass/fail for one value."""
    difference = abs(value - case.reference)
    if case.absolute_tolerance is not None:
        return difference, False, difference <= case.absolute_tolerance
    if case.reference >= SMALL_PC:
        error = difference / case.reference
        return error, True, error <= RELATIVE_TOLERANCE
    return difference, False, difference <= ABSOLUTE_TOLERANCE


def evaluate(cases: Iterable[ValidationCase]) -> list[Outcome]:
    """Run every method on every case."""
    outcomes: list[Outcome] = []
    for case in cases:
        for method in METHODS:
            value = compute_pc(case.conjunction, method).value
            error, relative, passed = judge(case, value)
            outcomes.append(Outcome(case, method, value, error, relative, passed))
    return outcomes


def _error(outcome: Outcome) -> str:
    return f"{outcome.error:.1e}" if outcome.relative else f"{outcome.error:.1e} (abs)"


def _mark(passed: bool) -> str:
    return "pass" if passed else "**FAIL**"


def _worst(outcomes: Sequence[Outcome]) -> str:
    relative = [o.error for o in outcomes if o.relative]
    return f"{max(relative):.1e}" if relative else "-"


def render_summary(outcomes: Sequence[Outcome]) -> str:
    """Markdown table: per method, cases passed and worst relative error."""
    lines = [
        "| Method | Cases passed | Worst relative error |",
        "|---|---|---|",
    ]
    for method in METHODS:
        mine = [o for o in outcomes if o.method == method]
        passed = sum(o.passed for o in mine)
        lines.append(f"| {method} | {passed} / {len(mine)} | {_worst(mine)} |")
    return "\n".join(lines)


def _suite_table(outcomes: Sequence[Outcome], with_paper: bool) -> list[str]:
    header = "| Case | HBR [m] | CARA Pc | Foster | error | Elrod | error |"
    rule = "|---|---|---|---|---|---|---|"
    if with_paper:
        header += " Alfano paper | vs paper |"
        rule += "---|---|"
    lines = [header, rule]
    by_case: dict[str, dict[Method, Outcome]] = {}
    for outcome in outcomes:
        by_case.setdefault(outcome.case.case_id, {})[outcome.method] = outcome
    for case_id, pair in by_case.items():
        foster, elrod = pair["foster"], pair["elrod"]
        case = foster.case
        row = (
            f"| {case_id} | {case.conjunction.hard_body_radius:g} "
            f"| {case.reference:.9e} "
            f"| {foster.value:.9e} | {_error(foster)} {_mark(foster.passed)} "
            f"| {elrod.value:.9e} | {_error(elrod)} {_mark(elrod.passed)} |"
        )
        if with_paper and case.paper is not None:
            paper_error = abs(foster.value - case.paper) / case.paper
            row += (
                f" {case.paper:.9e} | {paper_error:.1e} "
                f"{_mark(paper_error <= PAPER_TOLERANCE)} |"
            )
        lines.append(row)
    return lines


NOT_COMPARED = """\
## Not compared

- **Alfano case 12** has zero relative velocity: the encounter plane is
  undefined and CARA gives no 2-D value. keepout raises `GeometryError`.
- **Omitron inline cases** in CARA's `Pc2D_Foster_UnitTest.m` and
  `PcElrod_UnitTest.m` give states about 0.5 m from closest approach. CARA's
  2-D functions then place the full relative distance on the encounter plane;
  keepout projects the relative position onto the plane (the straight-line
  closest approach) and reports that the states are off TCA. The 53-CDM set
  above compares the two conventions where they coincide (CARA's `Pc2D`,
  computed after moving the states to TCA).
- **Alfano CDM files** (`DataFiles/SampleCDMs/AlfanoTestCase*.cdm`) give
  `RELATIVE_VELOCITY_*` in `[m]`; keepout rejects the unit, as CCSDS
  508.0-B-1 requires `[m/s]`.
"""


def render_page(outcomes: Sequence[Outcome]) -> str:
    """The full ``docs/validation.md`` page."""
    suites: dict[str, list[Outcome]] = {}
    for outcome in outcomes:
        suites.setdefault(outcome.case.suite, []).append(outcome)
    parts = [
        "# Validation against NASA CARA",
        "",
        "Generated by `python -m keepout.validation` from "
        f"[CARA Analysis Tools](https://github.com/nasa/CARA_Analysis_Tools) "
        f"commit `{CARA_COMMIT[:12]}`. Do not edit by hand.",
        "",
        f"Criterion: relative error at most {RELATIVE_TOLERANCE:g} where the "
        f"CARA Pc is at least {SMALL_PC:g}, absolute error at most "
        f"{ABSOLUTE_TOLERANCE:g} below that, unless a case states its own "
        "tolerance.",
        "",
        "## Summary",
        "",
        render_summary(outcomes),
        "",
    ]
    for suite, members in suites.items():
        parts += [
            f"## {suite}",
            "",
            *_suite_table(members, suite == "Alfano (2009)"),
            "",
        ]
    parts.append(NOT_COMPARED)
    return "\n".join(parts)


def replace_summary(readme: str, summary: str) -> str:
    """Put ``summary`` between the README's validation-summary markers.

    Raises
    ------
    ValueError
        If the markers are missing or out of order.
    """
    start = readme.find(SUMMARY_START)
    end = readme.find(SUMMARY_END)
    if start < 0 or end < start:
        raise ValueError(
            f"README needs {SUMMARY_START} and {SUMMARY_END}, in that order"
        )
    head = readme[: start + len(SUMMARY_START)]
    return f"{head}\n{summary}\n{readme[end:]}"
