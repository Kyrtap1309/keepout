"""Regenerate the validation reports and the README summaries from CARA cases.

Usage: ``uv run python tools/validation_report.py [--monte-carlo] [--check]``.
Downloads the pinned CARA files, runs every case, and writes the page and the
README summary block: ``docs/validation.md`` (2-D, 3-D and maximum Pc; a few
minutes) or, with ``--monte-carlo``, ``docs/validation-monte-carlo.md`` (up
to hours). ``--check`` fails instead of writing when a file is out of date.
Exits 1 if a case fails.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from keepout.pc.result import Method
from keepout.sources.cara import CARA_COMMIT
from keepout.validation.cases import ValidationCase, load_all, load_monte_carlo
from keepout.validation.evaluation import Outcome, evaluate

REPO = Path(__file__).resolve().parents[1]

PAPER_TOLERANCE = 1e-3
"""Relative agreement with Alfano's published values (as in CARA's own tests)."""

PAPER_METHOD: Method = "foster"
"""Method compared with Alfano's published values."""

NOT_COMPARED = """\
## Not compared

- **Alfano case 12, 2-D**: zero relative velocity leaves the encounter plane
  undefined; CARA gives no 2-D value and keepout raises `GeometryError`. Its
  3-D and Monte Carlo values are compared.
- **Omitron inline cases** in CARA's `Pc2D_Foster_UnitTest.m` and
  `PcElrod_UnitTest.m` give states about 0.5 m from closest approach. CARA's
  2-D functions then place the full relative distance on the encounter plane;
  keepout projects the relative position onto the plane (the straight-line
  closest approach) and reports that the states are off TCA. The 53-CDM set
  compares the two conventions where they coincide (CARA's `Pc2D`, computed
  after moving the states to TCA).
- **Alfano CDM files** (`DataFiles/SampleCDMs/AlfanoTestCase*.cdm`) give
  `RELATIVE_VELOCITY_*` in `[m]`; keepout's CDM reader rejects the unit, as
  CCSDS 508.0-B-1 requires `[m/s]`. The same cases are compared from the
  `.xls` sheets.
- **CARA's Coppola 3-D Pc** is another method; the 53-CDM 3-D table shows it
  for information.
- **Frisbee tests 1-11 and 14-27** of `FrisbeeMaxPc_UnitTest.m`: with both
  covariances known CARA drops the one giving the larger Mahalanobis distance
  (tests 1-11 only assert the result is at least the 2-D Pc), and tests 14-27
  use a square hard body of equal area. keepout's `frisbee` needs one unknown
  covariance and a circular hard body.
- **Dilution test 4** of `DilutionMaxPc_UnitTest.m`: its CDM is not in the
  repository.
- **Alfano (2005) maximum Pc** (`alfano`): CARA has no reference value; unit
  tests check it against a numerical maximisation over every covariance.
- **Covariance cross-correlation** (Casali et al. 2018; Hall 2021, eq. 57):
  not implemented. CARA computed the 53-CDM set without it
  (`apply_covXcorr_corrections = false` in `PcMultiStep_FromCDM.m`).
- **Monte Carlo** results are on a separate page,
  [validation-monte-carlo.md](validation-monte-carlo.md).
"""

NOT_COMPARED_MONTE_CARLO = """\
## Not compared

- **53-CDM cases with a CARA Monte Carlo Pc below 1e-4**: about 400 expected
  collisions would need more than 4e6 trials; the 3-D page compares Hall's
  3D-Nc with CARA on every case instead.
"""


@dataclass(frozen=True)
class Report:
    """One validation page and its README summary block."""

    title: str
    page: Path
    start: str
    end: str
    not_compared: str


MAIN = Report(
    "Validation against NASA CARA",
    REPO / "docs/validation.md",
    "<!-- validation-summary:start -->",
    "<!-- validation-summary:end -->",
    NOT_COMPARED,
)
MONTE_CARLO = Report(
    "Monte Carlo validation against NASA CARA",
    REPO / "docs/validation-monte-carlo.md",
    "<!-- validation-monte-carlo-summary:start -->",
    "<!-- validation-monte-carlo-summary:end -->",
    NOT_COMPARED_MONTE_CARLO,
)

SUMMARY_START, SUMMARY_END = MAIN.start, MAIN.end


def _error(outcome: Outcome) -> str:
    error, relative, _ = outcome.verdict
    return f"{error:.1e}" if relative else f"{error:.1e} (abs)"


def _mark(passed: bool) -> str:
    return "pass" if passed else "**FAIL**"


def _worst(outcomes: Sequence[Outcome]) -> str:
    relative = [o.verdict.error for o in outcomes if o.verdict.relative]
    return f"{max(relative):.1e}" if relative else "-"


def _groups(outcomes: Sequence[Outcome]) -> dict[str, list[Outcome]]:
    suites: defaultdict[str, list[Outcome]] = defaultdict(list)
    for outcome in outcomes:
        suites[outcome.case.suite].append(outcome)
    return dict(suites)


def render_summary(outcomes: Sequence[Outcome]) -> str:
    """Markdown table: per suite and method, cases passed and worst error."""
    lines = [
        "| Suite | Method | Cases passed | Worst relative error |",
        "|---|---|---|---|",
    ]
    for suite, members in _groups(outcomes).items():
        methods = dict.fromkeys(o.method for o in members)
        for method in methods:
            mine = [o for o in members if o.method == method]
            passed = sum(o.passed for o in mine)
            lines.append(
                f"| {suite} | {method} | {passed} / {len(mine)} | {_worst(mine)} |"
            )
    return "\n".join(lines)


def _criteria(members: Sequence[Outcome]) -> list[str]:
    return list(dict.fromkeys(o.case.criterion.describe() for o in members))


def _suite_table(members: Sequence[Outcome]) -> list[str]:
    methods: list[Method] = list(dict.fromkeys(o.method for o in members))
    cases: dict[str, ValidationCase] = {}
    by_case: defaultdict[str, dict[Method, Outcome]] = defaultdict(dict)
    for outcome in members:
        cases[outcome.case.case_id] = outcome.case
        by_case[outcome.case.case_id][outcome.method] = outcome
    every = list(cases.values())
    with_paper = any(c.paper is not None for c in every)
    with_interval = any(c.interval is not None for c in every)
    informative = next((c.informative[0] for c in every if c.informative), None)
    mixed = len(_criteria(members)) > 1
    header = "| Case | HBR [m] | CARA Pc |"
    if with_interval:
        header += " CARA 95% interval |"
    for method in methods:
        header += f" {method.capitalize()} | error |"
        if method == "montecarlo":
            header += " keepout interval | trials |"
    if with_paper:
        header += " Alfano paper | vs paper |"
    if informative:
        header += f" {informative} |"
    if mixed:
        header += " Criterion |"
    lines = [header, "|" + "---|" * header.count(" |")]
    for case_id, case in cases.items():
        row = (
            f"| {case_id} | {case.conjunction.hard_body_radius:g} "
            f"| {case.reference:.9e} |"
        )
        if with_interval:
            low, high = case.interval if case.interval else (0.0, 0.0)
            row += f" [{low:.4e}, {high:.4e}] |" if case.interval else " - |"
        for method in methods:
            found = by_case[case_id].get(method)
            if found is None:
                row += " - | - |" + (" - | - |" if method == "montecarlo" else "")
                continue
            row += f" {found.value:.9e} | {_error(found)} {_mark(found.passed)} |"
            if method == "montecarlo":
                details = found.result.details
                row += (
                    f" [{details['ci_low']:.4e}, {details['ci_high']:.4e}] "
                    f"| {details['samples']:.0f} |"
                )
        if with_paper:
            paper_outcome = by_case[case_id].get(PAPER_METHOD)
            if case.paper is not None and paper_outcome is not None:
                paper_error = abs(paper_outcome.value - case.paper) / case.paper
                row += (
                    f" {case.paper:.9e} | {paper_error:.1e} "
                    f"{_mark(paper_error <= PAPER_TOLERANCE)} |"
                )
            else:
                row += " - | - |"
        if informative:
            row += f" {case.informative[1]:.9e} |" if case.informative else " - |"
        if mixed:
            row += f" {case.criterion.describe()} |"
        lines.append(row)
    return lines


def render_page(outcomes: Sequence[Outcome], report: Report = MAIN) -> str:
    """A full validation page."""
    parts = [
        f"# {report.title}",
        "",
        "Generated by `tools/validation_report.py` from "
        f"[CARA Analysis Tools](https://github.com/nasa/CARA_Analysis_Tools) "
        f"commit `{CARA_COMMIT[:12]}`. Do not edit by hand.",
        "",
        "## Summary",
        "",
        render_summary(outcomes),
        "",
    ]
    for suite, members in _groups(outcomes).items():
        parts += [f"## {suite}", ""]
        criteria = _criteria(members)
        if len(criteria) == 1:
            parts += [f"Criterion: {criteria[0]}.", ""]
        parts += [*_suite_table(members), ""]
    parts.append(report.not_compared)
    return "\n".join(parts)


def replace_summary(
    readme: str, summary: str, start: str = SUMMARY_START, end: str = SUMMARY_END
) -> str:
    """Put ``summary`` between the README's ``start`` and ``end`` markers.

    Raises
    ------
    ValueError
        If the markers are missing or out of order.
    """
    first = readme.find(start)
    last = readme.find(end)
    if first < 0 or last < first:
        raise ValueError(f"README needs {start} and {end}, in that order")
    head = readme[: first + len(start)]
    return f"{head}\n{summary}\n{readme[last:]}"


def rendered_files(
    outcomes: Sequence[Outcome], page: Path, readme: Path, report: Report = MAIN
) -> dict[Path, str]:
    """What the page and the README should contain for ``outcomes``."""
    summary = render_summary(outcomes)
    return {
        page: render_page(outcomes, report),
        readme: replace_summary(
            readme.read_text(encoding="utf-8"), summary, report.start, report.end
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Write (or with ``--check`` verify) a validation page and README summary.

    Returns
    -------
    int
        0 on success; 1 if a case fails or, with ``--check``, a file is stale.
    """
    parser = argparse.ArgumentParser(prog="tools/validation_report.py")
    parser.add_argument("--check", action="store_true", help="fail if out of date")
    parser.add_argument(
        "--monte-carlo", action="store_true", help="the Monte Carlo cases and page"
    )
    parser.add_argument("--page", type=Path, default=None)
    parser.add_argument("--readme", type=Path, default=REPO / "README.md")
    args = parser.parse_args(argv)

    report = MONTE_CARLO if args.monte_carlo else MAIN
    cases = load_monte_carlo() if args.monte_carlo else load_all()
    outcomes = evaluate(cases)
    page = args.page if args.page is not None else report.page
    status = 0
    for path, text in rendered_files(outcomes, page, args.readme, report).items():
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current == text:
            continue
        if args.check:
            print(f"{path} is out of date; run tools/validation_report.py")
            status = 1
        else:
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path}")
    failed = [o for o in outcomes if not o.passed]
    for outcome in failed:
        print(f"FAIL {outcome.case.suite} {outcome.case.case_id} {outcome.method}")
    return 1 if failed else status


if __name__ == "__main__":
    sys.exit(main())
