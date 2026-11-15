"""The validation reports and the README summaries, generated from CARA cases.

Usage: ``uv run python tools/validation_report.py [--monte-carlo] [--check]``.
Downloads the pinned CARA files, runs every case, and writes the results, the
page rendered from them and the README summary block: ``docs/validation.json``
and ``.md`` (2-D, 3-D and maximum Pc; a few minutes) or, with
``--monte-carlo``, ``docs/validation-monte-carlo.json`` and ``.md`` (up to
hours). Published results are kept while every fresh one reproduces them (see
:meth:`~keepout.validation.criteria.Criterion.reproduces`). ``--check`` fails
instead of writing when a file is out of date. Exits 1 if a case fails.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias, TypedDict, TypeVar

from generated import (
    README,
    REPO,
    generated_notice,
    parse_json,
    positive,
    readme_with,
    table,
    write_or_check,
)

from keepout.pc import METHODS
from keepout.pc.result import Method
from keepout.sources.cara import CARA_COMMIT
from keepout.validation.cases import (
    MC_MIN_PC,
    MC_SAMPLES,
    MC_SUITE_ALPHA,
    MC_TARGET_HITS,
    load_all,
    load_monte_carlo_cases,
)
from keepout.validation.criteria import (
    HALL,
    HALL_PC_SET,
    TWO_D,
    BoundCriterion,
    Criterion,
    IntervalCriterion,
)
from keepout.validation.evaluation import Outcome, evaluate

TOOL: Final = "tools/validation_report.py"

_A = TypeVar("_A")
_B = TypeVar("_B")

PAPER: Final = BoundCriterion(relative=1e-3)
"""Agreement with Alfano's published values (as in CARA's own tests). The
page states it under the table, apart from the suite's criterion against
CARA."""

PAPER_METHOD: Final[Method] = "foster"
"""Method compared with Alfano's published values."""

METHOD_TITLES: Final[Mapping[Method, str]] = {
    "foster": "Foster",
    "elrod": "Elrod",
    "hall3d": "Hall 3D-Nc",
    "montecarlo": "Monte Carlo",
    "frisbee": "Frisbee",
    "dilution": "Dilution",
    "alfano": "Alfano",
}
"""Column titles of the methods in the case tables."""

PC_FORMAT: Final = ".4e"
"""Format of the Pc values and intervals on the pages."""

CARA_REPOSITORY: Final = "https://github.com/nasa/CARA_Analysis_Tools"

CONJUNCTION_ID: Final = re.compile(r"\d{9}_conj_\d{9}_\d{8}_\d{6}_\d{8}_\d{6}")
"""A case id that is a conjunction id of CARA's Pc method test set."""

CONJUNCTION_ID_KEY: Final = (
    "Case: CARA's conjunction id, "
    "`<primary>_conj_<secondary>_<TCA date>_<TCA time>_<CDM date>_<CDM time>`: "
    "the catalog numbers of the two objects, then the TCA and the creation of "
    "the CDM (UTC, `YYYYMMDD_hhmmss`)."
)

INFORMATIVE_NOTES: Final[Mapping[str, str]] = {
    "Coppola 3-D Pc": (
        "Coppola 3-D Pc: CARA's Pc by another 3-D method (Coppola 2012), "
        "shown for information and not compared. It is not a reference for "
        "keepout: on some conjunctions it is orders of magnitude below the "
        "3D-Nc value, or 0, and CARA's own results table flags the cases where "
        "it underestimates."
    ),
}
"""What an informative column (:attr:`Row.informative`) shows, by its title."""

NOT_COMPARED: Final = """\
## Not compared

- **Alfano case 12, 2-D**: zero relative velocity leaves the encounter plane
  undefined; CARA gives no 2-D value and keepout raises
  [`GeometryError`](reference/errors.md#keepout.errors.GeometryError). Its 3-D
  and Monte Carlo values are compared.
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
  for information (see the note under that table).
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
  [Monte Carlo validation](validation-monte-carlo.md).
"""

NOT_COMPARED_MONTE_CARLO: Final = f"""\
## Not compared

- **53-CDM cases with a CARA Monte Carlo Pc below {MC_MIN_PC:.0e}.** keepout
  sizes a Monte Carlo case for about {MC_TARGET_HITS} expected collisions.
  Below a Pc of {MC_MIN_PC:.0e} that takes more than {MC_SAMPLES[1]:,} trials,
  the most a case is given, so these cases are left out. The
  [3-D validation](validation.md#cara-pc-method-test-set-3-d) compares Hall's
  3D-Nc with CARA on every case instead.
"""


@dataclass(frozen=True, slots=True)
class Report:
    """One validation page and its README summary block."""

    title: str
    page: Path
    start: str
    end: str
    not_compared: str
    introduction: Callable[[Record], list[str]]


def _verdict(rows: Sequence[Row]) -> str:
    """How many comparisons pass, in bold: ``**All 12 comparisons pass**``."""
    passed = sum(r.passed for r in rows)
    if passed == len(rows) == 1:
        return "**The one comparison passes**"
    if passed == len(rows):
        return f"**All {len(rows)} comparisons pass**"
    return (
        f"**{passed} of {len(rows)} comparisons pass** (the failures are marked "
        "**FAIL**)"
    )


def _cara_text(record: Record) -> str:
    return (
        f"[CARA Analysis Tools]({CARA_REPOSITORY}) (commit `{record.cara_commit[:12]}`)"
    )


def _main_introduction(record: Record) -> list[str]:
    """What the main page compares, its verdict and its tolerances."""
    return [
        (
            "NASA's Conjunction Assessment Risk Analysis (CARA) team publishes "
            f"{_cara_text(record)}: MATLAB implementations of the Pc methods with "
            "test cases and their results. This page runs keepout on those cases "
            "and compares its Pc with CARA's."
        ),
        "",
        (f"{_verdict(record.rows)}, each one case and one keepout method. The suites:"),
        "",
        (
            "- **Alfano (2009)**: the test cases of Alfano's paper, with CARA's "
            "2-D Pc, CARA's 3-D Pc and the values the paper publishes."
        ),
        (
            "- **CARA Pc method test set**: real conjunctions from CARA's test "
            "CDMs, with CARA's 2-D Pc and its 3-D Pc (Hall's 3D-Nc)."
        ),
        (
            "- **CARA unit tests**: the values CARA's MATLAB unit tests check, for "
            "the 2-D, 3-D and maximum-Pc methods."
        ),
        "",
        (
            "Worst relative error is the largest |keepout - CARA| / CARA over a "
            "suite's cases. Cases judged by an absolute error are left out of it."
        ),
        "",
        "The tolerances:",
        "",
        (
            f"- **2-D (Foster, Elrod)**: relative {TWO_D.relative:g}. Both codes "
            "compute the same integral, so they should agree to the precision of "
            f"the integration. Below a Pc of {TWO_D.small:g} the error is absolute, "
            f"{TWO_D.absolute:g}, as tiny values are integration noise."
        ),
        (
            f"- **3-D (Hall 3D-Nc)**: relative {HALL.relative:g}, the tolerance of "
            "CARA's own unit test of the method. On the Pc method test set it is "
            f"{HALL_PC_SET.relative:g}: there CARA's 3-D values sit systematically "
            "above its own 2-D values on nearly straight-line "
            "encounters (see [Validation](pc-3d.md#validation))."
        ),
        (
            "- **Maximum Pc (Frisbee, dilution)**: CARA's unit tests give these "
            "reference values rounded to two or three digits, so the tolerance is "
            "about half a unit in the last digit given: an absolute error for "
            "Frisbee, a relative one of 0.5% for dilution."
        ),
    ]


def _monte_carlo_introduction(record: Record) -> list[str]:
    """What the Monte Carlo page compares, its verdict and its criterion."""
    count = len(record.rows)
    confidence = 1.0 - MC_SUITE_ALPHA / max(count, 1)
    return [
        (
            f"{_verdict(record.rows)}: on every case CARA's Monte Carlo Pc "
            "lies inside keepout's confidence interval."
            if all(r.passed for r in record.rows)
            else f"{_verdict(record.rows)}."
        ),
        "",
        (
            "keepout's Monte Carlo Pc is compared with CARA's on the cases of "
            f"{_cara_text(record)} that have a Monte Carlo reference: Alfano's "
            "test cases and those of CARA's Pc method test set with a large "
            "enough Pc."
        ),
        "",
        (
            "Both values are estimates from random samples, so they never agree "
            "exactly. A case passes when CARA's value lies inside keepout's "
            f"confidence interval. Each interval is at {100 * confidence:.2f}% "
            f"confidence (1 - {MC_SUITE_ALPHA:g} / {count}): a correct keepout "
            f"fails any of the {count} cases with a chance of at most "
            f"{100 * MC_SUITE_ALPHA:g}%. keepout sizes each case for about "
            f"{MC_TARGET_HITS} expected collisions, so its own sampling error is "
            "about 5%; a difference of 5 to 10% from CARA is sampling noise, not "
            "disagreement."
        ),
    ]


MAIN: Final = Report(
    "Validation against NASA CARA",
    REPO / "docs/validation.md",
    "<!-- validation-summary:start -->",
    "<!-- validation-summary:end -->",
    NOT_COMPARED,
    _main_introduction,
)
MONTE_CARLO: Final = Report(
    "Monte Carlo validation against NASA CARA",
    REPO / "docs/validation-monte-carlo.md",
    "<!-- validation-monte-carlo-summary:start -->",
    "<!-- validation-monte-carlo-summary:end -->",
    NOT_COMPARED_MONTE_CARLO,
    _monte_carlo_introduction,
)


class CriterionJson(TypedDict):
    """A :class:`~keepout.validation.criteria.Criterion` as JSON."""

    relative: float | None
    absolute: float | None
    small: float
    interval: bool


def _criterion_json(criterion: Criterion) -> CriterionJson:
    """A criterion as JSON: its bounds, or ``interval`` true."""
    if isinstance(criterion, BoundCriterion):
        return {
            "relative": criterion.relative,
            "absolute": criterion.absolute,
            "small": criterion.small,
            "interval": False,
        }
    return {"relative": None, "absolute": None, "small": 0.0, "interval": True}


def _criterion(data: CriterionJson) -> Criterion:
    """The criterion :func:`_criterion_json` gave.

    Raises
    ------
    ValueError
        If the criterion is invalid, e.g. an interval with bounds.
    """
    if not data["interval"]:
        return BoundCriterion(data["relative"], data["absolute"], data["small"])
    if (data["relative"], data["absolute"], data["small"]) != (None, None, 0.0):
        raise ValueError("an interval criterion takes no error bounds")
    return IntervalCriterion()


class RowJson(TypedDict):
    """A :class:`Row` as JSON: its fields, with lists for the pairs."""

    suite: str
    case_id: str
    hard_body_radius: float
    reference: float
    criterion: CriterionJson
    method: Method
    value: float
    error: float
    relative: bool
    passed: bool
    paper: float | None
    interval: list[float] | None
    informative: list[str | float] | None
    ci: list[float] | None
    samples: int | None


class RecordJson(TypedDict):
    """A :class:`Record` as JSON."""

    cara_commit: str
    rows: list[RowJson]


def _list(pair: tuple[_A, _B] | None) -> list[_A | _B] | None:
    """A pair as a JSON list; ``None`` stays ``None``."""
    return None if pair is None else [pair[0], pair[1]]


def _pair(values: list[float] | None) -> tuple[float, float] | None:
    """A JSON pair of numbers as a tuple.

    Raises
    ------
    ValueError
        If ``values`` is not two numbers.
    """
    if values is None:
        return None
    if len(values) != 2:
        raise ValueError(f"expected a pair of numbers, got {values}")
    return values[0], values[1]


def _labelled(values: list[str | float] | None) -> tuple[str, float] | None:
    """A JSON ``[label, value]`` as a tuple.

    Raises
    ------
    ValueError
        If ``values`` is not a string and a number.
    """
    if values is None:
        return None
    if len(values) != 2:
        raise ValueError(f"expected [label, value], got {values}")
    label, value = values
    if not isinstance(label, str) or isinstance(value, str):
        raise ValueError(f"expected [label, value], got {values}")
    return label, value


@dataclass(frozen=True, slots=True)
class Row:
    """One method on one case as published: CARA's data and keepout's result.

    Parameters
    ----------
    suite, case_id, hard_body_radius, reference, criterion, paper, interval, informative
        As in :class:`~keepout.validation.cases.ValidationCase` (``interval``
        is CARA's).
    method
        The keepout method.
    value
        keepout's Pc.
    error, relative, passed
        The :class:`~keepout.validation.criteria.Verdict`.
    ci, samples
        keepout's Monte Carlo interval and trials, if any.
    """

    suite: str
    case_id: str
    hard_body_radius: float
    reference: float
    criterion: Criterion
    method: Method
    value: float
    error: float
    relative: bool
    passed: bool
    paper: float | None = None
    interval: tuple[float, float] | None = None
    informative: tuple[str, float] | None = None
    ci: tuple[float, float] | None = None
    samples: int | None = None

    @classmethod
    def of(cls, outcome: Outcome) -> Row:
        """The row of an evaluated case."""
        case, result, ci = outcome.case, outcome.result, outcome.result.interval
        return cls(
            suite=case.suite,
            case_id=case.case_id,
            hard_body_radius=case.conjunction.combined_hard_body_radius,
            reference=case.reference,
            criterion=case.criterion,
            method=result.method,
            value=result.value,
            error=outcome.verdict.error,
            relative=outcome.verdict.relative,
            passed=outcome.passed,
            paper=case.paper,
            interval=case.reference_interval,
            informative=case.informative,
            ci=ci,
            samples=int(result.details["samples"]) if ci else None,
        )

    @classmethod
    def from_json(cls, data: RowJson) -> Row:
        """The row :meth:`to_json` gave (checked by :func:`generated.parse_json`).

        Raises
        ------
        ValueError
            If a pair is not two numbers, or the criterion is invalid.
        """
        return cls(
            suite=data["suite"],
            case_id=data["case_id"],
            hard_body_radius=data["hard_body_radius"],
            reference=data["reference"],
            criterion=_criterion(data["criterion"]),
            method=data["method"],
            value=data["value"],
            error=data["error"],
            relative=data["relative"],
            passed=data["passed"],
            paper=data["paper"],
            interval=_pair(data["interval"]),
            informative=_labelled(data["informative"]),
            ci=_pair(data["ci"]),
            samples=data["samples"],
        )

    def to_json(self) -> RowJson:
        """The row as JSON values."""
        return {
            "suite": self.suite,
            "case_id": self.case_id,
            "hard_body_radius": self.hard_body_radius,
            "reference": self.reference,
            "criterion": _criterion_json(self.criterion),
            "method": self.method,
            "value": self.value,
            "error": self.error,
            "relative": self.relative,
            "passed": self.passed,
            "paper": self.paper,
            "interval": _list(self.interval),
            "informative": _list(self.informative),
            "ci": _list(self.ci),
            "samples": self.samples,
        }

    def difference(self, previous: Row) -> str | None:
        """Why this row does not repeat ``previous`` (``None`` if it does).

        It repeats it with the same case, method, CARA data and verdict, and
        keepout's values within the criterion's reproducibility.
        """
        settled = dataclasses.replace(
            previous, value=self.value, error=self.error, ci=self.ci
        )
        if settled != self:
            return "CARA data or verdict changed"
        values = [(previous.value, self.value)]
        if previous.ci is not None and self.ci is not None:
            values += list(zip(previous.ci, self.ci, strict=True))
        for old, new in values:
            if not self.criterion.reproduces(old, new, self.reference, previous.ci):
                return f"{old:.9e} -> {new:.9e}"
        return None


@dataclass(frozen=True, slots=True)
class Record:
    """A validation run as published: the CARA commit and every row."""

    cara_commit: str
    rows: tuple[Row, ...]

    @classmethod
    def of(cls, outcomes: Sequence[Outcome]) -> Record:
        """The record of evaluated cases at the pinned CARA commit."""
        return cls(CARA_COMMIT, tuple(Row.of(o) for o in outcomes))

    @classmethod
    def loads(cls, text: str, source: str = "validation results") -> Record:
        """The record :meth:`dumps` gave.

        Raises
        ------
        ValueError
            If ``text`` is not such a record; the message names ``source``.
        """
        data = parse_json(text, RecordJson, source)
        return cls(data["cara_commit"], tuple(Row.from_json(r) for r in data["rows"]))

    def dumps(self) -> str:
        """The record as JSON text."""
        data: RecordJson = {
            "cara_commit": self.cara_commit,
            "rows": [row.to_json() for row in self.rows],
        }
        return json.dumps(data, indent=1) + "\n"

    def differences(self, previous: Record) -> list[str]:
        """Why this record does not reproduce ``previous`` (empty if it does)."""
        if self.cara_commit != previous.cara_commit:
            return [f"CARA commit {previous.cara_commit} -> {self.cara_commit}"]
        keys = [_key(r) for r in self.rows]
        if keys != [_key(r) for r in previous.rows]:
            return ["the cases or methods changed"]
        found = [
            (_key(new), new.difference(old))
            for old, new in zip(previous.rows, self.rows, strict=True)
        ]
        return [f"{' '.join(key)}: {why}" for key, why in found if why is not None]


def _key(row: Row) -> tuple[str, str, str]:
    return row.suite, row.case_id, row.method


def _pc(value: float) -> str:
    return f"{value:{PC_FORMAT}}"


def _error(row: Row) -> str:
    return f"{row.error:.1e}" if row.relative else f"{row.error:.1e} (abs)"


def _mark(*, passed: bool) -> str:
    return "pass" if passed else "**FAIL**"


def _worst(rows: Sequence[Row]) -> str:
    relative = [r.error for r in rows if r.relative]
    return f"{max(relative):.1e}" if relative else "-"


def _groups(rows: Sequence[Row]) -> dict[str, list[Row]]:
    suites: defaultdict[str, list[Row]] = defaultdict(list)
    for row in rows:
        suites[row.suite].append(row)
    return dict(suites)


def render_summary(rows: Sequence[Row]) -> str:
    """Markdown table: per suite and method, cases passed and worst error."""
    cells = []
    for suite, members in _groups(rows).items():
        for method in dict.fromkeys(r.method for r in members):
            mine = [r for r in members if r.method == method]
            passed = sum(r.passed for r in mine)
            cells.append((suite, method, f"{passed} / {len(mine)}", _worst(mine)))
    header = ("Suite", "Method", "Cases passed", "Worst relative error")
    return "\n".join(table(header, cells))


def _criteria(members: Sequence[Row]) -> list[str]:
    return list(dict.fromkeys(r.criterion.describe() for r in members))


Cells: TypeAlias = Callable[[Row, Mapping[Method, Row]], list[str]]
"""The cells of one column group for a case (given by its first row), given
its rows by method."""


@dataclass(frozen=True, slots=True)
class _Columns:
    """Adjacent table columns: their titles and how to fill them for a case."""

    titles: tuple[str, ...]
    cells: Cells


def _method_columns(method: Method) -> _Columns:
    monte_carlo = METHODS[method].kind == "montecarlo"
    titles: tuple[str, ...] = (METHOD_TITLES[method], "error")
    if monte_carlo:
        titles += ("keepout interval", "trials")

    def cells(_: Row, rows: Mapping[Method, Row]) -> list[str]:
        found = rows.get(method)
        if found is None:
            return ["-"] * len(titles)
        values = [_pc(found.value), f"{_error(found)} {_mark(passed=found.passed)}"]
        if monte_carlo:
            low, high = found.ci or (0.0, 0.0)
            values += [f"[{_pc(low)}, {_pc(high)}]", f"{found.samples}"]
        return values

    return _Columns(titles, cells)


def _paper_cells(case: Row, rows: Mapping[Method, Row]) -> list[str]:
    found = rows.get(PAPER_METHOD)
    if case.paper is None or found is None:
        return ["-", "-"]
    error, _, passed = PAPER.judge(found.value, case.paper)
    return [_pc(case.paper), f"{error:.1e} {_mark(passed=passed)}"]


def _suite_table(members: Sequence[Row]) -> list[str]:
    cases: dict[str, Row] = {}
    by_case: defaultdict[str, dict[Method, Row]] = defaultdict(dict)
    for row in members:
        cases.setdefault(row.case_id, row)
        by_case[row.case_id][row.method] = row
    every = list(cases.values())
    columns = [
        _Columns(
            ("Case", "HBR [m]", "CARA Pc"),
            lambda case, _: [
                case.case_id,
                f"{case.hard_body_radius:g}",
                _pc(case.reference),
            ],
        )
    ]
    if any(c.interval is not None for c in every):
        columns.append(
            _Columns(
                ("CARA 95% interval",),
                lambda case, _: [
                    f"[{_pc(case.interval[0])}, {_pc(case.interval[1])}]"
                    if case.interval
                    else "-"
                ],
            )
        )
    methods: dict[Method, None] = dict.fromkeys(r.method for r in members)
    columns += [_method_columns(m) for m in methods]
    if any(c.paper is not None for c in every):
        columns.append(_Columns(("Alfano paper", "vs paper"), _paper_cells))
    informative = next((c.informative[0] for c in every if c.informative), None)
    if informative:
        columns.append(
            _Columns(
                (informative,),
                lambda case, _: [_pc(case.informative[1]) if case.informative else "-"],
            )
        )
    if len(_criteria(members)) > 1:
        columns.append(
            _Columns(("Criterion",), lambda case, _: [case.criterion.describe()])
        )
    titles = [title for group in columns for title in group.titles]
    return table(
        titles,
        [
            [cell for g in columns for cell in g.cells(case, by_case[case_id])]
            for case_id, case in cases.items()
        ],
    )


def _notes(members: Sequence[Row]) -> list[str]:
    """What a suite's table needs explained below it: the paper's columns and
    an informative column."""
    notes: list[str] = []
    if any(r.paper is not None for r in members):
        notes.append(
            "Alfano paper: the Pc Alfano (2009) publishes for the case. vs paper: "
            f"keepout's {METHOD_TITLES[PAPER_METHOD]} Pc against it, judged by "
            f"{PAPER.describe()} (as CARA's own tests compare with the paper)."
        )
    titles = dict.fromkeys(r.informative[0] for r in members if r.informative)
    notes += [INFORMATIVE_NOTES[t] for t in titles if t in INFORMATIVE_NOTES]
    return [line for note in notes for line in ("", note)]


def render_page(record: Record, report: Report = MAIN) -> str:
    """A full validation page."""
    parts = [
        f"# {report.title}",
        "",
        generated_notice(TOOL),
        "",
        *report.introduction(record),
        "",
        "## Summary",
        "",
        render_summary(record.rows),
        "",
    ]
    keyed = False
    for suite, members in _groups(record.rows).items():
        parts += [f"## {suite}", ""]
        criteria = _criteria(members)
        if len(criteria) == 1:
            parts += [f"Criterion: {criteria[0]}.", ""]
        if not keyed and any(CONJUNCTION_ID.fullmatch(r.case_id) for r in members):
            parts += [CONJUNCTION_ID_KEY, ""]
            keyed = True
        parts += [*_suite_table(members), *_notes(members), ""]
    parts.append(report.not_compared)
    return "\n".join(parts)


def published(fresh: Record, data: Path) -> tuple[Record, list[str]]:
    """The record to publish, and why ``fresh`` replaces the one at ``data``.

    The published record is kept while ``fresh`` reproduces it.
    """
    if not data.exists():
        return fresh, [f"{data} is missing"]
    previous = Record.loads(data.read_text(encoding="utf-8"), str(data))
    differences = fresh.differences(previous)
    return (fresh if differences else previous), differences


def rendered_files(
    record: Record, data: Path, page: Path, readme: Path, report: Report = MAIN
) -> dict[Path, str]:
    """What the results, the page and the README should contain for ``record``."""
    summary = render_summary(record.rows)
    return {
        data: record.dumps(),
        page: render_page(record, report),
        readme: readme_with(readme, report.start, report.end, summary),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """The entry point writing (or with ``--check`` verifying) a page and summary.

    Returns
    -------
    int
        0 on success; 1 if a case fails or, with ``--check``, a file is stale.
    """
    parser = argparse.ArgumentParser(prog=TOOL)
    parser.add_argument("--check", action="store_true", help="fail if out of date")
    parser.add_argument(
        "--monte-carlo", action="store_true", help="the Monte Carlo cases and page"
    )
    parser.add_argument(
        "--page", type=Path, default=None, help="the page; its results go next to it"
    )
    parser.add_argument("--readme", type=Path, default=README)
    parser.add_argument(
        "--workers",
        type=positive,
        default=None,
        help="processes to run the cases in (default: one per CPU, at most 4)",
    )
    args = parser.parse_args(argv)

    report = MONTE_CARLO if args.monte_carlo else MAIN
    cases = load_monte_carlo_cases() if args.monte_carlo else load_all()
    fresh = Record.of(evaluate(cases, workers=args.workers))
    page = args.page if args.page is not None else report.page
    data = page.with_suffix(".json")
    record, differences = published(fresh, data)
    files = rendered_files(record, data, page, args.readme, report)
    status = write_or_check(files, check=args.check, tool=TOOL)
    for difference in differences:
        print(f"not reproduced: {difference}")
    failed = [r for r in fresh.rows if not r.passed]
    for row in failed:
        print(f"FAIL {row.suite} {row.case_id} {row.method}")
    return 1 if failed else status


if __name__ == "__main__":
    sys.exit(main())
