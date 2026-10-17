"""Compare keepout's screening with CelesTrak's SOCRATES and write the report.

Usage: ``uv run python tools/screening_report.py [--check] [--workers N]``.
Screens the objects of the committed SOCRATES excerpt (``tests/screening/data/``,
see ``tools/screening_snapshot.py``) with :func:`keepout.screening.screen` over
SOCRATES's week and with its defaults (the 5 km sphere SOCRATES also uses),
matches every approach of the excerpt with one of keepout's, and writes
``docs/screening-socrates.json`` (the numbers), ``docs/screening-socrates.md``
(the page) and the README summary block. Needs no network. ``--check`` fails
instead of writing when a file is out of date. Exits 1 if an approach differs
from SOCRATES beyond the tolerances (:data:`TOLERANCES`) without a known cause
(:data:`KNOWN_DIFFERENCES`), or a known cause no longer shows.

Each approach depends only on its own pair's element sets (see
:mod:`keepout.screening.pipeline`), so screening the excerpt's objects alone
finds, for the pairs SOCRATES lists, the same approaches as screening the
whole catalog, in seconds instead of minutes.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Literal, TypeAlias, TypedDict, get_args

import screening_snapshot as snapshot
from astropy.time import Time
from generated import (
    README,
    REPO,
    json_text,
    plural,
    positive,
    readme_with,
    relative,
    table,
    write_or_check,
)
from screening_snapshot import SocratesSummary

from keepout.omm import Omm
from keepout.screening import DEFAULT_VOLUME, CloseApproach, ScreeningResult, screen
from keepout.screening.pipeline import DEFAULT_DURATION, MAX_AGE, MAX_LEAD
from keepout.sources.socrates import SocratesApproach, read_socrates
from keepout.units import DAY, KM, seconds_between

PAGE: Final = REPO / "docs/screening-socrates.md"
START: Final = "<!-- screening-socrates-summary:start -->"
END: Final = "<!-- screening-socrates-summary:end -->"
TOOL: Final = "tools/screening_report.py"

FAST: Final = snapshot.FAST
"""Approaches at this relative speed or more [m/s] (SOCRATES's value) are
*fast*; the others, mostly formation flyers, *slow*."""

MATCH_WINDOW: Final = 60.0
"""A keepout approach of the same pair is SOCRATES's approach when their TCAs
are at most this far apart [s] (the nearest one; a pair's minima are minutes
apart)."""

LOOKUP_RADIUS: Final = 50e3
"""Radius of the sphere [m] in which keepout looks, around SOCRATES's TCA,
for its closest approach of a pair it did not report."""

Group: TypeAlias = Literal["fast", "slow"]
"""The kind of an approach (:data:`FAST`)."""

GROUPS: Final[tuple[Group, ...]] = get_args(Group)


@dataclass(frozen=True, slots=True)
class Tolerance:
    """How far keepout's values of an approach may be from SOCRATES's.

    Parameters
    ----------
    tca
        TCA [s].
    miss_distance
        Range at TCA [m].
    relative_speed
        Relative speed at TCA [m/s].
    """

    tca: float
    miss_distance: float
    relative_speed: float


TOLERANCES: Final[Mapping[Group, Tolerance]] = {
    "fast": Tolerance(tca=1e-3, miss_distance=0.501, relative_speed=0.501),
    "slow": Tolerance(tca=0.5, miss_distance=0.501, relative_speed=0.501),
}
"""SOCRATES prints TCA to the millisecond, the range in km and the speed in
km/s with three decimals: rounding alone leaves 0.5 ms, 0.5 m and 0.5 m/s.
Range and speed get 1 mm (per second) more, keepout's own accuracy target;
the TCA of a fast approach 0.5 ms more, of a slow one 0.5 s, as the range of
a slow pair hardly changes for seconds around its minimum (the report gives
the range change over each difference)."""

KNOWN_DIFFERENCES: Final[Mapping[int, str]] = {
    47834: (
        "STARLINK-2424 (47834) has one comparable approach, with FENGYUN 1C DEB "
        "(38513), and it differs; the ten other comparable approaches of 38513 "
        "agree. SOCRATES's element set of 47834 has the snapshot's epoch to the "
        "43 s that DSE resolves, but evidently other values."
    ),
    100379: (
        "All four comparable approaches of STARLINK-38284 (100379) differ, by a "
        "TCA growing with the time since its epoch (0.21 s after 6.3 days, 0.31 s "
        "after 7.8 days). A B* 3.4 % lower than the snapshot's reproduces all four "
        "to 2.3 ms and 2.8 m: SOCRATES used an element set of the same epoch, to "
        "the 43 s that DSE resolves, with other values."
    ),
}
"""Objects whose element set in the snapshot is evidently not the one
SOCRATES used, though the epochs match: why their approaches differ."""


@dataclass(frozen=True, slots=True)
class Inputs:
    """What the report compares.

    Parameters
    ----------
    approaches
        The SOCRATES excerpt, in file order.
    catalog
        The snapshot's element sets.
    socrates
        Counts and times of the whole SOCRATES report (from the manifest).
    retrieved
        The day the data was retrieved (UTC, ISO 8601).
    """

    approaches: tuple[SocratesApproach, ...]
    catalog: tuple[Omm, ...]
    socrates: SocratesSummary
    retrieved: str


def read_inputs(data: Path) -> Inputs:
    """The excerpt, the snapshot and the manifest's facts in ``data``."""
    manifest = snapshot.read_manifest(data)
    return Inputs(
        approaches=read_socrates(snapshot.read_excerpt(data)),
        catalog=snapshot.read_catalog(data),
        socrates=manifest["socrates"],
        retrieved=snapshot.retrieval_day(manifest),
    )


def objects(approaches: Sequence[SocratesApproach]) -> set[int]:
    """The catalog numbers of the objects of ``approaches``."""
    return {a.primary for a in approaches} | {a.secondary for a in approaches}


def group(approach: SocratesApproach) -> Group:
    """Whether SOCRATES's approach is fast or slow (:data:`FAST`)."""
    return "fast" if approach.relative_speed >= FAST else "slow"


def window_start(approaches: Sequence[SocratesApproach]) -> Time:
    """0 h UTC of the day of the first TCA: where SOCRATES's week starts."""
    first = min(approaches, key=lambda a: (a.tca.jd1, a.tca.jd2))
    return Time(first.tca.isot[:10], scale="utc")


@dataclass(frozen=True, slots=True)
class Match:
    """A SOCRATES approach and keepout's.

    Parameters
    ----------
    socrates
        SOCRATES's approach.
    keepout
        keepout's approach of the same pair whose TCA is the nearest within
        :data:`MATCH_WINDOW`, if any.
    """

    socrates: SocratesApproach
    keepout: CloseApproach | None

    @property
    def tca_difference(self) -> float:
        """TCA by keepout minus TCA by SOCRATES [s] (NaN if keepout has none)."""
        if self.keepout is None:
            return float("nan")
        return seconds_between(self.socrates.tca, self.keepout.tca)

    @property
    def miss_distance_difference(self) -> float:
        """Range by keepout minus range by SOCRATES [m] (NaN if keepout has
        none)."""
        if self.keepout is None:
            return float("nan")
        return self.keepout.miss_distance - self.socrates.miss_distance

    @property
    def relative_speed_difference(self) -> float:
        """Relative speed by keepout minus by SOCRATES [m/s] (NaN if keepout
        has none)."""
        if self.keepout is None:
            return float("nan")
        return self.keepout.relative_speed - self.socrates.relative_speed

    @property
    def range_change(self) -> float:
        """How much the range changes over the TCA difference [m]:
        ``v^2 dt^2 / (2 rho)`` at keepout's minimum (NaN if keepout has none
        or the range is 0)."""
        if self.keepout is None or self.keepout.miss_distance == 0.0:
            return float("nan")
        speed, distance = self.keepout.relative_speed, self.keepout.miss_distance
        return speed**2 * self.tca_difference**2 / (2.0 * distance)

    def agrees(self) -> bool:
        """Whether keepout found it within its group's tolerances."""
        if self.keepout is None:
            return False
        tolerance = TOLERANCES[group(self.socrates)]
        return (
            abs(self.tca_difference) <= tolerance.tca
            and abs(self.miss_distance_difference) <= tolerance.miss_distance
            and abs(self.relative_speed_difference) <= tolerance.relative_speed
        )


@dataclass(frozen=True, slots=True)
class Comparison:
    """SOCRATES's approaches against keepout's.

    Parameters
    ----------
    matches
        Every SOCRATES approach whose objects were both screened, in file
        order.
    skipped
        The SOCRATES approaches with an object :func:`screen` left out, in
        file order.
    keepout_only
        keepout's approaches of the pairs SOCRATES lists that match none of
        its approaches, in chronological order.
    result
        keepout's screening.
    primaries
        Objects screened as primaries (SOCRATES's first objects).
    """

    matches: tuple[Match, ...]
    skipped: tuple[SocratesApproach, ...]
    keepout_only: tuple[CloseApproach, ...]
    result: ScreeningResult
    primaries: int


def _pair_of(approach: CloseApproach | SocratesApproach) -> tuple[int, int]:
    if isinstance(approach, SocratesApproach):
        return approach.primary, approach.secondary
    return approach.primary_number, approach.secondary_number


def compare(
    approaches: Sequence[SocratesApproach],
    catalog: Sequence[Omm],
    start: Time,
    *,
    workers: int = 1,
) -> Comparison:
    """Screen the objects of ``approaches`` for SOCRATES's week and match.

    SOCRATES's first objects are the primaries, every object of
    ``approaches`` the catalog. Each SOCRATES approach takes keepout's
    approach of the same pair with the nearest TCA within
    :data:`MATCH_WINDOW`, the nearest candidates first, each keepout
    approach at most once.
    """
    by_number = {omm.catalog_number: omm for omm in catalog}
    primary_numbers = sorted({a.primary for a in approaches})
    result = screen(
        [by_number[number] for number in primary_numbers],
        [by_number[number] for number in sorted(objects(approaches))],
        start,
        DEFAULT_DURATION,
        DEFAULT_VOLUME,
        workers=workers,
    )
    left_out = {item.catalog_number for item in result.skipped}
    compared = [
        a
        for a in approaches
        if a.primary not in left_out and a.secondary not in left_out
    ]
    pairs = {_pair_of(a) for a in compared}
    found = [a for a in result.approaches if _pair_of(a) in pairs]
    chosen = _nearest_matches(compared, found, start)
    taken = set(chosen.values())
    return Comparison(
        matches=tuple(
            Match(a, found[chosen[i]] if i in chosen else None)
            for i, a in enumerate(compared)
        ),
        skipped=tuple(
            a for a in approaches if a.primary in left_out or a.secondary in left_out
        ),
        keepout_only=tuple(a for p, a in enumerate(found) if p not in taken),
        result=result,
        primaries=len(primary_numbers),
    )


def _nearest_matches(
    socrates: Sequence[SocratesApproach], found: Sequence[CloseApproach], start: Time
) -> dict[int, int]:
    """Index in ``found`` matched to each index in ``socrates`` that has one.

    Candidates are approaches of the same pair within :data:`MATCH_WINDOW`,
    taken nearest first; each of ``found`` is matched at most once.
    """
    by_pair: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    for position, approach in enumerate(found):
        by_pair[_pair_of(approach)].append(position)
    found_at = [seconds_between(start, approach.tca) for approach in found]
    socrates_at = [seconds_between(start, approach.tca) for approach in socrates]
    candidates = sorted(
        (abs(found_at[position] - socrates_at[index]), index, position)
        for index, approach in enumerate(socrates)
        for position in by_pair[_pair_of(approach)]
    )
    chosen: dict[int, int] = {}
    taken: set[int] = set()
    for gap, index, position in candidates:
        if gap <= MATCH_WINDOW and index not in chosen and position not in taken:
            chosen[index] = position
            taken.add(position)
    return chosen


class Worst(TypedDict):
    """The largest differences of the approaches within tolerance."""

    tca: float
    miss_distance: float
    relative_speed: float
    range_change: float


class GroupSummary(TypedDict):
    """Fast or slow approaches."""

    compared: int
    agree: int
    worst: Worst | None


class Values(TypedDict):
    """keepout's values of an approach."""

    tca_difference: float
    miss_distance: float
    relative_speed: float


class Difference(TypedDict):
    """A SOCRATES approach keepout does not match within tolerance."""

    primary: int
    secondary: int
    tca: str
    group: Group
    miss_distance: float
    relative_speed: float
    keepout: Values | None
    reported: bool
    cause: int | None


class KeepoutOnly(TypedDict):
    """A keepout approach of a pair SOCRATES lists, matching none of its."""

    primary: int
    secondary: int
    tca: str
    miss_distance: float
    relative_speed: float
    nearest_other: float | None


class Skipped(TypedDict):
    """A SOCRATES approach with an object :func:`screen` left out."""

    primary: int
    secondary: int
    tca: str
    object: int
    reason: str
    detail: str


class NotCompared(TypedDict):
    """SOCRATES approaches not compared."""

    outside_catalog: int
    other_element_sets: int
    skipped: list[Skipped]


class Settings(TypedDict):
    """How keepout screened."""

    start: str
    days: float
    volume: str
    max_age_days: float
    max_lead_days: float
    match_window: float
    fast_from: float
    primaries: int
    objects: int


class Summary(TypedDict):
    """The report's numbers, as written to the JSON file."""

    retrieved: str
    socrates: SocratesSummary
    settings: Settings
    tolerances: dict[Group, dict[str, float]]
    groups: dict[Group, GroupSummary]
    differences: list[Difference]
    keepout_only: list[KeepoutOnly]
    not_compared: NotCompared


def _round_s(seconds: float) -> float:
    return round(seconds, 6)


def _round_m(metres: float) -> float:
    return round(metres, 5)


def _values(approach: CloseApproach, tca: Time) -> Values:
    return {
        "tca_difference": _round_s(seconds_between(tca, approach.tca)),
        "miss_distance": _round_m(approach.miss_distance),
        "relative_speed": _round_m(approach.relative_speed),
    }


def _cause(approach: SocratesApproach, known: Mapping[int, str]) -> int | None:
    return next((n for n in (approach.primary, approach.secondary) if n in known), None)


def _group_summary(matches: Sequence[Match]) -> GroupSummary:
    agree = [m for m in matches if m.agrees()]
    worst: Worst | None = None
    if agree:
        worst = {
            "tca": _round_s(max(abs(m.tca_difference) for m in agree)),
            "miss_distance": _round_m(
                max(abs(m.miss_distance_difference) for m in agree)
            ),
            "relative_speed": _round_m(
                max(abs(m.relative_speed_difference) for m in agree)
            ),
            "range_change": float(f"{max(m.range_change for m in agree):.3g}"),
        }
    return {"compared": len(matches), "agree": len(agree), "worst": worst}


def _nearest_other(
    approach: CloseApproach, of_pair: Sequence[CloseApproach]
) -> float | None:
    """Seconds to the nearest other approach of ``of_pair``, the approaches of
    ``approach``'s pair."""
    gaps = [
        abs(seconds_between(approach.tca, other.tca))
        for other in of_pair
        if other is not approach
    ]
    return round(min(gaps), 3) if gaps else None


def _list_differences(
    comparison: Comparison, inputs: Inputs, known: Mapping[int, str]
) -> list[Difference]:
    by_number = {omm.catalog_number: omm for omm in inputs.catalog}
    differences: list[Difference] = []
    for match in comparison.matches:
        if match.agrees():
            continue
        a = match.socrates
        keepout = match.keepout or snapshot.nearest_approach(
            by_number[a.primary],
            by_number[a.secondary],
            a.tca,
            window=MATCH_WINDOW,
            radius=LOOKUP_RADIUS,
        )
        differences.append(
            {
                "primary": a.primary,
                "secondary": a.secondary,
                "tca": a.tca.iso,
                "group": group(a),
                "miss_distance": a.miss_distance,
                "relative_speed": a.relative_speed,
                "keepout": None if keepout is None else _values(keepout, a.tca),
                "reported": match.keepout is not None,
                "cause": _cause(a, known),
            }
        )
    return differences


def _list_keepout_only(comparison: Comparison) -> list[KeepoutOnly]:
    by_pair: dict[tuple[int, int], list[CloseApproach]] = {}
    for approach in comparison.result.approaches:
        by_pair.setdefault(_pair_of(approach), []).append(approach)
    return [
        {
            "primary": a.primary_number,
            "secondary": a.secondary_number,
            "tca": a.tca.iso,
            "miss_distance": _round_m(a.miss_distance),
            "relative_speed": _round_m(a.relative_speed),
            "nearest_other": _nearest_other(a, by_pair[_pair_of(a)]),
        }
        for a in comparison.keepout_only
    ]


def _list_skipped(comparison: Comparison) -> list[Skipped]:
    left_out = {item.catalog_number: item for item in comparison.result.skipped}
    skipped: list[Skipped] = []
    for a in comparison.skipped:
        number = a.primary if a.primary in left_out else a.secondary
        skipped.append(
            {
                "primary": a.primary,
                "secondary": a.secondary,
                "tca": a.tca.iso,
                "object": number,
                "reason": left_out[number].reason,
                "detail": left_out[number].detail,
            }
        )
    return skipped


def _settings(comparison: Comparison, inputs: Inputs, start: Time) -> Settings:
    return {
        "start": start.isot,
        "days": DEFAULT_DURATION / DAY,
        "volume": DEFAULT_VOLUME.label,
        "max_age_days": MAX_AGE / DAY,
        "max_lead_days": MAX_LEAD / DAY,
        "match_window": MATCH_WINDOW,
        "fast_from": FAST,
        "primaries": comparison.primaries,
        "objects": len(objects(inputs.approaches)),
    }


def summarise(
    comparison: Comparison,
    inputs: Inputs,
    start: Time,
    known: Mapping[int, str] = KNOWN_DIFFERENCES,
) -> Summary:
    """The report's numbers."""
    whole = inputs.socrates
    return {
        "retrieved": inputs.retrieved,
        "socrates": whole,
        "settings": _settings(comparison, inputs, start),
        "tolerances": {
            name: asdict(tolerance) for name, tolerance in TOLERANCES.items()
        },
        "groups": {
            name: _group_summary(
                [m for m in comparison.matches if group(m.socrates) == name]
            )
            for name in GROUPS
        },
        "differences": _list_differences(comparison, inputs, known),
        "keepout_only": _list_keepout_only(comparison),
        "not_compared": {
            "outside_catalog": whole["approaches"] - whole["both_in_catalog"],
            "other_element_sets": whole["both_in_catalog"] - whole["comparable"],
            "skipped": _list_skipped(comparison),
        },
    }


def known_differences(approaches: Sequence[SocratesApproach]) -> dict[int, str]:
    """The :data:`KNOWN_DIFFERENCES` of the objects of ``approaches``."""
    numbers = objects(approaches)
    return {n: text for n, text in KNOWN_DIFFERENCES.items() if n in numbers}


def problems(
    summary: Summary, known: Mapping[int, str] = KNOWN_DIFFERENCES
) -> list[str]:
    """Differences without a known cause, and known causes that no longer
    show."""
    found: list[str] = [
        f"{d['primary']}-{d['secondary']} at {d['tca']} differs from SOCRATES "
        "beyond the tolerances"
        for d in summary["differences"]
        if d["cause"] is None
    ]
    causes = {d["cause"] for d in summary["differences"]}
    found += [
        f"{number} is a known difference but none of its approaches differs"
        for number in known
        if number not in causes
    ]
    return found


GROUP_TITLES: Final[Mapping[Group, str]] = {
    "fast": "fast (>= 1 km/s)",
    "slow": "slow (< 1 km/s)",
}


def _ms(seconds: float) -> str:
    return f"{seconds * 1e3:.3f} ms"


def _m(metres: float) -> str:
    return f"{metres:.3f} m"


def _speed(speed: float) -> str:
    return f"{speed:.3f} m/s"


def _pair(primary: int, secondary: int) -> str:
    return f"{primary} - {secondary}"


AGREEMENT_HEADER: Final = (
    "Approaches",
    "Compared",
    "Within tolerance",
    "Worst TCA",
    "Worst range",
    "Worst relative speed",
    "Range change over the TCA difference",
)
"""The columns of the agreement table; the README shows the first five."""


def _agreement(summary: Summary, columns: int = len(AGREEMENT_HEADER)) -> list[str]:
    """The agreement table, its first ``columns`` columns."""
    rows: list[tuple[object, ...]] = []
    for name, numbers in summary["groups"].items():
        worst = numbers["worst"]
        cells: list[object] = ["-"] * 4
        if worst is not None:
            cells = [
                _ms(worst["tca"]),
                f"{worst['miss_distance']:.5f} m",
                f"{worst['relative_speed']:.5f} m/s",
                f"{worst['range_change']:.2g} m",
            ]
        rows.append((GROUP_TITLES[name], numbers["compared"], numbers["agree"], *cells))
    return table(AGREEMENT_HEADER[:columns], [row[:columns] for row in rows])


def render_summary(summary: Summary, page: str) -> str:
    """The README block."""
    socrates = summary["socrates"]
    differences = summary["differences"]
    causes = {d["cause"] for d in differences if d["cause"] is not None}
    unexplained = sum(d["cause"] is None for d in differences)
    notes = []
    if causes:
        explained = plural(len(differences) - unexplained, "approach", "approaches")
        notes.append(
            f"{explained} outside tolerance involve "
            f"{plural(len(causes), 'object')} whose element sets "
            "evidently differ from SOCRATES's."
        )
    if unexplained:
        notes.append(
            f"{plural(unexplained, 'approach differs', 'approaches differ')} "
            "without a known cause."
        )
    if summary["keepout_only"]:
        minima = plural(len(summary["keepout_only"]), "minimum", "minima")
        notes.append(
            f"keepout also reports {minima} of these pairs that SOCRATES does not list."
        )
    return "\n".join(
        [
            (
                "Close approaches from `screen` against CelesTrak's SOCRATES, on the "
                f"{socrates['comparable']} SOCRATES approaches computed from the same "
                f"element sets ([report]({page})):"
            ),
            "",
            *_agreement(summary, columns=5),
            *(["", " ".join(notes)] if notes else []),
        ]
    )


def _differences(summary: Summary, known: Mapping[int, str]) -> list[str]:
    if not summary["differences"]:
        return ["None."]
    rows = []
    for d in summary["differences"]:
        keepout = d["keepout"]
        if keepout is None:
            cells: list[object] = ["-", "none within 50 km", "-"]
        else:
            distance = _m(keepout["miss_distance"])
            if not d["reported"]:
                distance += " (not reported: outside the volume)"
            cells = [
                _ms(keepout["tca_difference"]),
                distance,
                _speed(keepout["relative_speed"]),
            ]
        rows.append(
            (
                _pair(d["primary"], d["secondary"]),
                d["tca"],
                f"{d['miss_distance']:.0f} m",
                f"{d['relative_speed']:.0f} m/s",
                *cells,
                "unexplained" if d["cause"] is None else d["cause"],
            )
        )
    causes = sorted(
        {d["cause"] for d in summary["differences"] if d["cause"] is not None}
    )
    return [
        *table(
            (
                "Pair",
                "SOCRATES TCA (UTC)",
                "SOCRATES range",
                "SOCRATES speed",
                "keepout TCA - SOCRATES",
                "keepout range",
                "keepout speed",
                "Cause",
            ),
            rows,
        ),
        "",
        *[f"- **{number}**: {known[number]}" for number in causes],
    ]


def _keepout_only(summary: Summary) -> list[str]:
    if not summary["keepout_only"]:
        return ["None."]
    return [
        (
            "Formation flyers can have two range minima per orbit, minutes apart, "
            "with a maximum at most a few metres higher between them. keepout reports "
            "every local minimum inside the volume; SOCRATES lists only one of the two "
            "once they come close. The last column gives the time to the nearest "
            "other approach keepout reports for the pair."
        ),
        "",
        *table(
            ("Pair", "TCA (UTC)", "Range", "Relative speed", "Nearest other approach"),
            [
                (
                    _pair(a["primary"], a["secondary"]),
                    a["tca"],
                    _m(a["miss_distance"]),
                    _speed(a["relative_speed"]),
                    "-"
                    if a["nearest_other"] is None
                    else f"{a['nearest_other'] / 60:.1f} min",
                )
                for a in summary["keepout_only"]
            ],
        ),
    ]


def _skipped(summary: Summary) -> list[str]:
    skipped = summary["not_compared"]["skipped"]
    if not skipped:
        return []
    return [
        "",
        (
            f"{plural(len(skipped), 'approach has', 'approaches have')} an object "
            "`screen` left out, with its defaults:"
        ),
        "",
        *table(
            ("Pair", "SOCRATES TCA (UTC)", "Object", "Reason", "Detail"),
            [
                (
                    _pair(s["primary"], s["secondary"]),
                    s["tca"],
                    s["object"],
                    s["reason"],
                    s["detail"],
                )
                for s in skipped
            ],
        ),
    ]


def _introduction(summary: Summary) -> list[str]:
    settings = summary["settings"]
    socrates = summary["socrates"]
    return [
        (
            "CelesTrak's [SOCRATES](https://celestrak.org/SOCRATES/) lists, for a "
            "week, the approaches within 5 km of the objects it screens, from SGP4 "
            "on the GP element sets of its time. keepout screened the same element "
            "sets with `keepout.screening.screen` and its defaults: the "
            f"{settings['volume']}, {settings['days']:g} days from "
            f"{settings['start'][:16].replace('T', ' ')} UTC (SOCRATES's window), "
            f"element sets at most {settings['max_age_days']:g} days older than the "
            f"window's end and {settings['max_lead_days']:g} days newer than its "
            "start."
        ),
        "",
        (
            "SOCRATES gives each object's epoch only as days from it to TCA (DSE, "
            "three decimals: 43 s), and CelesTrak serves only the latest element "
            f"sets. Of the {socrates['approaches']} approaches of SOCRATES's report "
            f"(TCA {socrates['first_tca'][:16].replace('T', ' ')} to "
            f"{socrates['last_tca'][:16].replace('T', ' ')} UTC), "
            f"{socrates['comparable']} have both objects in the public catalog with "
            "the epochs SOCRATES used; these are compared. SOCRATES's "
            f"{settings['primaries']} first objects are the primaries, its "
            f"{settings['objects']} objects the catalog (an approach depends on its "
            "own pair only, so the rest of the catalog changes nothing). Each "
            "SOCRATES approach is matched with keepout's approach of the same pair "
            f"whose TCA is nearest, within {settings['match_window']:g} s. "
            f"*Fast* approaches are those of {settings['fast_from'] / KM:g} km/s or "
            "more by SOCRATES; the others, mostly formation flyers, are *slow*."
        ),
        "",
        (
            "Data: USSPACECOM/18 SDS via Space-Track.org, retrieved from CelesTrak "
            f"(celestrak.org) on {summary['retrieved']}; the public catalog and the "
            "SOCRATES report sorted by minimum range (see "
            "`tests/screening/data/NOTICE`)."
        ),
    ]


def _tolerances(summary: Summary) -> list[str]:
    slow = summary["groups"]["slow"]["worst"]
    reason = (
        "SOCRATES prints TCA (UTC) to the millisecond, `TCA_RANGE` in km and "
        "`TCA_RELATIVE_SPEED` in km/s with three decimals, so rounding alone "
        "leaves up to 0.5 ms, 0.5 m and 0.5 m/s; read in these units, every "
        "approach within tolerance agrees to half the last printed digit. Range "
        "and speed are allowed 1 mm and 1 mm/s more, keepout's accuracy target. "
        "A fast approach's TCA is allowed 1 ms: the rounding, and as much again "
        "for the two TCA searches. A slow pair's range hardly changes for "
        "seconds around its minimum, which leaves its TCA ill-conditioned: it "
        "is allowed 0.5 s"
    )
    if slow is not None:
        reason += (
            f", and moving TCA by the differences found changes the range by "
            f"{slow['range_change']:.2g} m at most"
        )
    return [
        *table(
            ("Approaches", "TCA", "Range", "Relative speed"),
            [
                (
                    GROUP_TITLES[name],
                    f"{tolerance['tca'] * 1e3:g} ms",
                    f"{tolerance['miss_distance']:g} m",
                    f"{tolerance['relative_speed']:g} m/s",
                )
                for name, tolerance in summary["tolerances"].items()
            ],
        ),
        "",
        reason + ".",
    ]


def _not_compared(summary: Summary) -> list[str]:
    not_compared = summary["not_compared"]
    return [
        (
            f"- **{not_compared['outside_catalog']} approaches with an object outside "
            "the public catalog**: SOCRATES screens the whole catalog of "
            "Space-Track.org; CelesTrak publishes only part of it."
        ),
        (
            f"- **{not_compared['other_element_sets']} approaches whose objects are "
            "in the public catalog with other element sets**: the snapshot was "
            "downloaded hours after SOCRATES's data cut-off, and most objects had new "
            "element sets by then."
        ),
        (
            "- **Pairs SOCRATES does not list.** SOCRATES does not report every pair "
            "that comes within 5 km: TerraSAR-X (31698) and TanDEM-X (36605), for "
            "example, fly in formation a few hundred metres apart and are not in its "
            "report. keepout's approaches are compared only for the pairs SOCRATES "
            "lists."
        ),
        *_skipped(summary),
    ]


def render_page(summary: Summary, known: Mapping[int, str] = KNOWN_DIFFERENCES) -> str:
    """The report page."""
    lines = [
        "# Screening against CelesTrak's SOCRATES",
        "",
        "<!-- Generated by tools/screening_report.py; do not edit. -->",
        "",
        *_introduction(summary),
        "",
        "## Agreement",
        "",
        *_agreement(summary),
        "",
        (
            "Worst: the largest difference between keepout and SOCRATES among the "
            "approaches within tolerance; the last column is how much the range "
            "changes when TCA moves by the TCA difference (v² Δt² / 2r, with r "
            "the range at keepout's minimum). The approaches outside tolerance are "
            "under [Differences](#differences)."
        ),
        "",
        "## Tolerances",
        "",
        *_tolerances(summary),
        "",
        "## Differences",
        "",
        *_differences(summary, known),
        "",
        "## Found by keepout only",
        "",
        *_keepout_only(summary),
        "",
        "## Not compared",
        "",
        *_not_compared(summary),
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Write (or with ``--check`` verify) the SOCRATES screening report.

    Returns
    -------
    int
        0 on success; 1 if, with ``--check``, a file is out of date, or if an
        approach differs without a known cause (:func:`problems`).
    """
    parser = argparse.ArgumentParser(prog=TOOL)
    parser.add_argument("--check", action="store_true", help="fail if out of date")
    parser.add_argument("--data", type=Path, default=snapshot.DATA)
    parser.add_argument("--page", type=Path, default=PAGE)
    parser.add_argument("--readme", type=Path, default=README)
    parser.add_argument(
        "--workers", type=positive, default=4, help="processes to screen with"
    )
    args = parser.parse_args(argv)

    inputs = read_inputs(args.data)
    start = window_start(inputs.approaches)
    comparison = compare(inputs.approaches, inputs.catalog, start, workers=args.workers)
    known = known_differences(inputs.approaches)
    summary = summarise(comparison, inputs, start, known)
    page = relative(args.page, args.readme.parent)
    files = {
        args.page.with_suffix(".json"): json_text(summary),
        args.page: render_page(summary, known),
        args.readme: readme_with(
            args.readme, START, END, render_summary(summary, page)
        ),
    }
    status = write_or_check(files, check=args.check, tool=TOOL)
    for problem in problems(summary, known):
        print(problem)
        status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
