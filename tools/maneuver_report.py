"""The manoeuvre report: the Pc of along-track burns on a CARA test event, and
a re-screen of burns on the frozen catalog.

Usage: ``uv run python tools/maneuver_report.py [--check] [--keep-timings]
[--cdm PATH] [--hbr METRES] [--workers N] [--repeat N]``.

Plans the burns of CARA test CDM :data:`CDM_ID` (Metop-B against a Fengyun
1C fragment, from :data:`keepout.sources.cara.PC_METHOD_CDMS`, downloaded on
demand at :data:`~keepout.sources.cara.CARA_COMMIT`; ``--cdm`` reads another
file instead) with :func:`keepout.maneuver.plan_maneuver`, the combined
hard-body radius CARA's own test table gives the CDM (``HBR_m`` of
:data:`keepout.sources.cara.PC_METHOD_TABLE`, 10 m; ``--hbr`` gives another)
and the default grid and target. Then runs the example synthetic operator on
the frozen public catalog (as ``tools/synthetic_operator_report.py``), plans
its yellow event of the highest Pc with the target ``green_below`` and
re-screens each option against the catalog with
:func:`keepout.maneuver.rescreen`: a demonstration of the mechanism, not a
recommendation.

Writes ``docs/maneuver-report.json`` (the numbers), ``docs/maneuver-report.md``
(the page), the heatmap ``docs/img/maneuver/heatmap.svg`` and the README
summary block. The plan of the CARA event is timed over ``--repeat`` runs
(3): the range of wall and CPU times, with the CPU model. ``--check`` fails
instead of writing when the numbers, the page or the README are out of date;
it plans once and takes the timings from the committed JSON, so they are
shown but never checked, and the heatmap is then only required to exist (SVG
bytes vary with the matplotlib version). ``--keep-timings`` writes the files
with the committed JSON's timings instead of measuring again, for a change of
the text alone. CARA's files are not redistributed:
only numbers derived from them and the figure are written.
"""

from __future__ import annotations

import argparse
import datetime
import math
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, TypedDict

import matplotlib as mpl
import numpy as np
import screening_snapshot as snapshot
from astropy.time import Time
from generated import (
    README,
    REPO,
    generated_notice,
    json_text,
    minute_text,
    positive,
    read_json,
    readme_with,
    relative,
    second_text,
    significant,
    table,
    threshold_settings,
    threshold_text,
    write_or_check,
)
from matplotlib.colors import Colormap, LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from screening_benchmark import cpu_model
from synthetic_operator_report import FLEET, level_key, shown, window_start

from keepout._defaults import TCA_TOLERANCE
from keepout._methods import Method
from keepout._text import pc_text, plural
from keepout._workers import DEFAULT_WORKERS, ONE_THREAD
from keepout.cdm import Cdm, load
from keepout.events import ConjunctionEvent, group_events, time_to_tca
from keepout.figures import (
    LOWEST_LOG_PC,
    NO_PC,
    SEQUENTIAL,
    TARGET_COLOUR,
    grid_figure,
)
from keepout.maneuver import (
    DEFAULT_BURN_GRID,
    TARGET_PC,
    ManeuverOption,
    ManeuverPlan,
    NewApproach,
    RescreenResult,
    judge,
    plan_maneuver,
    rescreen,
)
from keepout.risk import RiskLevel, Thresholds
from keepout.screening import DEFAULT_DURATION, DEFAULT_VOLUME
from keepout.sources import cara
from keepout.synthetic import OperatorEvent, OperatorResult, load_fleet
from keepout.synthetic.operator import run
from keepout.units import DAY, HOUR, MINUTE
from keepout.validation.cara_inputs import column_number

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh

    from keepout.figures import GridFigure, GridPanel
    from keepout.omm import Omm

PAGE: Final = REPO / "docs/maneuver-report.md"
IMAGES: Final = REPO / "docs/img/maneuver"
HEATMAP: Final = "heatmap.svg"
START: Final = "<!-- maneuver-report-summary:start -->"
END: Final = "<!-- maneuver-report-summary:end -->"
TOOL: Final = "tools/maneuver_report.py"

CDM_ID: Final = "000038771_conj_000030802_20201216_182131_20201215_171306"
"""The CARA test CDM of the heatmap: Metop-B (38771) against 30802."""

HBR_COLUMN: Final = "HBR_m"
"""The column of :data:`~keepout.sources.cara.PC_METHOD_TABLE` that gives a
CDM's combined hard-body radius [m]."""

THRESHOLDS: Final = Thresholds()
DEMONSTRATION_TARGET: Final = THRESHOLDS.green_below
"""The target of the re-screen demonstration: green."""

TOLERANCE: Final = f"{TCA_TOLERANCE / MINUTE:g} minutes"
"""How far apart the TCAs of one approach may be, for the page."""

INK: Final = "#0b0b0b"
MUTED: Final = "#52514e"
RULE: Final = "#c3c2b7"


class Option(TypedDict):
    """A burn, as the report writes it."""

    burn_lead_hours: float
    delta_v: float
    pc: float
    range_at_tca: float
    method: Method
    meets_target: bool


class Plan(TypedDict):
    """An event's plan, as the report writes it."""

    event: str
    primary: str
    primary_name: str
    secondary: str
    secondary_name: str
    tca: str
    created: str
    hours_to_tca: float
    relative_speed: float
    primary_drag_term: float | None
    target_pc: float
    baseline: Option
    burn_times: int
    cells: int
    cells_without_pc: int
    below_target: dict[str, int]
    prograde: Option | None
    retrograde: Option | None
    notes: list[str]


class Grid(TypedDict):
    """The grid of burns."""

    earliest_lead_hours: float
    latest_lead_hours: float
    steps_per_orbit: int
    smallest_delta_v: float
    largest_delta_v: float
    delta_v_per_direction: int


class Approach(TypedDict):
    """A new or a closer approach of a re-screen."""

    secondary: int
    tca: str
    range_at_tca: float
    relative_speed: float
    pc: float
    level: str


class Rescreen(TypedDict):
    """The re-screen of one option."""

    direction: str
    start: str
    end: str
    catalog_objects: int
    skipped_objects: int
    secondary_screened: bool
    event_range: float | None
    conflict: bool
    clean: bool
    """No approach at red and no event return."""
    new_approaches: list[Approach]
    closer_approaches: list[Approach]
    event_returns: list[Approach]
    """The event's pair back away from its TCA at or above the target."""


class Demonstration(TypedDict):
    """The re-screen demonstration on the frozen catalog."""

    fleet: str
    retrieved: str
    start: str
    days: float
    volume: str
    pc: float | None
    level: str | None
    action: str
    plan: Plan | None
    rescreens: list[Rescreen]
    recommendable: bool
    """Whether ``keepout maneuver`` would recommend a burn (exit status 0)."""


class Settings(TypedDict):
    """The report's settings."""

    cdm: str
    cara_commit: str
    hard_body_radius: float
    hard_body_radius_source: Literal["cara", "given"]
    """CARA's test table, or ``--hbr``."""
    target_pc: float
    grid: Grid
    green_below: float
    red_from: float
    basis: str


class Timings(TypedDict):
    """The time of the CARA event's plan and where it was measured: shown,
    never checked."""

    cpu_model: str
    logical_cpus: int
    python: str
    date: str
    one_thread: bool
    runs: int
    wall: list[float]
    cpu: list[float]


class Summary(TypedDict):
    """The report's numbers, as written to the JSON file."""

    settings: Settings
    event: Plan
    demonstration: Demonstration | None
    timings: Timings | None


def _time_text(time: Time) -> str:
    """A time in UTC to the millisecond: the microseconds of a computed time
    (a TCA, a burn) differ with the machine's floating-point library."""
    text: str = Time(time, precision=3).utc.isot
    return text


def _round_hours(seconds: float) -> float:
    return round(seconds / HOUR, 2)


def option_summary(option: ManeuverOption) -> Option:
    """A burn, rounded as the report writes it."""
    return {
        "burn_lead_hours": _round_hours(option.burn_lead),
        "delta_v": significant(option.delta_v),
        "pc": significant(option.pc),
        "range_at_tca": round(option.range_at_tca, 1),
        "method": option.method,
        "meets_target": option.meets_target,
    }


def _optional(option: ManeuverOption | None) -> Option | None:
    return None if option is None else option_summary(option)


def plan_summary(plan: ManeuverPlan) -> Plan:
    """A plan's numbers."""
    cdm = plan.event.latest
    below = plan.pc < plan.target_pc
    return {
        "event": plan.event.event_id,
        "primary": cdm.object1.object_id,
        "primary_name": cdm.object1.name,
        "secondary": cdm.object2.object_id,
        "secondary_name": cdm.object2.name,
        "tca": _time_text(cdm.tca),
        "created": _time_text(cdm.creation_date),
        "hours_to_tca": _round_hours(time_to_tca(cdm)),
        "relative_speed": round(plan.conjunction.relative_speed, 3),
        "primary_drag_term": cdm.object1.cd_area_over_mass,
        "target_pc": plan.target_pc,
        "baseline": option_summary(plan.baseline),
        "burn_times": len(plan.burn_leads),
        "cells": int(plan.pc.size),
        "cells_without_pc": int(np.count_nonzero(np.isnan(plan.pc))),
        "below_target": {
            "prograde": int(np.count_nonzero(below[:, plan.delta_v > 0.0])),
            "retrograde": int(np.count_nonzero(below[:, plan.delta_v < 0.0])),
        },
        "prograde": _optional(plan.prograde),
        "retrograde": _optional(plan.retrograde),
        "notes": list(plan.notes),
    }


def grid_settings() -> Grid:
    """The default grid of burns."""
    grid = DEFAULT_BURN_GRID
    return {
        "earliest_lead_hours": _round_hours(grid.earliest_lead),
        "latest_lead_hours": _round_hours(grid.latest_lead),
        "steps_per_orbit": grid.steps_per_orbit,
        "smallest_delta_v": grid.smallest_delta_v,
        "largest_delta_v": grid.largest_delta_v,
        "delta_v_per_direction": grid.delta_v_per_direction,
    }


def _approach(item: NewApproach) -> Approach:
    approach = item.approach
    return {
        "secondary": approach.secondary_number,
        "tca": _time_text(approach.tca),
        "range_at_tca": round(approach.range_at_tca, 1),
        "relative_speed": round(approach.relative_speed, 3),
        "pc": significant(item.pc),
        "level": item.level.key,
    }


def rescreen_summary(result: RescreenResult) -> Rescreen:
    """A re-screen's numbers."""
    start, end = result.window
    return {
        "direction": str(result.option.direction),
        "start": _time_text(start),
        "end": _time_text(end),
        "catalog_objects": result.catalog_objects,
        "skipped_objects": result.skipped_objects,
        "secondary_screened": result.secondary_screened,
        "event_range": None
        if result.event_range is None
        else round(result.event_range, 1),
        "conflict": result.conflict,
        "clean": result.clean,
        "new_approaches": [_approach(item) for item in result.new_approaches],
        "closer_approaches": [_approach(item) for item in result.closer_approaches],
        "event_returns": [_approach(item) for item in result.event_returns],
    }


def demonstration_event(result: OperatorResult) -> OperatorEvent | None:
    """The yellow event of the highest Pc (the earliest of equals)."""
    yellow = [
        event
        for event in result.events
        if event.assessment.level is RiskLevel.YELLOW and event.observation.pc
    ]
    return max(yellow, key=lambda event: event.observation.pc or 0.0, default=None)


def demonstrate(
    catalog: Sequence[Omm], start: Time, fleet: Path, workers: int
) -> tuple[OperatorEvent | None, ManeuverPlan | None, list[RescreenResult]]:
    """The demonstration: the operator's run from ``start``, its yellow event
    of the highest Pc, that event's plan and the re-screen of each option."""
    result = run(
        load_fleet(fleet),
        catalog,
        start,
        DEFAULT_DURATION,
        DEFAULT_VOLUME,
        workers=workers,
        thresholds=THRESHOLDS,
    )
    chosen = demonstration_event(result)
    if chosen is None:
        return None, None, []
    plan = plan_maneuver(
        chosen.event, target_pc=DEMONSTRATION_TARGET, thresholds=THRESHOLDS
    )
    results = [rescreen(plan, item, catalog, workers=workers) for item in plan.options]
    return chosen, plan, results


def timed_plan(
    event: ConjunctionEvent, radius: float, repeat: int
) -> tuple[ManeuverPlan, list[float], list[float]]:
    """The event's plan, made ``repeat`` times (at least once), with the
    lowest and highest wall and CPU time of the runs [s], to 0.1 s."""

    def timed() -> tuple[ManeuverPlan, float, float]:
        wall, cpu = time.perf_counter(), time.process_time()
        plan = plan_maneuver(
            event, combined_hard_body_radius=radius, thresholds=THRESHOLDS
        )
        return plan, time.perf_counter() - wall, time.process_time() - cpu

    runs = [timed() for _ in range(max(repeat, 1))]
    plan = runs[-1][0]
    walls = [wall for _, wall, _ in runs]
    cpus = [cpu for _, _, cpu in runs]
    return (
        plan,
        [round(min(walls), 1), round(max(walls), 1)],
        [round(min(cpus), 1), round(max(cpus), 1)],
    )


def cara_radius(table: Path, conjunction_id: str) -> float:
    """The combined hard-body radius [m] CARA's test table gives a CDM.

    Raises
    ------
    ValueError
        If the table has no row of ``conjunction_id``.
    CaraFormatError
        If its radius is not a number.
    """
    for row in cara.load_xlsx(table):
        if row.get("Conjunction_ID") == conjunction_id:
            return column_number(row, HBR_COLUMN)
    raise ValueError(f"{table} has no row {conjunction_id}")


def delta_v_text(delta_v: float) -> str:
    """A signed Δv to 3 significant figures, trailing zeros kept: ``-0.00200``."""
    return f"{delta_v:+#.3g}"


def primary_text(plan: Plan) -> str:
    """The primary of a plan, named as the page names it."""
    return _object(plan["primary"], plan["primary_name"])


def _level_text(settings: Settings, plan: Plan) -> str:
    thresholds = Thresholds(
        green_below=settings["green_below"], red_from=settings["red_from"]
    )
    return thresholds.level(plan["baseline"]["pc"]).key


def _alt_text(plan: Plan) -> str:
    return f"Pc after an along-track burn of {primary_text(plan)}"


def _option_row(name: str, option: Option | None) -> tuple[str, ...]:
    if option is None:
        return (name, "-", "-", "none on the grid", "-", "-")
    if option["delta_v"] == 0.0:
        lead = delta_v = "-"
    else:
        lead = f"{option['burn_lead_hours']:.1f} h"
        delta_v = f"{delta_v_text(option['delta_v'])} m/s"
    return (
        name,
        lead,
        delta_v,
        pc_text(option["pc"]),
        f"{option['range_at_tca']:.0f} m",
        option["method"],
    )


def options_table(plan: Plan) -> list[str]:
    """The baseline and the minimum burn each way."""
    return table(
        ("Burn", "Before TCA", "Δv", "Pc", "Range at TCA", "Method"),
        [
            _option_row("none", plan["baseline"]),
            _option_row("retrograde", plan["retrograde"]),
            _option_row("prograde", plan["prograde"]),
        ],
    )


DIRECTION_NOTE: Final = re.compile(r"(?P<direction>prograde|retrograde): (?P<note>.*)")
"""A plan's note about one direction's option, as the planner words it."""


def _caveat_groups(notes: Sequence[str]) -> dict[str, list[tuple[str, str]]]:
    """The notes by their lead (the text before the first colon, or the whole
    note), each with what it is about (the event, or a direction's option)
    and its text."""
    groups: dict[str, list[tuple[str, str]]] = {}
    for note in notes:
        match = DIRECTION_NOTE.fullmatch(note)
        about, text = (
            (match["direction"], match["note"]) if match else ("without a burn", note)
        )
        lead = text.split(": ", 1)[0]
        groups.setdefault(lead, []).append((about, text))
    return groups


def _caveats(plan: Plan) -> list[str]:
    """The plan's notes as a list, a note repeated for several options given
    once with what differs."""
    if not plan["notes"]:
        return []
    lines = ["", "Caveats:", ""]
    for lead, members in _caveat_groups(plan["notes"]).items():
        if len(members) == 1:
            about, text = members[0]
            prefix = "" if about == "without a burn" else f"{about}: "
            lines.append(f"- {prefix}{text}.")
            continue
        details = [
            text.split(": ", 1)[1] if ": " in text else text for _, text in members
        ]
        ends = {detail.rsplit("; ", 1)[-1] for detail in details}
        shared = ends.pop() if len(ends) == 1 and "; " in details[0] else None
        if shared is not None:
            details = [detail.rsplit("; ", 1)[0] for detail in details]
        lines.append(f"- {lead}" + (f" ({shared})" if shared else "") + ":")
        lines += [
            f"  - {about}: {detail}."
            for (about, _), detail in zip(members, details, strict=True)
        ]
    return lines


def _object(identifier: str, name: str) -> str:
    return f"{name} ({identifier.lstrip('0') or '0'})" if name else identifier


def _event_table(plan: Plan) -> list[str]:
    drag = plan["primary_drag_term"]
    return table(
        ("", ""),
        [
            ("Primary (burns)", primary_text(plan)),
            ("Secondary", _object(plan["secondary"], plan["secondary_name"])),
            ("TCA (UTC)", second_text(plan["tca"])),
            (
                "CDM created (UTC)",
                (
                    f"{second_text(plan['created'])}, "
                    f"{plan['hours_to_tca']:.1f} h before TCA"
                ),
            ),
            ("Range at TCA", f"{plan['baseline']['range_at_tca']:.1f} m"),
            ("Relative speed", f"{plan['relative_speed']:.1f} m/s"),
            (
                "Primary's CD·A/m",
                "not given: drag off" if drag is None else f"{drag:g} m²/kg",
            ),
        ],
    )


def _approach_text(a: Approach) -> str:
    return (
        f"range {a['range_at_tca']:.0f} m, {a['relative_speed']:.1f} m/s, assumed "
        f"Pc {pc_text(a['pc'])}, {a['level']}"
    )


def _approach_lines(kind: str, approaches: Sequence[Approach]) -> list[str]:
    return [
        f"  - {kind}: {a['secondary']} at {minute_text(a['tca'])} UTC, "
        + _approach_text(a)
        for a in approaches
    ]


def _return_lines(item: Rescreen, pair: str, target: float) -> list[str]:
    return [
        (
            f"  - **back above the target**: the burn brings {pair} back together "
            f"at {minute_text(a['tca'])} UTC, {_approach_text(a)}: above the "
            f"target {pc_text(target)}, so this option is not clean"
        )
        for a in item["event_returns"]
    ]


def _key(approach: Approach) -> tuple[int, str]:
    return approach["secondary"], approach["tca"]


def _rescreen_lines(
    item: Rescreen, secondary: str, pair: str, target: float
) -> list[str]:
    view = (
        "on the element sets (SGP4, not the CDM's orbit) the event's "
        f"secondary {secondary}"
    )
    if not item["secondary_screened"]:
        event = f"the event's secondary {secondary} was not screened"
    elif item["event_range"] is None:
        event = f"{view} has no approach within {TOLERANCE} of the CDM's TCA"
    else:
        event = f"{view} passes at {item['event_range']:.0f} m"
    # A return of the pair is also a new or closer approach: shown once.
    returns = {_key(a) for a in item["event_returns"]}
    new = [a for a in item["new_approaches"] if _key(a) not in returns]
    closer = [a for a in item["closer_approaches"] if _key(a) not in returns]
    if returns and not new and not closer:
        found = "no other new or closer approach"
    else:
        other = "other " if returns else ""
        found = (
            f"{plural(len(new), f'{other}new approach', f'{other}new approaches')} "
            f"and {plural(len(closer), 'closer approach', 'closer approaches')}"
        )
    status = "clean" if item["clean"] else "not clean"
    listed = bool(returns or new or closer)
    return [
        (
            f"- **{item['direction']}** ({status}): {item['catalog_objects']} "
            f"objects screened from {minute_text(item['start'])} to "
            f"{minute_text(item['end'])} UTC, "
            f"{plural(item['skipped_objects'], 'catalog object')} left out; {event}; "
            f"{found}{', one at red' if item['conflict'] else ', none at red'}"
            f"{':' if listed else '.'}"
        ),
        *_return_lines(item, pair, target),
        *_approach_lines("new", new),
        *_approach_lines("closer", closer),
    ]


def _outcome(demo: Demonstration) -> list[str]:
    """What the re-screens make of the options, as ``keepout maneuver`` does."""
    rescreens = demo["rescreens"]
    returning = [item["direction"] for item in rescreens if item["event_returns"]]
    if not returning:
        return []
    several = len(returning) > 1
    burns = f"{' and the '.join(returning)} burn{'s' if several else ''}"
    clean = [item["direction"] for item in rescreens if item["clean"]]
    others = [item["direction"] for item in rescreens if not item["clean"]]
    if demo["recommendable"]:
        verdict = (
            f"Only the {' and the '.join(clean)} burn "
            f"{'is' if len(clean) == 1 else 'are'} clean; `keepout maneuver` "
            f"still lists the {' and the '.join(others)} "
            f"{'one' if len(others) == 1 else 'ones'}, marked not clean."
        )
    else:
        verdict = (
            "No burn is clean, so `keepout maneuver` exits with status 4: on "
            "this re-screen no burn it found clears the event and stays clear of "
            "the pair."
        )
    return [
        "",
        (
            f"{verdict} The {burns} clear{'' if several else 's'} the event at its "
            "TCA, but the two objects meet again hours later, above the target. "
            "The planner looks for the new TCA only near the old one (see the "
            "slow-encounter caveat above), so only the re-screen finds such a "
            "return."
        ),
    ]


def _demonstration(summary: Summary) -> list[str]:
    demo = summary["demonstration"]
    lines = [
        '!!! note "A demonstration of the mechanism, not a recommendation"',
        (
            "    This section shows what a re-screen does, on an event whose "
            "covariances are keepout's assumed model of GP element-set errors. "
            "Nothing here is advice about these satellites, and keepout is not "
            "their operator."
        ),
        "",
    ]
    if demo is None:
        return [*lines, "Not run."]
    settings = summary["settings"]
    lines.append(
        "The [example synthetic operator](synthetic-operator.md) screened its "
        f"fleet (`{demo['fleet']}`) against the public catalog as CelesTrak "
        f"served it on {demo['retrieved']} (a copy kept in the repository, so "
        f"the run repeats exactly), {demo['days']:g} days from "
        f"{minute_text(demo['start'])} UTC with the {demo['volume']}. Its "
        "highest-Pc yellow event is planned here, with the target set to the "
        f"green threshold ({settings['green_below']:.0e}), and each option is "
        "re-screened against the same catalog for "
        f"{DEFAULT_DURATION / DAY:g} days from the burn."
    )
    plan = demo["plan"]
    if plan is None:
        return [*lines, "", "The run has no yellow event."]
    lines += [
        "",
        (
            f"{primary_text(plan)} and "
            f"{_object(plan['secondary'], plan['secondary_name'])}: Pc "
            f"{pc_text(demo['pc'])}, {demo['level']}, action {demo['action']} "
            f"(event `{plan['event']}`)."
        ),
        "",
        *_event_table(plan),
        "",
        *options_table(plan),
        *_caveats(plan),
        "",
        "### Re-screen",
        "",
    ]
    if not demo["rescreens"]:
        return [*lines, "No option to re-screen."]
    secondary = _object(plan["secondary"], plan["secondary_name"])
    pair = f"{plan['primary_name']} and {plan['secondary_name']}"
    for item in demo["rescreens"]:
        lines += _rescreen_lines(item, secondary, pair, plan["target_pc"])
    lines += _outcome(demo)
    lines += [
        "",
        (
            "A new approach is one the run without the burn does not have within "
            f"{TOLERANCE} of its TCA; a closer approach is one it has at a larger "
            "range. Their Pc is the synthetic operator's (`auto`) on its "
            "covariance model, an assumption. The re-screen searches for no "
            "other option: a burn that clears one event but brings another is "
            "for the analyst to weigh (see "
            "[Manoeuvre planning](maneuver.md#re-screen))."
        ),
    ]
    return lines


def _span(values: Sequence[float]) -> str:
    """A range of times: ``11.9 to 16.0 s``, or ``12.0 s`` if it is one."""
    low, high = min(values), max(values)
    return f"{low:.1f} s" if low == high else f"{low:.1f} to {high:.1f} s"


def radius_text(settings: Settings, *, short: bool = False) -> str:
    """Where the combined hard-body radius comes from: for the page, or
    ``short`` for the README."""
    radius = f"HBR {settings['hard_body_radius']:g} m"
    if settings["hard_body_radius_source"] == "given":
        return f"{radius}, given with `--hbr`"
    if short:
        return f"{radius}, CARA's value"
    return (
        f"{radius}, CARA's value for this CDM (column `{HBR_COLUMN}` of "
        f"`{cara.PC_METHOD_TABLE.path}` at the same commit)"
    )


def _timing_line(timings: Timings | None, plan: Plan, *, check_note: bool) -> str:
    """The plan's timings; ``check_note`` adds that ``--check`` ignores them."""
    if timings is None:
        return "Not measured yet."
    threads = "one thread" if timings["one_thread"] else "their default threads"
    runs = timings["runs"]
    return (
        f"The plan of the {plan['cells']} cells took {_span(timings['wall'])} "
        f"(CPU time {_span(timings['cpu'])}, numerical libraries on {threads}) "
        f"{'in 1 run' if runs == 1 else f'over {runs} runs'} "
        f"on {timings['date']}, {timings['cpu_model']} "
        f"({timings['logical_cpus']} logical CPUs), Python {timings['python']}"
        + ("; `--check` checks the numbers, not the times." if check_note else ".")
    )


def _caption(summary: Summary) -> str:
    settings, plan = summary["settings"], summary["event"]
    missing = plan["cells_without_pc"]
    gaps = (
        f" Grey: {plural(missing, 'cell')} without a closest approach to trust, "
        "never an option."
        if missing
        else ""
    )
    drag = (
        "no drag (the CDM gives the primary no CD·A/m)"
        if plan["primary_drag_term"] is None
        else "drag from the CDM's CD·A/m"
    )
    return (
        "Foster's 2-D Pc after an impulsive along-track burn of the primary, by "
        "burn size (log scale; retrograde to the left) and hours before TCA, "
        f"{plan['burn_times']} burn times by "
        f"{2 * settings['grid']['delta_v_per_direction']} Δv. The CDM's "
        f"covariances are used unchanged, with HBR "
        f"{settings['hard_body_radius']:g} m; gravity "
        f"to J6 and {drag}. Orange: the target "
        f"{settings['target_pc']:.0e}; dots: the smallest burn each way that "
        "meets it, confirmed with the event's own method." + gaps
    )


def render_page(summary: Summary, images: str) -> str:
    """The report page, with the heatmap under ``images``."""
    settings, plan = summary["settings"], summary["event"]
    grid = settings["grid"]
    commit = settings["cara_commit"]
    lines = [
        "# Manoeuvre planning on a CARA test event",
        "",
        generated_notice(TOOL),
        "",
        (
            "keepout's [manoeuvre planner](maneuver.md) on CARA test CDM "
            f"`{settings['cdm']}` from the "
            "[NASA CARA Analysis Tools](https://github.com/nasa/CARA_Analysis_Tools) "
            f"(commit `{commit[:12]}`): a {plan['tca'][:4]} test conjunction of "
            f"{primary_text(plan)} with "
            f"{_object(plan['secondary'], plan['secondary_name'])} at "
            f"{plan['relative_speed']:.0f} m/s, {_level_text(settings, plan)} "
            "without a burn. keepout is not "
            f"{primary_text(plan)}'s operator, and nothing here is advice about "
            "it. "
            f"Combined hard-body radius: {radius_text(settings)}. The burns are "
            "along-track and impulsive, the covariances unchanged; "
            "[Assumptions and limits](maneuver.md#assumptions-and-limits) lists every "
            "assumption."
        ),
        "",
        (
            "Settings: target Pc "
            f"{settings['target_pc']:.0e} (keepout's default: ten times below the "
            f"red threshold, {settings['red_from']:.0e}); burns from "
            f"{grid['earliest_lead_hours']:g} h (or the CDM's creation, if later) "
            f"to {grid['latest_lead_hours']:g} h before TCA, "
            f"{grid['steps_per_orbit']} per orbit; Δv from "
            f"{grid['smallest_delta_v']:g} to {grid['largest_delta_v']:g} m/s, "
            f"{grid['delta_v_per_direction']} per direction; "
            f"{threshold_text(settings['green_below'], settings['red_from'])} "
            f"({settings['basis']} basis)."
        ),
        "",
        "## The event",
        "",
        *_event_table(plan),
        "",
        "## Pc over burn size and time",
        "",
        f"![{_alt_text(plan)}]({images}/{HEATMAP})",
        "",
        _caption(summary),
        "",
        "## Minimum burns",
        "",
        *options_table(plan),
        *_caveats(plan),
        "",
        (
            f"Cells below the target: {plan['below_target']['retrograde']} "
            f"retrograde and {plan['below_target']['prograde']} prograde of "
            f"{plan['cells']}."
        ),
        "",
        "## Re-screen demonstration",
        "",
        *_demonstration(summary),
        "",
        "## Data",
        "",
        (
            "CARA test CDM: NASA Conjunction Assessment Risk Analysis, "
            "[CARA Analysis Tools](https://github.com/nasa/CARA_Analysis_Tools), "
            f"`DataFiles/PcTestCaseCDMs/{settings['cdm']}.cdm` at commit "
            f"`{commit}`, distributed under the NASA Open Source Agreement and "
            "downloaded on demand; keepout publishes only numbers derived from it "
            "and the figure. GP data: USSPACECOM/18 SDS via Space-Track.org, "
            "retrieved from CelesTrak (celestrak.org) (see "
            "`tests/screening/data/NOTICE`)."
        ),
        "",
        "## Timing",
        "",
        _timing_line(summary["timings"], plan, check_note=False),
        "",
        f"<!-- {TOOL} --check checks the numbers, not the times. -->",
    ]
    return "\n".join(lines) + "\n"


def render_summary(summary: Summary, images: str) -> str:
    """The README block: the heatmap and the minimum burns."""
    settings, plan = summary["settings"], summary["event"]
    return "\n".join(
        [
            (
                f"The burns of {primary_text(plan)} that clear a "
                f"{_level_text(settings, plan)} event, CARA test CDM "
                f"`{settings['cdm']}` ({plan['tca'][:4]} test data; "
                f"{radius_text(settings, short=True)}; "
                "[report](docs/maneuver-report.md)):"
            ),
            "",
            f"![{_alt_text(plan)}]({images}/{HEATMAP})",
            "",
            *options_table(plan),
            "",
            _timing_line(summary["timings"], plan, check_note=True),
        ]
    )


def _mark(axes: Axes, panel: GridPanel) -> None:
    """The option of a panel: a ringed dot and its label."""
    if panel.minimum is None:
        return
    sign = panel.direction.sign
    lead, size = panel.minimum
    axes.scatter(
        [size],
        [lead],
        s=64,
        color=TARGET_COLOUR,
        edgecolors="white",
        linewidths=2.0,
        zorder=3,
    )
    axes.annotate(
        f"{delta_v_text(sign * size)} m/s\n{lead:.1f} h before TCA",
        (size, lead),
        xytext=(10 * sign, -10),
        textcoords="offset points",
        ha="left" if sign > 0 else "right",
        va="top",
        fontsize="small",
        color=INK,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none"},
    )


def _panel(
    axes: Axes, figure: GridFigure, panel: GridPanel, cmap: Colormap
) -> QuadMesh:
    """One direction's cells, the target contour and the option."""
    values = panel.log_pc
    mesh = axes.pcolormesh(
        panel.delta_v,
        panel.hours_before_tca,
        np.ma.masked_invalid(values),
        cmap=cmap,
        vmin=LOWEST_LOG_PC,
        vmax=figure.top_log_pc,
        shading="flat",
        edgecolors="none",
        rasterized=True,
    )
    if panel.crosses(figure.target_log_pc):
        axes.contour(
            panel.delta_v_centres,
            panel.hour_centres,
            np.ma.masked_invalid(values),
            levels=[figure.target_log_pc],
            colors=TARGET_COLOUR,
            linewidths=2.0,
            linestyles="solid",
        )
    _mark(axes, panel)
    axes.set_xscale("log")
    axes.set_title(panel.direction, color=INK)
    axes.set_xlabel("|Δv| along the velocity (m/s)", color=MUTED)
    axes.tick_params(colors=MUTED)
    for spine in axes.spines.values():
        spine.set_color(RULE)
    return mesh


def _legend(drawing: Figure, figure: GridFigure) -> None:
    handles: list[Artist] = [
        Line2D([], [], color=TARGET_COLOUR, linewidth=2.0),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=8,
            markerfacecolor=TARGET_COLOUR,
            markeredgecolor="white",
        ),
    ]
    labels = [f"target Pc {figure.target_pc:.0e}", "smallest burn meeting it"]
    if any(np.isnan(panel.log_pc).any() for panel in figure.directions):
        handles.append(Patch(facecolor=NO_PC))
        labels.append("no Pc (no minimum to trust)")
    drawing.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        fontsize="small",
        labelcolor=INK,
    )


def heatmap_figure(figure: GridFigure, title: str) -> Figure:
    """The heatmap of a plan: log10 Pc over |Δv| (retrograde mirrored to the
    left) and hours before TCA."""
    cmap = LinearSegmentedColormap.from_list("pc", SEQUENTIAL).with_extremes(bad=NO_PC)
    retrograde, prograde = figure.directions
    with mpl.rc_context({"svg.hashsalt": "keepout", "svg.fonttype": "none"}):
        drawing = Figure(figsize=(9.0, 4.8))
        grid = drawing.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 0.05), wspace=0.12)
        left = drawing.add_subplot(grid[0, 0])
        right = drawing.add_subplot(grid[0, 1], sharey=left)
        _panel(left, figure, retrograde, cmap)
        mesh = _panel(right, figure, prograde, cmap)
        left.invert_xaxis()
        left.set_ylabel("burn time before TCA (h)", color=MUTED)
        right.tick_params(labelleft=False)
        colourbar = drawing.colorbar(
            mesh, cax=drawing.add_subplot(grid[0, 2]), extend="min"
        )
        colourbar.set_label("log10 Pc (Foster)", color=MUTED)
        colourbar.ax.axhline(figure.target_log_pc, color=TARGET_COLOUR, linewidth=2.0)
        colourbar.ax.tick_params(colors=MUTED)
        colourbar.outline.set_edgecolor(RULE)
        _legend(drawing, figure)
        drawing.suptitle(title, color=INK, fontsize=11)
        drawing.subplots_adjust(left=0.08, right=0.93, top=0.84, bottom=0.2)
    return drawing


def plot(figure: GridFigure, title: str, path: Path) -> None:
    """The heatmap of a plan (:func:`heatmap_figure`), saved as a
    reproducible SVG."""
    drawing = heatmap_figure(figure, title)
    with mpl.rc_context({"svg.hashsalt": "keepout", "svg.fonttype": "none"}):
        drawing.savefig(
            path, format="svg", metadata={"Date": None}, facecolor="white", dpi=200
        )


def heatmap_title(plan: Plan) -> str:
    """The title of a plan's heatmap."""
    secondary = _object(plan["secondary"], plan["secondary_name"])
    return (
        f"{primary_text(plan)} vs {secondary}\nTCA {minute_text(plan['tca'])} UTC, "
        f"Pc {plan['baseline']['pc']:.2e} without a burn"
    )


def read_timings(path: Path) -> Timings | None:
    """The timings of the committed JSON at ``path``, if it exists.

    Raises
    ------
    ValueError
        If the file is not a JSON file this tool writes.
    """
    if not path.exists():
        return None
    return read_json(path, Summary)["timings"]


def load_event(path: Path) -> tuple[ConjunctionEvent, str]:
    """The event of the CDM at ``path``, and the CDM's name (its stem).

    Raises
    ------
    ValueError
        If the file holds more than one event.
    """
    cdm: Cdm = load(path)
    events = group_events([cdm])
    if len(events) != 1:
        raise ValueError(f"{path} must hold one event")
    name = path.name.removesuffix(".gz").removesuffix(".cdm").removesuffix(".kvn")
    return events[0], name


def positive_radius(text: str) -> float:
    """A command-line radius [m]: a finite number above 0.

    Raises
    ------
    argparse.ArgumentTypeError
        If ``text`` is not one.
    """
    value = float(text)
    if not (math.isfinite(value) and value > 0.0):
        raise argparse.ArgumentTypeError(f"must be finite and above 0, got {text}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=TOOL)
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "fail if out of date (the timings are not checked, and the heatmap "
            "only needs to exist: its SVG bytes vary with the matplotlib version)"
        ),
    )
    parser.add_argument(
        "--keep-timings",
        action="store_true",
        help="write with the committed JSON's timings instead of measuring again",
    )
    parser.add_argument(
        "--cdm", type=Path, default=None, help="a local CDM instead of CARA's"
    )
    parser.add_argument("--data", type=Path, default=snapshot.DATA)
    parser.add_argument("--fleet", type=Path, default=FLEET)
    parser.add_argument("--page", type=Path, default=PAGE)
    parser.add_argument("--images", type=Path, default=IMAGES)
    parser.add_argument("--readme", type=Path, default=README)
    parser.add_argument(
        "--hbr",
        type=positive_radius,
        default=None,
        help="combined hard-body radius [m] instead of CARA's",
    )
    parser.add_argument(
        "--workers",
        type=positive,
        default=DEFAULT_WORKERS,
        help="processes to screen with",
    )
    parser.add_argument(
        "--repeat", type=positive, default=3, help="runs of the timed plan"
    )
    return parser


def demonstration_summary(
    catalog: Sequence[Omm], manifest: snapshot.Manifest, fleet: Path, workers: int
) -> Demonstration:
    """The re-screen demonstration's numbers, on the frozen catalog."""
    start = window_start(manifest)
    chosen, plan, results = demonstrate(catalog, start, fleet, workers)
    pc = None if chosen is None else chosen.observation.pc
    return {
        "fleet": shown(fleet),
        "retrieved": snapshot.retrieval_day(manifest),
        "start": _time_text(start),
        "days": DEFAULT_DURATION / DAY,
        "volume": DEFAULT_VOLUME.label,
        "pc": None if pc is None else significant(pc),
        "level": None if chosen is None else level_key(chosen),
        "action": "-" if chosen is None else chosen.action.label,
        "plan": None if plan is None else plan_summary(plan),
        "rescreens": [rescreen_summary(result) for result in results],
        "recommendable": judge(
            [(result.option, result) for result in results]
        ).recommendable,
    }


def measured_timings(repeat: int, wall: list[float], cpu: list[float]) -> Timings:
    """The timings of the plan, with where they were measured."""
    return {
        "cpu_model": cpu_model(),
        "logical_cpus": os.cpu_count() or 1,
        "python": platform.python_version(),
        "date": datetime.datetime.now(datetime.UTC).date().isoformat(),
        "one_thread": all(os.environ.get(name) == "1" for name in ONE_THREAD),
        "runs": repeat,
        "wall": wall,
        "cpu": cpu,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """The entry point writing (or with ``--check`` verifying) the manoeuvre
    report.

    Returns
    -------
    int
        0 on success; 1 if, with ``--check``, a file is out of date or the
        heatmap is missing.
    """
    args = _parser().parse_args(argv)

    path = args.cdm or cara.fetch(cara.PC_METHOD_CDMS[CDM_ID])
    event, name = load_event(path)
    source: Literal["cara", "given"]
    if args.hbr is None:
        radius = cara_radius(cara.fetch(cara.PC_METHOD_TABLE), name)
        source = "cara"
    else:
        radius, source = args.hbr, "given"
    committed_timings = args.check or args.keep_timings
    plan, wall, cpu = timed_plan(event, radius, 1 if committed_timings else args.repeat)

    demonstration = demonstration_summary(
        snapshot.read_catalog(args.data),
        snapshot.read_manifest(args.data),
        args.fleet,
        args.workers,
    )
    timings = (
        read_timings(args.page.with_suffix(".json"))
        if committed_timings
        else measured_timings(args.repeat, wall, cpu)
    )
    summary: Summary = {
        "settings": {
            "cdm": name,
            "cara_commit": cara.CARA_COMMIT,
            "hard_body_radius": radius,
            "hard_body_radius_source": source,
            "target_pc": TARGET_PC,
            "grid": grid_settings(),
            **threshold_settings(THRESHOLDS),
        },
        "event": plan_summary(plan),
        "demonstration": demonstration,
        "timings": timings,
    }
    files = {
        args.page.with_suffix(".json"): json_text(summary),
        args.page: render_page(summary, relative(args.images, args.page.parent)),
        args.readme: readme_with(
            args.readme,
            START,
            END,
            render_summary(summary, relative(args.images, args.readme.parent)),
        ),
    }
    status = write_or_check(files, check=args.check, tool=TOOL)
    heatmap = args.images / HEATMAP
    if args.check:
        # Only its existence: matplotlib versions write different SVG bytes
        # for the same figure, so comparing them would fail on another setup.
        if not heatmap.exists():
            print(f"{heatmap} is missing; run {TOOL}")
            status = 1
        return status
    args.images.mkdir(parents=True, exist_ok=True)
    plot(grid_figure(plan), heatmap_title(summary["event"]), heatmap)
    print(f"wrote {heatmap}")
    return status


if __name__ == "__main__":
    sys.exit(main())
