"""The Iridium-Cosmos case study: keepout on the element sets public before
the 2009 collision.

Runs the computations of :mod:`iridium_cosmos` on the committed excerpt of
Space-Track's 2009 bulk file (``tools/iridium_cosmos_data.py``) and on the
committed Space-Track snapshot of 2026-10-15 (``tools/spacetrack_snapshot.py``),
and writes ``docs/case-study-iridium-cosmos.json`` (the numbers), the page,
its three figures and the summary block in ``docs/reports.md``. ``--check``
recomputes everything and fails if a committed file differs (the figures
only have to exist: SVG bytes vary with the matplotlib version). It takes a
few minutes, so the nightly workflow runs it. ``--help`` lists the options.

The values quoted from Kelso (2009) are in :data:`KELSO`, read from the
paper; nothing of them is recomputed.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Final, TypedDict

import astropy.units as u
import iridium_cosmos as study
import iridium_cosmos_data
import spacetrack_snapshot as snapshot
from astropy.time import Time
from generated import (
    REPO,
    REPORTS,
    generated_notice,
    minute_text,
    relative,
    report_files,
    report_parser,
    significant,
    table,
    threshold_settings,
    threshold_text,
    write_or_check,
    write_or_check_figures,
)
from maneuver_heatmap import DPI, FACECOLOR, heatmap_figure
from matplotlib.figure import Figure

from keepout._text import pc_text, plural
from keepout.figures import (
    DECISION_POINT_COLOUR,
    FIT_COLOUR,
    INK_COLOUR,
    MUTED_COLOUR,
    RISK_COLOURS,
    THRESHOLD_LABELS,
    grid_figure,
    trend_figure,
)
from keepout.risk import (
    DEFAULT_DECISION_CONFIG,
    DEFAULT_THRESHOLDS,
    Action,
    Assessment,
    RiskLevel,
)
from keepout.units import DAY, HOUR, KM

if TYPE_CHECKING:
    from collections.abc import Sequence

    from keepout.events import ConjunctionEvent
    from keepout.omm import Omm

TOOL: Final = "tools/iridium_cosmos_report.py"
PAGE: Final = REPO / "docs/case-study-iridium-cosmos.md"
IMAGES: Final = REPO / "docs/img/iridium-cosmos"
TREND: Final = "trend.svg"
RANK: Final = "rank.svg"
HEATMAP: Final = "heatmap.svg"
START: Final = "<!-- iridium-cosmos-summary:start -->"
END: Final = "<!-- iridium-cosmos-summary:end -->"

DIGITS: Final = 3
"""Significant digits of the probabilities written: the last digits of a
Pc vary between machines."""

KELSO: Final = {
    "citation": (
        'T.S. Kelso, "Analysis of the Iridium 33-Cosmos 2251 Collision", '
        "10th AMOS Conference, Maui, 2009 September 2"
    ),
    "url": "https://celestrak.org/publications/AMOS/2009/AMOS-2009.pdf",
    "last_report": "2009-02-10 15:02 UTC",
    "last_report_miss": "584 m",
    "reports": "14",
    "first_report": "2009-02-04 04:01 UTC",
    "tca_span": "16:55:59.670 to 16:55:59.990 UTC",
    "miss_span": "117 m to 1.812 km",
    "report_5": "117 m (2009-02-06 04:02 UTC), then 1.243 km the next day",
    "overall_rank": "1,611 (report 3) to 11; 152 at the collision",
    "iridium_rank": "149 (report 3) to 2 (report 5), 64 on average",
    "weekly": "10 to 15 approaches within 5 km a week",
    "constellation_weekly": "1,007 to 1,095",
}
"""What Kelso (2009, secs. 1-3) reports of SOCRATES's predictions."""

KELSO_AAS: Final = "https://celestrak.org/publications/AAS/09-368/AAS-09-368.pdf"
"""Kelso's AAS 09-368 paper, the source of the two objects' volumes."""

CELESTRAK_EVENT: Final = "https://celestrak.org/events/collision/"
"""CelesTrak's page on the collision."""


class Settings(TypedDict):
    """What the report was computed with."""

    tca_reference: str
    as_of: str
    base_hbr: float
    iridium_hbr: float
    hbr_sensitivity: list[float]
    series_days: float
    window_days: float
    excerpt_radius_km: float
    designators_blanked: int
    unreadable_sets: int
    unreadable_objects: int
    green_below: float
    red_from: float
    basis: str
    maneuver_target: float
    burn_earliest_hours: float
    burn_latest_hours: float
    bulk_sha256: str
    snapshot_retrieved: str


class Issue(TypedDict):
    """One synthetic CDM issue, or why there is none."""

    created: str
    tca: str | None
    miss_m: float | None
    pc: float | None
    frisbee: float | None
    dilution: float | None
    alfano: float | None
    known_at_decision: bool
    reason: str | None


class Verdict(TypedDict):
    """An assessment: action, level, latest Pc, CDMs and reasons."""

    action: str
    level: str | None
    latest_pc: float | None
    cdms: int
    reasons: list[str]


class Sensitivity(TypedDict):
    """The Pc at one combined radius."""

    radius: float
    pc_at_decision: float
    pc_last: float


class Approach(TypedDict):
    """One of Iridium 33's approaches in the ranking."""

    secondary: int
    designator: str
    tca: str
    miss_m: float
    pc: float | None
    collision: bool


class RankSummary(TypedDict):
    """The collision's ranks and every approach, by Pc."""

    approaches: int
    by_pc: int
    by_miss: int
    table: list[Approach]


class Option(TypedDict):
    """One burn option and its re-screen."""

    direction: str
    delta_v: float
    burn_lead_hours: float
    pc: float
    range_at_tca_m: float
    shift_km: float
    new: int
    closer: int
    worst_pc: float | None
    conflict: bool


class ManeuverSummary(TypedDict):
    """The burn plan at the decision point."""

    baseline_pc: float
    target: float
    options: list[Option]


class DebrisApproach(TypedDict):
    """A fragment's approach to another object."""

    piece: int
    piece_designator: str
    other: int
    other_name: str
    tca: str
    miss_m: float


class Debris(TypedDict):
    """The fragments in the catalog of 2026."""

    on_orbit: dict[str, int]
    approaches: int
    closest: list[DebrisApproach]


class Summary(TypedDict):
    """Every number of the page (the committed JSON)."""

    settings: Settings
    series: list[Issue]
    decision: Verdict
    final: Verdict
    sensitivity: list[Sensitivity]
    rank: RankSummary
    maneuver: ManeuverSummary
    debris: Debris


def _pc(value: float | None) -> float | None:
    return None if value is None else significant(value, DIGITS)


def _metres(value: float) -> float:
    return float(round(value))


def _iso(time: Time) -> str:
    """``time`` in ISO, to the millisecond (element set epochs carry more)."""
    return str(time.utc.isot)[:23]


def _issues(issues: Sequence[study.Issue], event: ConjunctionEvent) -> list[Issue]:
    observations = {o.message_id: o for o in event.observations()}
    rows: list[Issue] = []
    for issue in issues:
        row: Issue = {
            "created": _iso(issue.time),
            "tca": None,
            "miss_m": None,
            "pc": None,
            "frisbee": None,
            "dilution": None,
            "alfano": None,
            "known_at_decision": bool(issue.time <= study.IRIDIUM_AS_OF),
            "reason": issue.reason,
        }
        if (cdm := issue.cdm) is not None:
            bounds = study.max_pcs(cdm, study.BASE_HBR)
            row["tca"] = _iso(cdm.tca)
            row["miss_m"] = _metres(cdm.miss_distance)
            row["pc"] = _pc(observations[cdm.message_id].pc)
            row["frisbee"] = _pc(bounds["frisbee"])
            row["dilution"] = _pc(bounds["dilution"])
            row["alfano"] = _pc(bounds["alfano"])
        rows.append(row)
    return rows


def _verdict(assessment: Assessment) -> Verdict:
    latest = assessment.trend.latest_observation
    level = assessment.level
    return {
        "action": assessment.decision.action.label,
        "level": None if level is None else level.key,
        "latest_pc": None if latest is None else _pc(latest.pc),
        "cdms": len(assessment.trend.observations),
        "reasons": list(assessment.decision.reasons),
    }


def _rank(ranked: study.Ranked, excerpt: Sequence[Omm]) -> RankSummary:
    designators = {omm.catalog_number: omm.international_designator for omm in excerpt}
    return {
        "approaches": ranked.approaches,
        "by_pc": ranked.by_pc,
        "by_miss": ranked.by_miss,
        "table": [
            {
                "secondary": event.approach.secondary_number,
                "designator": designators.get(
                    event.approach.secondary_number, "UNKNOWN"
                ),
                "tca": _iso(event.approach.tca),
                "miss_m": _metres(event.approach.range_at_tca),
                "pc": _pc(event.observation.pc),
                "collision": event is ranked.collision,
            }
            for event in ranked.events_by_pc
        ],
    }


def _maneuver(result: study.Maneuver) -> ManeuverSummary:
    options: list[Option] = []
    for option in result.plan.options:
        rescreen = result.rescreens[option.direction]
        found = (*rescreen.new_approaches, *rescreen.closer_approaches)
        worst = max((item.pc for item in found), default=None)
        options.append(
            {
                "direction": str(option.direction),
                "delta_v": significant(option.delta_v, 4),
                "burn_lead_hours": round(option.burn_lead / HOUR, 2),
                "pc": significant(option.pc, DIGITS),
                "range_at_tca_m": _metres(option.range_at_tca),
                "shift_km": round(result.shifts[option.direction] / KM, 1),
                "new": len(rescreen.new_approaches),
                "closer": len(rescreen.closer_approaches),
                "worst_pc": _pc(worst),
                "conflict": rescreen.conflict,
            }
        )
    return {
        "baseline_pc": significant(result.plan.baseline.pc, DIGITS),
        "target": study.MANEUVER_TARGET,
        "options": options,
    }


def _debris(catalog: Sequence[Omm], start: Time) -> Debris:
    today = study.debris_today(catalog, start)
    by_number = {omm.catalog_number: omm for omm in catalog}
    return {
        "on_orbit": dict(today.on_orbit),
        "approaches": today.approaches,
        "closest": [
            {
                "piece": approach.primary_number,
                "piece_designator": by_number[
                    approach.primary_number
                ].international_designator,
                "other": approach.secondary_number,
                "other_name": by_number[approach.secondary_number].object_name,
                "tca": _iso(approach.tca),
                "miss_m": _metres(approach.range_at_tca),
            }
            for approach in today.closest
        ],
    }


def summarise() -> tuple[Summary, study.Maneuver, Assessment]:
    """Every number of the page, the manoeuvre (for its heatmap) and the
    assessment on all CDMs (for the trend figure)."""
    excerpt = study.read_catalog()
    issues = study.cdm_series(study.pair_of(excerpt))
    event = study.collision_event(issues)
    at_decision, final = study.assessments(event)
    sensitivity = study.hbr_sensitivity(event)
    excerpt_manifest = iridium_cosmos_data.read_manifest(study.DATA)
    bulk = excerpt_manifest["bulk"]
    maneuver = study.maneuver(event, excerpt, excerpt_manifest["selection"])
    retrieved = snapshot.read_manifest(snapshot.DATA)["download"]["retrieved"]
    start = Time(retrieved.removesuffix("Z"), scale="utc")
    thresholds = threshold_settings(DEFAULT_THRESHOLDS)
    unreadable = bulk["counts"]["unreadable"]
    window = study.TCA_REFERENCE - iridium_cosmos_data.WINDOW_START
    summary: Summary = {
        "settings": {
            "tca_reference": _iso(study.TCA_REFERENCE),
            "as_of": _iso(study.IRIDIUM_AS_OF),
            "base_hbr": study.BASE_HBR,
            "iridium_hbr": study.IRIDIUM_HBR,
            "hbr_sensitivity": list(study.HBR_SENSITIVITY),
            "series_days": study.SERIES_SPAN / DAY,
            "window_days": round(float(window.to_value(u.day)), 3),
            "excerpt_radius_km": study.EXCERPT_RADIUS / KM,
            "designators_blanked": len(bulk["counts"]["designators_blanked"]),
            "unreadable_sets": len(unreadable),
            "unreadable_objects": len({item["catalog_number"] for item in unreadable}),
            "green_below": thresholds["green_below"],
            "red_from": thresholds["red_from"],
            "basis": thresholds["basis"],
            "maneuver_target": study.MANEUVER_TARGET,
            "burn_earliest_hours": study.BURN_GRID.earliest_lead / HOUR,
            "burn_latest_hours": study.BURN_GRID.latest_lead / HOUR,
            "bulk_sha256": bulk["sha256"],
            "snapshot_retrieved": retrieved,
        },
        "series": _issues(issues, event),
        "decision": _verdict(at_decision),
        "final": _verdict(final),
        "sensitivity": [
            {
                "radius": radius,
                "pc_at_decision": significant(values[0], DIGITS),
                "pc_last": significant(values[1], DIGITS),
            }
            for radius, values in sensitivity.items()
        ],
        "rank": _rank(study.rank(excerpt), excerpt),
        "maneuver": _maneuver(maneuver),
        "debris": _debris(snapshot.read_catalog(snapshot.DATA), start),
    }
    return summary, maneuver, final


# The page.


def _km(metres: float) -> str:
    return f"{metres / KM:.3f} km" if metres >= KM else f"{metres:.0f} m"


def _hours(iso: str, reference: str) -> float:
    return float(
        (Time(reference, scale="utc") - Time(iso, scale="utc")).to_value(u.hour)
    )


def _found(summary: Summary) -> list[Issue]:
    return [row for row in summary["series"] if row["pc"] is not None]


def _last_known(summary: Summary) -> Issue:
    return [row for row in _found(summary) if row["known_at_decision"]][-1]


def _best(summary: Summary) -> Option:
    return min(
        summary["maneuver"]["options"], key=lambda option: abs(option["delta_v"])
    )


BURN_ACTIONS: Final = frozenset({Action.MANEUVER.label, Action.PREPARE_MANEUVER.label})
"""The decisions that call for a burn."""

KELSO_QUOTE: Final = (
    "Determining that the predicted conjunction for Iridium 33 and Cosmos 2251 "
    "was more significant than the many dozens of other Iridium conjunctions for "
    "that week is simply not possible using the TLE data."
)
"""Kelso (2009, sec. 2)."""


def _never_red(summary: Summary) -> bool:
    """Whether no CDM's model Pc reached the red threshold."""
    red = summary["settings"]["red_from"]
    return all((row["pc"] or 0.0) < red for row in _found(summary))


def _dilution_red_by_decision(summary: Summary) -> bool:
    """Whether the dilution maximum Pc of a CDM known at the decision point
    reached the red threshold."""
    red = summary["settings"]["red_from"]
    return any(
        (row["dilution"] or 0.0) >= red
        for row in _found(summary)
        if row["known_at_decision"]
    )


def _rank_text(summary: Summary) -> str:
    """Where the collision ranked by Pc among Iridium 33's own approaches."""
    rank = summary["rank"]
    text = (
        f"among Iridium 33's own {rank['approaches']} approaches that week it ranked "
        f"{_ordinal(rank['by_pc'])} by Pc"
    )
    collision = next(item["pc"] or 0.0 for item in rank["table"] if item["collision"])
    others = sorted(
        (item["pc"] or 0.0 for item in rank["table"] if not item["collision"]),
        reverse=True,
    )
    if rank["by_pc"] == 1 and others and others[0] > 0:
        text += f", {collision / others[0]:.0f} times the next"
    return text


def meaning(summary: Summary) -> str:
    """What the result means, worded from the summary's own numbers."""
    decision = summary["decision"]
    action = decision["action"]
    red = summary["settings"]["red_from"]
    lead = (
        f"At the decision point keepout's default rules on the model Pc say "
        f"**{action}**: the event was {decision['level']}, its Pc "
        f"{pc_text(decision['latest_pc'])} on the assumed covariance"
        + (", and no CDM was red." if _never_red(summary) else ".")
    )
    if action in BURN_ACTIONS:
        return f"{lead} The burn they call for is planned below."
    watch = f"Still, it was the one to watch: {_rank_text(summary)}"
    if _dilution_red_by_decision(summary):
        watch += f", and its dilution maximum Pc had reached {red:.0e}"
    return (
        f"{lead} {watch}. The rules do not call for a burn on a {decision['level']} "
        "event, and whether it stood out among the whole constellation's "
        "conjunctions keepout did not compute. Kelso found that, on the TLE data, "
        f'it could not be told apart: "{KELSO_QUOTE}" keepout\'s Pc is only as good '
        "as the covariance it is given, and here the covariance is a model of GP "
        "accuracy, not of these two orbits."
    )


def _summary_section(summary: Summary) -> list[str]:
    settings, rank, debris = summary["settings"], summary["rank"], summary["debris"]
    found = _found(summary)
    misses = [row["miss_m"] or 0.0 for row in found]
    best = _best(summary)
    fragments = sum(debris["on_orbit"].values())
    never_red = ", never red." if _never_red(summary) else "."
    return [
        "## Summary",
        "",
        (
            "On 2009 February 10, Iridium 33 (24946), an operational communications "
            "satellite, and Cosmos 2251 (22675), a long-defunct one, collided at "
            "nearly right angles over northern Russia: the first known collision of "
            "two satellites (Kelso 2009). This page runs keepout on the element sets "
            "that were public then, as an operator could have, and asks what it "
            "would have said."
        ),
        "",
        (
            f"- **Screening.** Each new element set of the pair in the "
            f"{settings['series_days']:g} days before gives a synthetic CDM. Their "
            f"miss distances run from {_km(min(misses))} to {_km(max(misses))}; "
            f"the last, {_km(misses[-1])}. They match what CelesTrak's SOCRATES "
            f"printed from the same kind of data ({KELSO['miss_span']}; "
            f"{KELSO['last_report_miss']} in its last report)."
        ),
        (
            f"- **Pc.** On keepout's assumed covariance the Pc rose from "
            f"{pc_text(found[0]['pc'])} to {pc_text(found[-1]['pc'])}: "
            f"{summary['decision']['level']} at the decision point{never_red}"
        ),
        f"- **Decision.** Two days before TCA: **{summary['decision']['action']}**.",
        (
            f"- **Rank.** Among Iridium 33's {rank['approaches']} approaches in the "
            f"week from the decision point, the collision was "
            f"{_ordinal(rank['by_pc'])} by Pc and {_ordinal(rank['by_miss'])} by "
            "miss distance."
        ),
        (
            f"- **Manoeuvre.** A {best['direction']} burn of "
            f"{abs(best['delta_v']) * 100:.1f} cm/s {best['burn_lead_hours']:.0f} h "
            f"before TCA would have brought the Pc below "
            f"{summary['maneuver']['target']:.0e}."
        ),
        (
            f"- **Today.** {fragments:,} catalogued fragments of the two were still "
            f"in orbit on {settings['snapshot_retrieved'][:10]}; in one day they "
            f"came within 5 km of other objects {debris['approaches']:,} times."
        ),
        "",
        meaning(summary),
    ]


def _ordinal(rank: int) -> str:
    suffixes = ("th", "st", "nd", "rd")  # codespell:ignore nd
    last = rank % 10 if rank % 100 not in (11, 12, 13) else 0
    suffix = suffixes[last] if last < len(suffixes) else "th"
    return f"{rank}{suffix}"


def _data_section(summary: Summary) -> list[str]:
    settings = summary["settings"]
    blanked = plural(settings["designators_blanked"], "designator")
    unreadable = plural(settings["unreadable_sets"], "set")
    objects = plural(settings["unreadable_objects"], "object")
    return [
        "## Data",
        "",
        (
            "Space-Track.org allows the history of an object's element sets to be "
            "queried once per account, and sends queries of many objects to its "
            "yearly bulk files. The excerpt in `tests/case_study/data/iridium-cosmos/` "
            f"comes from the 2009 file (SHA-256 `{settings['bulk_sha256'][:16]}...`, "
            "see its `manifest.json`): every element set of Iridium 33 and Cosmos "
            f"2251 from {settings['window_days']:g} days before the collision, and "
            "the newest set known at the decision point of each object that passed "
            f"within {settings['excerpt_radius_km']:g} km of Iridium 33 from then to "
            "a week after the latest burn the planner may try. "
            "`tools/iridium_cosmos_data.py` makes it; the manifest counts what it "
            "left out: re-issued and corrected sets of the same epoch (the later "
            f"kept), {blanked} without piece letters (blanked) and {unreadable} "
            f"keepout cannot read (a malformed B\\*, {objects}). Its TLEs carry no "
            "names, so objects other than the pair are named by catalog number and "
            "international designator."
        ),
        "",
        (
            f"Times: the collision, {settings['tca_reference']} UTC, is the TCA of "
            "SOCRATES's last report, which Kelso (2009) takes as the time of the "
            f"collision. The decision point is two days before, {settings['as_of']} "
            "UTC; an analysis at a time sees only the element sets with an epoch up "
            "to it."
        ),
    ]


def _screening_section(summary: Summary) -> list[str]:
    rows = [
        (
            minute_text(row["created"]),
            "yes" if row["known_at_decision"] else "no",
            "-" if row["tca"] is None else row["tca"][11:23],
            "-" if row["miss_m"] is None else _km(row["miss_m"]),
            pc_text(row["pc"]),
            pc_text(row["frisbee"]),
            pc_text(row["dilution"]),
            pc_text(row["alfano"]),
        )
        for row in summary["series"]
    ]
    missing = [row for row in summary["series"] if row["reason"] is not None]
    missing_text = (
        [
            f"No CDM at {len(missing)} issue time(s): "
            + "; ".join(
                f"{minute_text(row['created'])}: {row['reason']}" for row in missing
            )
            + ".",
            "",
        ]
        if missing
        else []
    )
    return [
        "## What screening saw",
        "",
        (
            "Each CDM is created at the epoch of the newer of the two element sets "
            "it uses (no publication latency) and screens the pair from then to an "
            "hour past the collision with SGP4. Pc is keepout's on the synthetic "
            "covariance; the maximum Pc columns bound it from above, each trusting "
            "less of that covariance: Frisbee's with Cosmos 2251's covariance "
            "unknown (Iridium 33's kept), dilution's over a scale of the combined "
            "one (its shape kept), Alfano's over any covariance."
        ),
        "",
        *table(
            (
                "Created (UTC)",
                "Known at decision point",
                "TCA (UTC)",
                "Miss",
                "Pc",
                "Max Pc (Frisbee)",
                "Max Pc (dilution)",
                "Max Pc (Alfano)",
            ),
            rows,
        ),
        "",
        *missing_text,
        (
            "SOCRATES, from the same kind of data, predicted this approach in each "
            f"of its {KELSO['reports']} reports from {KELSO['first_report']}, at "
            f"{KELSO['tca_span']}, with miss distances from {KELSO['miss_span']}; "
            f"its report 5 gave {KELSO['report_5']}; its last report, of "
            f"{KELSO['last_report']}, gave {KELSO['last_report_miss']} (Kelso "
            "2009). The CDMs above give the same spread: screening on these element "
            "sets finds what SOCRATES found."
        ),
    ]


def _trend_section(summary: Summary, images: str) -> list[str]:
    settings = summary["settings"]
    thresholds = threshold_text(settings["green_below"], settings["red_from"])
    misses = [row["miss_m"] or 0.0 for row in _found(summary)]
    return [
        "## Pc and trend",
        "",
        f"![Pc of the synthetic CDMs before the collision]({images}/{TREND})",
        "",
        (
            "Each point is one CDM's Pc on the synthetic covariance, coloured by "
            f"its risk level ({thresholds}); the grey line is Frisbee's maximum Pc. "
            "The Pc rises as TCA nears because the model's covariance shrinks as "
            "the element sets are propagated less far, not because the miss "
            f"distance falls: the misses jump between {_km(min(misses))} and "
            f"{_km(max(misses))}."
        ),
    ]


def _decision_section(summary: Summary) -> list[str]:
    settings = summary["settings"]
    decision, final = summary["decision"], summary["final"]
    rows = [
        (
            f"{row['radius']:g} m"
            + (" (base)" if row["radius"] == settings["base_hbr"] else ""),
            pc_text(row["pc_at_decision"]),
            pc_text(row["pc_last"]),
        )
        for row in summary["sensitivity"]
    ]
    return [
        "## Decision two days before",
        "",
        f"On the {decision['cdms']} CDMs created by {minute_text(settings['as_of'])} "
        f"UTC, `keepout.risk.assess` with the default rules gives "
        f"**{decision['action']}**: "
        + "; ".join(decision["reasons"])
        + f". On all {final['cdms']} CDMs: **{final['action']}** ("
        + "; ".join(final["reasons"])
        + ").",
        "",
        (
            f"The combined hard-body radius is {settings['base_hbr']:g} m, an "
            "assumption (no dimensions are published with the data). The Pc grows "
            "about as its square:"
        ),
        "",
        *table(
            ("Combined radius", "Pc at the decision point", "Pc of the last CDM"), rows
        ),
    ]


def _rank_section(summary: Summary, images: str) -> list[str]:
    rank = summary["rank"]

    def secondary(item: Approach) -> str:
        if item["secondary"] == study.COSMOS_2251:
            name = f"Cosmos 2251 ({study.COSMOS_2251})"
        else:
            name = f"{item['secondary']} ({item['designator']})"
        return f"**{name}**" if item["collision"] else name

    rows = [
        (
            secondary(item),
            minute_text(item["tca"]),
            _km(item["miss_m"]),
            pc_text(item["pc"]),
        )
        for item in rank["table"]
    ]
    return [
        "## Rank among the week's approaches",
        "",
        (
            "Iridium 33 screened against every object as known at the decision "
            "point, for 7 days, in the default 5 km sphere, each approach with the "
            "model's covariance and the same combined hard-body radius for every "
            "secondary (so the order reflects geometry and covariance only): "
            f"{rank['approaches']} approaches. The collision ranks "
            f"{_ordinal(rank['by_pc'])} by Pc and {_ordinal(rank['by_miss'])} by "
            "miss distance."
        ),
        "",
        (
            "![Iridium 33's approaches in the week from the decision point]"
            f"({images}/{RANK})"
        ),
        "",
        *table(("Secondary", "TCA (UTC)", "Miss", "Pc"), rows),
        "",
        (
            "SOCRATES ranked conjunctions across the whole catalog and the whole "
            f"Iridium constellation: the collision's overall rank went from "
            f"{KELSO['overall_rank']}, and among all Iridium conjunctions from "
            f"{KELSO['iridium_rank']}; Iridium 33 alone saw {KELSO['weekly']} out "
            f"of {KELSO['constellation_weekly']} for the whole constellation (Kelso "
            "2009). For one satellite, as here, the list is short; an operator of "
            "66 had to sort about a thousand a week, on Pc no better than this."
        ),
    ]


def _maneuver_section(summary: Summary, images: str) -> list[str]:
    settings, plan = summary["settings"], summary["maneuver"]
    rows = [
        (
            option["direction"],
            f"{option['burn_lead_hours']:.1f} h",
            f"{option['delta_v']:+.4f} m/s",
            pc_text(option["pc"]),
            _km(option["range_at_tca_m"]),
            f"{option['shift_km']:g} km",
            f"{option['new']} new, {option['closer']} closer",
            pc_text(option["worst_pc"]),
            "yes" if option["conflict"] else "no",
        )
        for option in plan["options"]
    ]
    return [
        "## A manoeuvre",
        "",
        (
            "Had the operator chosen to act at the decision point, `plan_maneuver` "
            "on the CDMs known then gives the smallest along-track burns, from "
            f"{settings['burn_earliest_hours']:g} h to "
            f"{settings['burn_latest_hours']:g} h before TCA, that bring the Pc "
            f"({pc_text(plan['baseline_pc'])} without a burn) below "
            f"{plan['target']:.0e}, green; the planner's own default, red, the "
            "event already meets. Each burn is then re-screened against the excerpt "
            "for a week (`rescreen`): the approaches it brings, and the worst Pc "
            "among them."
        ),
        "",
        f"![Pc after an along-track burn of Iridium 33]({images}/{HEATMAP})",
        "",
        *table(
            (
                "Burn",
                "Before TCA",
                "Δv",
                "Pc",
                "Range at TCA",
                "Largest shift in a week",
                "Re-screen",
                "Worst Pc",
                "Conflict",
            ),
            rows,
        ),
        "",
        (
            "A burn moves Iridium 33 along its orbit for the rest of the week; the "
            "excerpt holds every object that came within "
            f"{settings['excerpt_radius_km']:g} km of it from the decision point to "
            "a week after the latest burn, and the study stops if a burn moves it "
            "more than half that radius or a re-screen runs past that time, so no "
            "object the excerpt left out can be the one a burn brings near. A "
            "demonstration of the mechanism, not advice about what Iridium should "
            "have done."
        ),
    ]


def _debris_section(summary: Summary) -> list[str]:
    debris, settings = summary["debris"], summary["settings"]
    rows = [
        (
            f"{item['piece']} ({item['piece_designator']})",
            f"{item['other']} ({item['other_name']})",
            minute_text(item["tca"]),
            _km(item["miss_m"]),
        )
        for item in debris["closest"]
    ]
    counts = ", ".join(
        f"{count} of {name}" for name, count in debris["on_orbit"].items()
    )
    return [
        "## Seventeen years later",
        "",
        (
            "The committed Space-Track catalog of "
            f"{settings['snapshot_retrieved'][:10]} still holds "
            f"{sum(debris['on_orbit'].values()):,} catalogued fragments of the "
            f"collision ({counts}: objects of the two launches' designators "
            "catalogued after it). Screened against that catalog for one day from "
            "its retrieval, in the default 5 km sphere, they came within 5 km of "
            f"other objects {debris['approaches']:,} times. The closest:"
        ),
        "",
        *table(("Fragment", "Other object", "TCA (UTC)", "Range"), rows),
    ]


def _assumptions_section(summary: Summary) -> list[str]:
    settings = summary["settings"]
    return [
        "## Assumptions",
        "",
        (
            "- Covariance: keepout's GP element-set age model, assumed (GP age "
            "model), not a calibrated product ([Assumptions](assumptions.md))."
        ),
        (
            "- No publication latency: a CDM is created at the epoch of its newer "
            "element set."
        ),
        (
            "- Corrections: of two sets of one object and epoch, the one issued "
            "later is used, even if it was issued after a CDM's creation time; the "
            "bulk file has no issue times."
        ),
        (
            f"- Hard-body radius: {settings['base_hbr']:g} m combined "
            f"({settings['iridium_hbr']:g} m Iridium 33 with its arrays, "
            f"{settings['base_hbr'] - settings['iridium_hbr']:g} m Cosmos 2251), an "
            "enclosing-sphere estimate. Kelso's AAS 09-368 paper of the same title "
            "gives only their volumes (Table 1: 3.388 and 7.841 m³, a "
            "volume-equivalent combined radius of 2.2 m)."
        ),
        "- Rank: the same combined hard-body radius for every secondary.",
        (
            "- Risk rules: keepout's defaults (thresholds on the "
            f"{settings['basis']} basis, decision point two days before TCA)."
        ),
    ]


def _sources_section() -> list[str]:
    return [
        "## Sources",
        "",
        (
            f"- {KELSO['citation']}, <{KELSO['url']}>; also CelesTrak's page on the "
            f"collision, <{CELESTRAK_EVENT}>; and the AAS 09-368 paper of the same "
            "title, 19th AAS/AIAA Astrodynamics Specialist Conference, 2009, "
            f"<{KELSO_AAS}>."
        ),
        (
            "- Element sets: USSPACECOM/18 SDS via Space-Track.org, 2009 bulk TLE file "
            "and the GP catalog of 2026-10-15; basic SSA data, redistributed with this "
            "citation (see each data directory's NOTICE)."
        ),
    ]


def render_page(summary: Summary, images: str) -> str:
    """The report page, its figures under ``images`` (relative to it)."""
    lines = [
        "# Case study: Iridium 33 and Cosmos 2251",
        "",
        generated_notice(TOOL),
        "",
        *_summary_section(summary),
        "",
        *_data_section(summary),
        "",
        *_screening_section(summary),
        "",
        *_trend_section(summary, images),
        "",
        *_decision_section(summary),
        "",
        *_rank_section(summary, images),
        "",
        *_maneuver_section(summary, images),
        "",
        *_debris_section(summary),
        "",
        *_assumptions_section(summary),
        "",
        *_sources_section(),
    ]
    return "\n".join(lines) + "\n"


def render_summary(summary: Summary, page: str, images: str) -> str:
    """The summary block in ``docs/reports.md``."""
    rank, decision = summary["rank"], summary["decision"]
    best = _best(summary)
    return "\n".join(
        [
            (
                "keepout on the element sets public before the 2009 collision "
                f"([case study]({page})): two days before TCA the event was "
                f"{decision['level']} (Pc {pc_text(decision['latest_pc'])} on the "
                f"assumed covariance) and the rules said **{decision['action']}**; "
                f"it ranked {_ordinal(rank['by_pc'])} by Pc among Iridium 33's "
                f"{rank['approaches']} approaches that week; a "
                f"{abs(best['delta_v']) * 100:.1f} cm/s burn would have made it "
                "green."
            ),
            "",
            f"![Pc of the synthetic CDMs before the collision]({images}/{TREND})",
        ]
    )


# The figures.


def trend_plot(assessment: Assessment, summary: Summary) -> Figure:
    """The CDMs' Pc against days to TCA, with Frisbee's maximum Pc."""
    figure = trend_figure(assessment)
    days = figure.hours_to_tca * HOUR / DAY
    drawing = Figure(figsize=(8.5, 3.8))
    axes = drawing.subplots()
    axes.plot(days, figure.pc, color=FIT_COLOUR, linewidth=1.0, zorder=1)
    axes.scatter(
        days,
        figure.pc,
        s=36,
        zorder=2,
        c=[RISK_COLOURS[level] for level in figure.levels],
        edgecolors=INK_COLOUR,
        linewidths=0.5,
    )
    found = _found(summary)
    frisbee = [row["frisbee"] or 0.0 for row in found]
    tca = summary["settings"]["tca_reference"]
    bound_days = [_hours(row["created"], tca) / 24 for row in found]
    axes.plot(
        bound_days,
        frisbee,
        color=MUTED_COLOUR,
        linewidth=1.0,
        linestyle="-",
        marker=".",
        label="max Pc (Frisbee)",
    )
    for value, level in reversed(figure.thresholds):
        axes.axhline(
            value,
            color=RISK_COLOURS[level],
            linestyle="--",
            label=f"{THRESHOLD_LABELS[level]} {value:.0e}",
        )
    axes.axvline(
        DEFAULT_DECISION_CONFIG.decision_point / DAY,
        color=DECISION_POINT_COLOUR,
        linestyle=":",
        label="decision point",
    )
    axes.set_yscale("log")
    axes.set_ylim(1e-8, 1e-3)
    axes.set_xlim(summary["settings"]["series_days"] * 1.03, 0.0)
    axes.set_xlabel("days to the collision")
    axes.set_ylabel("Pc (assumed covariance)")
    axes.set_title("Iridium 33 vs Cosmos 2251: synthetic CDMs from public TLEs")
    axes.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize="small")
    axes.grid(visible=True, which="major", alpha=0.3)
    drawing.tight_layout()
    return drawing


def rank_plot(summary: Summary) -> Figure:
    """Iridium 33's approaches: miss distance against Pc, the collision marked."""
    items = summary["rank"]["table"]
    drawing = Figure(figsize=(6.5, 4.0))
    axes = drawing.subplots()
    floor = 1e-30

    def points(*, collision: bool) -> tuple[list[float], list[float]]:
        chosen = [item for item in items if item["collision"] is collision]
        return (
            [item["miss_m"] / KM for item in chosen],
            [max(item["pc"] or floor, floor) for item in chosen],
        )

    axes.scatter(*points(collision=False), s=30, color=MUTED_COLOUR, zorder=2)
    axes.scatter(
        *points(collision=True),
        s=70,
        color=RISK_COLOURS[RiskLevel.YELLOW],
        edgecolors=INK_COLOUR,
        zorder=3,
        label="Cosmos 2251",
    )
    axes.set_yscale("log")
    axes.set_xlabel("miss distance [km]")
    axes.set_ylabel("Pc (assumed covariance)")
    axes.set_title(
        f"Iridium 33's {len(items)} approaches, week from the decision point"
    )
    axes.legend(loc="lower left", fontsize="small")
    axes.grid(visible=True, which="major", alpha=0.3)
    drawing.tight_layout()
    return drawing


def main(argv: Sequence[str] | None = None) -> int:
    """The entry point writing (or with ``--check`` verifying) the report.

    Returns
    -------
    int
        0 on success; 1 if, with ``--check``, a file is out of date or a
        figure is missing.
    """
    parser = report_parser(
        TOOL, page=PAGE, images=IMAGES, readme=REPORTS, workers=False
    )
    args = parser.parse_args(argv)
    summary, maneuver, final = summarise()
    files = report_files(
        args.page,
        summary,
        render_page(summary, relative(args.images, args.page.parent)),
        readme=args.readme,
        markers=(START, END),
        block=render_summary(
            summary,
            relative(args.page, args.readme.parent),
            relative(args.images, args.readme.parent),
        ),
    )
    status = write_or_check(files, check=args.check, tool=TOOL)
    tca = minute_text(summary["settings"]["tca_reference"])
    figures = {
        args.images / TREND: lambda: trend_plot(final, summary),
        args.images / RANK: lambda: rank_plot(summary),
        args.images / HEATMAP: lambda: heatmap_figure(
            grid_figure(maneuver.plan),
            f"Iridium 33 ({study.IRIDIUM_33}) vs Cosmos 2251 ({study.COSMOS_2251})"
            f"\nTCA {tca} UTC, Pc "
            f"{summary['maneuver']['baseline_pc']:.2e} without a burn, as known "
            "two days before",
        ),
    }
    return max(
        status,
        write_or_check_figures(
            figures, check=args.check, tool=TOOL, facecolor=FACECOLOR, dpi=DPI
        ),
    )


if __name__ == "__main__":
    sys.exit(main())
