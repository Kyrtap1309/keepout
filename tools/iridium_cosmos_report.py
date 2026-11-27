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
from keepout.risk import DEFAULT_THRESHOLDS, Assessment, RiskLevel
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
    excerpt_radius_km: float
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
    return str(time.utc.isot)[:23]


def _issues(issues: Sequence[study.Issue], event: ConjunctionEvent) -> list[Issue]:
    observations = {o.message_id: o for o in event.observations()}
    rows: list[Issue] = []
    for issue in issues:
        known = bool(issue.time <= study.IRIDIUM_AS_OF)
        if issue.cdm is None:
            rows.append(
                {
                    "created": _iso(issue.time),
                    "tca": None,
                    "miss_m": None,
                    "pc": None,
                    "frisbee": None,
                    "dilution": None,
                    "alfano": None,
                    "known_at_decision": known,
                    "reason": issue.reason,
                }
            )
            continue
        bounds = study.max_pcs(issue.cdm, study.BASE_HBR)
        rows.append(
            {
                "created": _iso(issue.time),
                "tca": _iso(issue.cdm.tca),
                "miss_m": _metres(issue.cdm.miss_distance),
                "pc": _pc(observations[issue.cdm.message_id].pc),
                "frisbee": _pc(bounds["frisbee"]),
                "dilution": _pc(bounds["dilution"]),
                "alfano": _pc(bounds["alfano"]),
                "known_at_decision": known,
                "reason": None,
            }
        )
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
    events = sorted(
        ranked.result.events,
        key=lambda event: (
            -1.0 if event.observation.pc is None else -event.observation.pc
        ),
    )
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
            for event in events
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
        "target": result.target,
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
    pair = study.read_pair()
    excerpt = study.read_catalog()
    issues = study.cdm_series(pair)
    event = study.collision_event(issues)
    at_decision, final = study.assessments(event)
    sensitivity = study.hbr_sensitivity(event)
    maneuver = study.maneuver(event, excerpt)
    manifest = snapshot.read_manifest(snapshot.DATA)
    retrieved = manifest["download"]["retrieved"]
    start = Time(retrieved.removesuffix("Z"), scale="utc")
    bulk = study_manifest_sha()
    thresholds = threshold_settings(DEFAULT_THRESHOLDS)
    summary: Summary = {
        "settings": {
            "tca_reference": _iso(study.TCA_REFERENCE),
            "as_of": _iso(study.IRIDIUM_AS_OF),
            "base_hbr": study.BASE_HBR,
            "iridium_hbr": study.IRIDIUM_HBR,
            "hbr_sensitivity": list(study.HBR_SENSITIVITY),
            "series_days": study.SERIES_SPAN / DAY,
            "excerpt_radius_km": study.EXCERPT_RADIUS / KM,
            "green_below": thresholds["green_below"],
            "red_from": thresholds["red_from"],
            "basis": thresholds["basis"],
            "maneuver_target": study.MANEUVER_TARGET,
            "burn_earliest_hours": study.BURN_GRID.earliest_lead / HOUR,
            "burn_latest_hours": study.BURN_GRID.latest_lead / HOUR,
            "bulk_sha256": bulk,
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


def study_manifest_sha() -> str:
    """The SHA-256 of the bulk file the excerpt was made from."""
    import iridium_cosmos_data

    sha256: str = iridium_cosmos_data.read_manifest(study.DATA)["bulk"]["sha256"]
    return sha256


# The page.


def _pc_text(value: float | None) -> str:
    return "-" if value is None else f"{value:.2e}"


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


def _meaning(summary: Summary) -> str:
    """What the result means, worded after the decision at the decision point."""
    decision = summary["decision"]
    action = decision["action"]
    lead = (
        f"At the decision point keepout's rules would have said **{action}**: "
        f"the event was {decision['level']}, its Pc {_pc_text(decision['latest_pc'])} "
        "on "
        "the assumed covariance."
    )
    if action.lower().startswith("manoeuvre"):
        return lead
    return (
        f"{lead} Nothing in the element sets set this approach apart enough to "
        "call for a burn, which is Kelso's conclusion too: \"Determining that the "
        "predicted conjunction for Iridium 33 and Cosmos 2251 was more significant "
        "than the many dozens of other Iridium conjunctions for that week is simply "
        'not possible using the TLE data." keepout does not change that: its Pc is '
        "only as good as the covariance it is given, and here the covariance is a "
        "model of GP accuracy, not of these two orbits."
    )


def _summary_section(summary: Summary, images: str) -> list[str]:
    settings, rank, debris = summary["settings"], summary["rank"], summary["debris"]
    last = _found(summary)[-1]
    best = _best(summary)
    fragments = sum(debris["on_orbit"].values())
    return [
        "## Summary",
        "",
        (
            "On 2009 February 10, Iridium 33 (24946), an operational communications "
            "satellite, and Cosmos 2251 (22675), a long-defunct one, collided at "
            "nearly "
            "right angles over northern Russia: the first known collision of two "
            "satellites (Kelso (2009)). This page runs keepout on the element sets "
            "that "
            "were public then, as an operator could have, and asks what it would have "
            "said."
        ),
        "",
        (
            f"- **Screening.** Each new element set of the pair in the "
            f"{settings['series_days']:g} days before gives a synthetic CDM. Their "
            "miss "
            f"distances run from {_km(min(r['miss_m'] or 0 for r in _found(summary)))} "
            "to "
            f"{_km(max(r['miss_m'] or 0 for r in _found(summary)))}; the last, "
            f"{_km(last['miss_m'] or 0)}. They match what CelesTrak's SOCRATES printed "
            f"from the same kind of data ({KELSO['miss_span']}; "
            f"{KELSO['last_report_miss']} "
            "in its last report)."
        ),
        (
            f"- **Pc.** On keepout's assumed covariance the Pc rose from "
            f"{_pc_text(_found(summary)[0]['pc'])} to {_pc_text(last['pc'])}: "
            f"{summary['decision']['level']} at the decision point, never red."
        ),
        f"- **Decision.** Two days before TCA: **{summary['decision']['action']}**.",
        (
            f"- **Rank.** Among Iridium 33's {rank['approaches']} approaches in the "
            "week "
            f"from the decision point, the collision was {_ordinal(rank['by_pc'])} by "
            "Pc "
            f"and {_ordinal(rank['by_miss'])} by miss distance."
        ),
        (
            f"- **Manoeuvre.** A {best['direction']} burn of "
            f"{abs(best['delta_v']) * 100:.1f} "
            f"cm/s {best['burn_lead_hours']:.0f} h before TCA would have brought the "
            "Pc "
            f"below {summary['maneuver']['target']:.0e}."
        ),
        (
            f"- **Today.** {fragments} catalogued fragments of the two were still in "
            f"orbit on {settings['snapshot_retrieved'][:10]}; in one day they came "
            "within "
            f"5 km of other objects {debris['approaches']} times."
        ),
        "",
        _meaning(summary),
        "",
        f"![Pc of the synthetic CDMs before the collision]({images}/{TREND})",
    ]


def _ordinal(rank: int) -> str:
    suffixes = ("th", "st", "nd", "rd")  # codespell:ignore nd
    last = rank % 10 if rank % 100 not in (11, 12, 13) else 0
    suffix = suffixes[last] if last < len(suffixes) else "th"
    return f"{rank}{suffix}"


def _data_section(summary: Summary) -> list[str]:
    settings = summary["settings"]
    return [
        "## Data",
        "",
        (
            "Space-Track.org allows the history of an object's element sets to be "
            "queried once per account, and sends queries of many objects to its yearly "
            "bulk files. The excerpt in `tests/case_study/data/iridium-cosmos/` comes "
            "from the 2009 file (SHA-256 "
            f"`{settings['bulk_sha256'][:16]}...`, see its `manifest.json`): every "
            "element set of Iridium 33 and Cosmos 2251 from 30 days before the "
            "collision, "
            "and the newest set known at the decision point of each object that passed "
            f"within {settings['excerpt_radius_km']:g} km of Iridium 33 in the "
            "following "
            "week. `tools/iridium_cosmos_data.py` makes it; the manifest counts what "
            "it "
            "left out: re-issued and corrected sets of the same epoch (the later "
            "kept), "
            "two designators without piece letters (blanked) and four sets keepout "
            "cannot "
            "read (a malformed B\\*, one object). Its TLEs carry no names, so objects "
            "other than the pair are named by catalog number and international "
            "designator."
        ),
        "",
        (
            f"Times: the collision, {settings['tca_reference']} UTC, is the TCA of "
            "SOCRATES's last report, which Kelso (2009) takes as the time of the "
            f"collision. The decision point is two days before, {settings['as_of']} "
            "UTC; "
            "an analysis at a time sees only the element sets with an epoch up to it."
        ),
    ]


def _screening_section(summary: Summary) -> list[str]:
    rows = [
        (
            minute_text(row["created"]),
            "yes" if row["known_at_decision"] else "no",
            "-" if row["tca"] is None else row["tca"][11:23],
            "-" if row["miss_m"] is None else _km(row["miss_m"]),
            _pc_text(row["pc"]),
            _pc_text(row["frisbee"]),
            _pc_text(row["dilution"]),
            _pc_text(row["alfano"]),
        )
        for row in summary["series"]
    ]
    missing = [row for row in summary["series"] if row["reason"] is not None]
    return [
        "## What screening saw",
        "",
        (
            "Each CDM is created at the epoch of the newer of the two element sets it "
            "uses (no publication latency) and screens the pair from then to an hour "
            "past "
            "the collision with SGP4. Pc is keepout's on the synthetic covariance; the "
            "maximum Pc columns bound it without trusting that covariance: Frisbee's "
            "with Cosmos 2251's covariance unknown, dilution's over a scale of the "
            "combined one, Alfano's over any covariance."
        ),
        "",
        *table(
            (
                "Created (UTC)",
                "Known at decision point",
                "TCA",
                "Miss",
                "Pc",
                "Max Pc (Frisbee)",
                "Max Pc (dilution)",
                "Max Pc (Alfano)",
            ),
            rows,
        ),
        "",
        *(
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
        ),
        (
            "SOCRATES, from the same kind of data, predicted this approach in each of "
            "its "
            f"{KELSO['reports']} reports from {KELSO['first_report']}, at "
            f"{KELSO['tca_span']}, with miss distances from {KELSO['miss_span']}; its "
            f"report 5 gave {KELSO['report_5']}; its last report, of "
            f"{KELSO['last_report']}, gave {KELSO['last_report_miss']} (Kelso (2009)). "
            "The CDMs above give the same spread: screening on these element sets "
            "finds what SOCRATES found."
        ),
    ]


def _trend_section(summary: Summary, images: str) -> list[str]:
    settings = summary["settings"]
    thresholds = threshold_text(settings["green_below"], settings["red_from"])
    return [
        "## Pc and trend",
        "",
        f"![Pc of the synthetic CDMs before the collision]({images}/{TREND})",
        "",
        (
            "Each point is one CDM's Pc on the synthetic covariance, coloured by its "
            f"risk level ({thresholds}); "
            "the grey line is Frisbee's maximum Pc. The Pc rises as TCA nears because "
            "the model's covariance shrinks as the element sets are propagated less "
            "far, not because the miss distance falls: the misses jump between 117 m "
            "and 1.8 km."
        ),
    ]


def _decision_section(summary: Summary) -> list[str]:
    settings = summary["settings"]
    decision, final = summary["decision"], summary["final"]
    rows = [
        (
            f"{row['radius']:g} m"
            + (" (base)" if row["radius"] == settings["base_hbr"] else ""),
            _pc_text(row["pc_at_decision"]),
            _pc_text(row["pc_last"]),
        )
        for row in summary["sensitivity"]
    ]
    return [
        "## Decision two days before",
        "",
        f"On the {decision['cdms']} CDMs created by {minute_text(settings['as_of'])} "
        "UTC, "
        f"`keepout.risk.assess` with the default rules gives **{decision['action']}**: "
        + "; ".join(decision["reasons"])
        + f". On all {final['cdms']} CDMs: **{final['action']}** ("
        + "; ".join(final["reasons"])
        + ").",
        "",
        (
            f"The combined hard-body radius is {settings['base_hbr']:g} m, an "
            "assumption "
            "(no dimensions are published with the data). The Pc grows about as its "
            "square:"
        ),
        "",
        *table(
            ("Combined radius", "Pc at the decision point", "Pc of the last CDM"), rows
        ),
    ]


def _rank_section(summary: Summary, images: str) -> list[str]:
    rank = summary["rank"]
    rows = [
        (
            ("**" if item["collision"] else "")
            + (
                "Cosmos 2251 (22675)"
                if item["secondary"] == study.COSMOS_2251
                else f"{item['secondary']} ({item['designator']})"
            )
            + ("**" if item["collision"] else ""),
            minute_text(item["tca"]),
            _km(item["miss_m"]),
            _pc_text(item["pc"]),
        )
        for item in rank["table"]
    ]
    return [
        "## Rank among the week's approaches",
        "",
        (
            "Iridium 33 screened against every object as known at the decision point, "
            "for 7 days, in the default 5 km sphere, each approach with the model's "
            "covariance and the same combined hard-body radius for every secondary (so "
            f"the order reflects geometry and covariance only): {rank['approaches']} "
            f"approaches. The collision ranks {_ordinal(rank['by_pc'])} by Pc and "
            f"{_ordinal(rank['by_miss'])} by miss distance."
        ),
        "",
        f"![Iridium 33's approaches in the week after]({images}/{RANK})",
        "",
        *table(("Secondary", "TCA (UTC)", "Miss", "Pc"), rows),
        "",
        (
            "SOCRATES ranked conjunctions across the whole catalog and the whole "
            f"Iridium constellation: the collision's overall rank went from "
            f"{KELSO['overall_rank']}, and among all Iridium conjunctions from "
            f"{KELSO['iridium_rank']}; Iridium 33 alone saw {KELSO['weekly']} "
            f"out of {KELSO['constellation_weekly']} for the whole constellation "
            "(Kelso (2009)). For one satellite, as here, the list is short; an "
            "operator "
            "of 66 had to sort about a thousand a week, on Pc no better than this."
        ),
    ]


def _maneuver_section(summary: Summary, images: str) -> list[str]:
    settings, plan = summary["settings"], summary["maneuver"]
    rows = [
        (
            option["direction"],
            f"{option['burn_lead_hours']:.1f} h",
            f"{option['delta_v']:+.4f} m/s",
            _pc_text(option["pc"]),
            _km(option["range_at_tca_m"]),
            f"{option['shift_km']:g} km",
            f"{option['new']} new, {option['closer']} closer",
            _pc_text(option["worst_pc"]),
            "yes" if option["conflict"] else "no",
        )
        for option in plan["options"]
    ]
    return [
        "## A manoeuvre",
        "",
        (
            "Had the operator chosen to act at the decision point, `plan_maneuver` on "
            "the "
            "CDMs known then gives the smallest along-track burns, from "
            f"{settings['burn_earliest_hours']:g} h to "
            f"{settings['burn_latest_hours']:g} h "
            f"before TCA, that bring the Pc ({_pc_text(plan['baseline_pc'])} without a "
            f"burn) below {plan['target']:.0e}, green; the planner's own default, red, "
            "the "
            "event already meets. Each burn is then re-screened against the excerpt "
            "for a "
            "week (`rescreen`): the approaches it brings, and the worst Pc among them."
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
            f"{settings['excerpt_radius_km']:g} km of it, and the study stops if a "
            "burn "
            "moves it more than half that, so no object the excerpt left out can be "
            "the one a burn brings near. A demonstration of the mechanism, not advice "
            "about what Iridium should have done."
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
            f"{settings['snapshot_retrieved'][:10]} "
            f"still holds {sum(debris['on_orbit'].values())} catalogued fragments of "
            "the "
            f"collision ({counts}: objects of the two launches' designators catalogued "
            "after it). Screened against that catalog for one day from its retrieval, "
            "in the default 5 km sphere, they came within 5 km of other objects "
            f"{debris['approaches']} times. The closest:"
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
            "- Covariance: keepout's GP element-set age model, assumed (GP age model), "
            "not a calibrated product ([Assumptions](assumptions.md))."
        ),
        (
            "- No publication latency: a CDM is created at the epoch of its newer "
            "element set."
        ),
        (
            "- Corrections: of two sets of one object and epoch, the one issued later "
            "is "
            "used, even if it was issued after a CDM's creation time; the bulk file "
            "has "
            "no issue times."
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
        *_summary_section(summary, images),
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
                f"{decision['level']} (Pc {_pc_text(decision['latest_pc'])} on the "
                "assumed "
                f"covariance) and the rules said **{decision['action']}**; it ranked "
                f"{_ordinal(rank['by_pc'])} by Pc among Iridium 33's "
                f"{rank['approaches']} "
                f"approaches that week; a {abs(best['delta_v']) * 100:.1f} cm/s burn "
                "would "
                "have made it green."
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
    for day, value, level in zip(days, figure.pc, figure.levels, strict=True):
        axes.scatter([day], [value], s=36, zorder=2, color=RISK_COLOURS[level])
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
        2.0, color=DECISION_POINT_COLOUR, linestyle=":", label="decision point"
    )
    axes.set_yscale("log")
    axes.set_ylim(1e-8, 1e-3)
    axes.set_xlim(7.2, 0.0)
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
    for item in items:
        pc = max(item["pc"] or floor, floor)
        colour = RISK_COLOURS[
            RiskLevel.YELLOW if item["collision"] else RiskLevel.GREEN
        ]
        axes.scatter(
            [item["miss_m"] / KM],
            [pc],
            s=70 if item["collision"] else 30,
            color=colour if item["collision"] else MUTED_COLOUR,
            edgecolors=INK_COLOUR if item["collision"] else "none",
            zorder=3 if item["collision"] else 2,
            label="Cosmos 2251" if item["collision"] else None,
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
            f"Iridium 33 (24946) vs Cosmos 2251 (22675)\nTCA {tca} UTC, Pc "
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
