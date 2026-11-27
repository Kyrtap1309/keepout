"""The Iridium 33-Cosmos 2251 case study: its constants and data, shared by
the report (``tools/iridium_cosmos_report.py``), the data tool
(``tools/iridium_cosmos_data.py``) and the dashboard seed.

On 2009 February 10 Iridium 33 (24946), an operational communications
satellite, and Cosmos 2251 (22675), a long-defunct one, collided over northern
Russia (Kelso 2009). The study runs keepout on the element sets that were
public then: the 2009 bulk TLE file of Space-Track.org, reduced to the
committed excerpt in :data:`DATA` by the data tool.

Times. :data:`TCA_REFERENCE` is the collision time Kelso (2009, sec. 3) uses:
the time of closest approach predicted by the last SOCRATES report before it.
:data:`IRIDIUM_AS_OF` is the decision point two days before
(:data:`~keepout.risk.DEFAULT_DECISION_CONFIG`). An analysis "as known at" a
time sees only the element sets with an epoch up to that time
(:func:`known_at`).

References
----------
T.S. Kelso, "Analysis of the Iridium 33-Cosmos 2251 Collision", 10th AMOS
Conference, Maui, 2009 September 2.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Final

import astropy.units as u
import numpy as np
from astropy.time import Time
from generated import REPO

from keepout.events import DEFAULT_TCA_TOLERANCE, ConjunctionEvent
from keepout.maneuver import (
    BurnDirection,
    BurnGrid,
    ManeuverOption,
    ManeuverPlan,
    RescreenResult,
    plan_maneuver,
    rescreen,
)
from keepout.omm import load
from keepout.pc import compute_pc
from keepout.propagation import ManeuveredPropagator, Sgp4Propagator
from keepout.risk import (
    DEFAULT_DECISION_CONFIG,
    DEFAULT_THRESHOLDS,
    Assessment,
    assess,
)
from keepout.screening import DEFAULT_DURATION, DEFAULT_VOLUME, screen
from keepout.synthetic import Fleet, FleetSatellite, OperatorEvent, OperatorResult, run
from keepout.units import DAY, HOUR, KM, MINUTE

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

    from keepout.cdm import Cdm
    from keepout.omm import Omm
    from keepout.screening import CloseApproach

IRIDIUM_33: Final = 24946
"""Iridium 33's catalog number (international designator 1997-051C)."""

COSMOS_2251: Final = 22675
"""Cosmos 2251's catalog number (international designator 1993-036A)."""

PAIR: Final = (IRIDIUM_33, COSMOS_2251)
"""The two objects that collided, Iridium 33 first."""

NAMES: Final[Mapping[int, str]] = {IRIDIUM_33: "IRIDIUM 33", COSMOS_2251: "COSMOS 2251"}
"""The pair's names: the bulk file's TLEs carry none."""

TCA_REFERENCE: Final = Time("2009-02-10T16:55:59.806", scale="utc")
"""The collision time: the TCA of the last SOCRATES report before it (Kelso
2009, sec. 3)."""

IRIDIUM_AS_OF: Final = TCA_REFERENCE - DEFAULT_DECISION_CONFIG.decision_point * u.s
"""The decision point, two days before :data:`TCA_REFERENCE`: the time the
dashboard shows the event at and the ranking sees the catalog at."""

PAIR_SPAN: Final = 30 * DAY
"""How long before :data:`TCA_REFERENCE` the pair's element sets are kept [s]."""

SERIES_SPAN: Final = 7 * DAY
"""How long before :data:`TCA_REFERENCE` the synthetic CDMs are issued [s]."""

BASE_HBR: Final = 6.0
"""The combined hard-body radius [m]: an enclosing-sphere estimate (Iridium 33
with its arrays about 4 m, Cosmos 2251 about 2 m), an assumption."""

IRIDIUM_HBR: Final = 4.0
"""Iridium 33's part of :data:`BASE_HBR` [m]; Cosmos 2251 has the rest."""

HBR_SENSITIVITY: Final = (5.0, 10.0, 20.0)
"""Further combined radii [m] the Pc is shown at."""

EXCERPT_RADIUS: Final = 300 * KM
"""Radius of the sphere around Iridium 33 within which an object's approach
puts it in the excerpt [m]."""

DATA: Final = REPO / "tests/case_study/data/iridium-cosmos"
"""The committed excerpt, its NOTICE and manifest."""

CATALOG_FILE: Final = "iridium-cosmos-2009.csv.gz"
"""The excerpt's element sets, as gzipped CelesTrak CSV."""


def known_at(omms: Iterable[Omm], time: Time) -> tuple[Omm, ...]:
    """The element sets with an epoch up to ``time``: what was known then."""
    return tuple(omm for omm in omms if omm.epoch <= time)


def with_name(omm: Omm) -> Omm:
    """``omm`` with its object's name from :data:`NAMES`, if it has one there."""
    name = NAMES.get(omm.catalog_number)
    return omm if name is None else dataclasses.replace(omm, object_name=name)


def read_catalog(data: Path = DATA, *, named: bool = True) -> tuple[Omm, ...]:
    """The excerpt's element sets; with ``named``, the pair's named
    (:func:`with_name`)."""
    omms = load(data / CATALOG_FILE)
    return tuple(with_name(omm) for omm in omms) if named else omms


def read_pair(data: Path = DATA) -> tuple[Omm, ...]:
    """The pair's element sets from :data:`PAIR_SPAN` before the collision."""
    start = TCA_REFERENCE - PAIR_SPAN * u.s
    return tuple(
        omm
        for omm in read_catalog(data)
        if omm.catalog_number in PAIR and omm.epoch >= start
    )


MAX_PC_METHODS: Final = ("frisbee", "dilution", "alfano")
"""The maximum-Pc methods shown beside the model's Pc."""


@dataclasses.dataclass(frozen=True, slots=True)
class Issue:
    """One synthetic CDM issue: the time and its CDM of the collision's
    approach, or why there is none.

    Parameters
    ----------
    time
        The issue time: the epoch of the newest element set of either object.
    cdm
        The CDM of the approach at the collision, ``None`` if not found.
    reason
        Why there is no CDM; ``None`` when there is one.
    """

    time: Time
    cdm: Cdm | None
    reason: str | None


def fleet(hbr: float = BASE_HBR) -> Fleet:
    """Iridium 33 as a one-satellite fleet, manoeuvrable, the combined radius
    ``hbr`` split as :data:`IRIDIUM_HBR` is in :data:`BASE_HBR`."""
    share = IRIDIUM_HBR / BASE_HBR
    return Fleet(
        satellites=(
            FleetSatellite(
                IRIDIUM_33, hbr * share, maneuverable=True, name=NAMES[IRIDIUM_33]
            ),
        ),
        catalog_hard_body_radius=hbr * (1 - share),
    )


def _issue(pair: Sequence[Omm], time: Time, hbr: float) -> Issue:
    """The CDM issued at ``time`` from the pair's sets known then."""
    known = known_at(pair, time)
    for number in PAIR:
        if not any(omm.catalog_number == number for omm in known):
            return Issue(time, None, f"no element set of {number} known")
    duration = (TCA_REFERENCE - time).to_value(u.s) + HOUR
    result = run(fleet(hbr), known, time, duration, created=time)
    for event in result.events:
        approach = event.approach
        offset = abs((approach.tca - TCA_REFERENCE).to_value(u.s))
        if approach.secondary_number == COSMOS_2251 and offset <= DEFAULT_TCA_TOLERANCE:
            return Issue(time, event.cdm, None)
    return Issue(
        time,
        None,
        f"no approach within {DEFAULT_TCA_TOLERANCE:g} s of the collision",
    )


def cdm_series(pair: Sequence[Omm], *, hbr: float = BASE_HBR) -> tuple[Issue, ...]:
    """A synthetic CDM at each new element set of the pair in the
    :data:`SERIES_SPAN` before the collision.

    Each issue screens Iridium 33 against Cosmos 2251 as known at its time
    (:func:`known_at`) to an hour past the collision, with the synthetic
    operator (:func:`keepout.synthetic.run`, the GP age covariance model) and
    the combined radius ``hbr``; created at the issue time (no publication
    latency). Its CDM is the approach within the event TCA tolerance of
    :data:`TCA_REFERENCE`.
    """
    start = TCA_REFERENCE - SERIES_SPAN * u.s
    times = sorted(
        {
            omm.epoch.isot: omm.epoch
            for omm in pair
            if start <= omm.epoch < TCA_REFERENCE
        }.values(),
        key=lambda time: time.jd,
    )
    return tuple(_issue(pair, time, hbr) for time in times)


def collision_event(issues: Iterable[Issue]) -> ConjunctionEvent:
    """The event of the issues' CDMs."""
    return ConjunctionEvent.from_cdms(
        issue.cdm for issue in issues if issue.cdm is not None
    )


def known_cdms(event: ConjunctionEvent, time: Time) -> tuple[Cdm, ...]:
    """The event's CDMs created up to ``time``."""
    return tuple(cdm for cdm in event.cdms if cdm.creation_date <= time)


def max_pcs(cdm: Cdm, hbr: float) -> dict[str, float]:
    """The maximum Pc of each of :data:`MAX_PC_METHODS` for the CDM's
    conjunction with the combined radius ``hbr`` [m]: Frisbee's over Cosmos
    2251's covariance taken as unknown, dilution's over a scale of the
    combined covariance, Alfano's over every covariance."""
    conjunction = cdm.conjunction(combined_hard_body_radius=hbr)
    unknown_secondary = dataclasses.replace(
        conjunction,
        secondary=dataclasses.replace(conjunction.secondary, covariance=None),
    )
    return {
        "frisbee": compute_pc(unknown_secondary, "frisbee").value,
        "dilution": compute_pc(conjunction, "dilution").value,
        "alfano": compute_pc(conjunction, "alfano").value,
    }


def assessments(event: ConjunctionEvent) -> tuple[Assessment, Assessment]:
    """The event assessed at the decision point (its CDMs created up to
    :data:`IRIDIUM_AS_OF`) and on all its CDMs."""
    observations = event.observations()
    known = {cdm.message_id for cdm in known_cdms(event, IRIDIUM_AS_OF)}
    at_decision = [o for o in observations if o.message_id in known]
    return assess(at_decision), assess(observations)


def hbr_sensitivity(event: ConjunctionEvent) -> dict[float, tuple[float, float]]:
    """The Pc at each radius of :data:`BASE_HBR` and :data:`HBR_SENSITIVITY`
    [m], ascending: of the last CDM known at the decision point and of the
    last CDM."""
    at_decision = known_cdms(event, IRIDIUM_AS_OF)[-1]
    last = event.latest

    def pc(cdm: Cdm, radius: float) -> float:
        return compute_pc(cdm.conjunction(combined_hard_body_radius=radius)).value

    return {
        radius: (pc(at_decision, radius), pc(last, radius))
        for radius in sorted({BASE_HBR, *HBR_SENSITIVITY})
    }


BURN_GRID: Final = BurnGrid(earliest_lead=2 * DAY, latest_lead=2 * HOUR)
"""The burns tried: from the decision point to 2 hours before TCA."""

MANEUVER_TARGET: Final = DEFAULT_THRESHOLDS.green_below
"""The Pc a burn must bring the event below: green. The planner's default
target, red, is met without a burn by an event that is yellow."""

SHIFT_STEP: Final = MINUTE
"""Time between the states compared for a burn's position change [s]."""


class ExcerptTooSmall(Exception):
    """A burn moves Iridium 33 further than the excerpt's radius covers."""


@dataclasses.dataclass(frozen=True, slots=True)
class Ranked:
    """The collision's approach among every approach of Iridium 33 in the
    7 days from the decision point.

    Parameters
    ----------
    result
        The synthetic operator's run: screening, a CDM and Pc per approach.
    collision
        The event of the approach at the collision.
    by_pc, by_miss
        Its rank, from 1, by Pc (highest first; an approach without a Pc
        last) and by miss distance (smallest first).
    """

    result: OperatorResult
    collision: OperatorEvent
    by_pc: int
    by_miss: int

    @property
    def approaches(self) -> int:
        """How many approaches Iridium 33 had in the window."""
        return len(self.result.events)


def rank(excerpt: Sequence[Omm]) -> Ranked:
    """Iridium 33 screened against the excerpt as known at
    :data:`IRIDIUM_AS_OF`, for :data:`~keepout.screening.DEFAULT_DURATION`,
    in the default 5 km sphere, every approach with the same combined radius
    :data:`BASE_HBR` and the model's covariance; the collision's ranks.

    Raises
    ------
    LookupError
        If no approach of Cosmos 2251 lies within the TCA tolerance of
        :data:`TCA_REFERENCE`.
    """
    known = known_at(excerpt, IRIDIUM_AS_OF)
    result = run(fleet(), known, IRIDIUM_AS_OF, DEFAULT_DURATION, workers=None)
    events = result.events
    collision = next(
        (
            event
            for event in events
            if event.approach.secondary_number == COSMOS_2251
            and abs((event.approach.tca - TCA_REFERENCE).to_value(u.s))
            <= DEFAULT_TCA_TOLERANCE
        ),
        None,
    )
    if collision is None:
        raise LookupError("no approach of Cosmos 2251 at the collision")

    def pc_key(event: OperatorEvent) -> float:
        pc = event.observation.pc
        return -1.0 if pc is None else pc

    by_pc = sorted(events, key=pc_key, reverse=True)
    by_miss = sorted(events, key=lambda event: event.approach.range_at_tca)
    return Ranked(
        result=result,
        collision=collision,
        by_pc=by_pc.index(collision) + 1,
        by_miss=by_miss.index(collision) + 1,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class Maneuver:
    """The plan of a burn at the decision point, each option re-screened.

    Parameters
    ----------
    plan
        The plan on the event's CDMs known at :data:`IRIDIUM_AS_OF`.
    target
        The Pc the burns had to bring the event below.
    rescreens
        Each option's re-screen against the excerpt, by direction.
    shifts
        Each option's largest change of Iridium 33's position [m] over the
        re-screen's window, by direction.
    """

    plan: ManeuverPlan
    target: float
    rescreens: Mapping[BurnDirection, RescreenResult]
    shifts: Mapping[BurnDirection, float]


def largest_shift(plan: ManeuverPlan, option: ManeuverOption, primary: Omm) -> float:
    """The largest distance [m] between Iridium 33 with the burn of
    ``option`` and without it, every :data:`SHIFT_STEP` over the
    re-screen's window (the burn, then 7 days)."""
    base = Sgp4Propagator([primary])
    burn = plan.conjunction.tca - option.burn_lead * u.s
    burned = ManeuveredPropagator(base, burn, option.delta_v, DEFAULT_DURATION + HOUR)
    offsets = np.arange(0.0, DEFAULT_DURATION + SHIFT_STEP, SHIFT_STEP)
    before = base.propagate(burn, offsets).r
    after = burned.propagate(burn, offsets).r
    return float(np.max(np.linalg.norm(after - before, axis=-1)))


def require_within_excerpt(shifts: Mapping[BurnDirection, float]) -> None:
    """A check that no burn moves Iridium 33 more than half
    :data:`EXCERPT_RADIUS`: beyond it, an object the excerpt left out could
    come near.

    Raises
    ------
    ExcerptTooSmall
        If a shift is larger.
    """
    limit = EXCERPT_RADIUS / 2
    for direction, shift in shifts.items():
        if shift > limit:
            raise ExcerptTooSmall(
                f"the {direction} burn moves Iridium 33 by {shift:.0f} m, more than "
                f"half the excerpt's radius ({EXCERPT_RADIUS:g} m)"
            )


def maneuver(event: ConjunctionEvent, excerpt: Sequence[Omm]) -> Maneuver:
    """The burns of :data:`BURN_GRID` planned on the event as known at the
    decision point to :data:`MANEUVER_TARGET`, each option re-screened
    against the excerpt as known then.

    Raises
    ------
    ExcerptTooSmall
        As :func:`require_within_excerpt`.
    """
    known_event = ConjunctionEvent.from_cdms(
        known_cdms(event, IRIDIUM_AS_OF), event_id=event.event_id
    )
    plan = plan_maneuver(known_event, target_pc=MANEUVER_TARGET, grid=BURN_GRID)
    catalog = known_at(excerpt, IRIDIUM_AS_OF)
    primary = max(
        (omm for omm in catalog if omm.catalog_number == IRIDIUM_33),
        key=lambda omm: omm.epoch.jd,
    )
    shifts = {
        option.direction: largest_shift(plan, option, primary)
        for option in plan.options
    }
    require_within_excerpt(shifts)
    rescreens = {
        option.direction: rescreen(plan, option, catalog, workers=None)
        for option in plan.options
    }
    return Maneuver(
        plan=plan, target=MANEUVER_TARGET, rescreens=rescreens, shifts=shifts
    )


DEBRIS_PREFIXES: Final[Mapping[str, str]] = {
    "1997-051": NAMES[IRIDIUM_33],
    "1993-036": NAMES[COSMOS_2251],
}
"""The international designators of the collision's fragments, by launch:
each object's fragments share its launch's designator."""

DEBRIS_FIRST_NUMBER: Final = 33000
"""Catalog numbers of the fragments start above this (catalogued after the
collision); the launches' own pieces (satellites, rocket bodies, earlier
fragments) have lower ones."""

DEBRIS_CLOSEST: Final = 5
"""How many of the fragments' closest approaches are listed."""


def is_debris(omm: Omm) -> bool:
    """Whether ``omm`` is a fragment of the collision: a designator of
    :data:`DEBRIS_PREFIXES` and a number above :data:`DEBRIS_FIRST_NUMBER`."""
    return (
        omm.catalog_number > DEBRIS_FIRST_NUMBER
        and omm.international_designator[:8] in DEBRIS_PREFIXES
    )


@dataclasses.dataclass(frozen=True, slots=True)
class DebrisToday:
    """The collision's fragments in a catalog of today.

    Parameters
    ----------
    on_orbit
        Fragments in the catalog, by parent's name.
    approaches
        Approaches of a fragment with an object that is no fragment, in the
        day screened.
    closest
        The :data:`DEBRIS_CLOSEST` closest of those, nearest first.
    """

    on_orbit: Mapping[str, int]
    approaches: int
    closest: tuple[CloseApproach, ...]


def debris_today(
    catalog: Sequence[Omm], start: Time, *, workers: int | None = None
) -> DebrisToday:
    """The fragments in ``catalog`` (one set per object), screened against
    it for a day from ``start`` in the default 5 km sphere."""
    pieces = [omm for omm in catalog if is_debris(omm)]
    numbers = {omm.catalog_number for omm in pieces}
    counts = dict.fromkeys(DEBRIS_PREFIXES.values(), 0)
    for omm in pieces:
        counts[DEBRIS_PREFIXES[omm.international_designator[:8]]] += 1
    result = screen(pieces, catalog, start, DAY, DEFAULT_VOLUME, workers=workers)
    others = sorted(
        (a for a in result.approaches if a.secondary_number not in numbers),
        key=lambda approach: approach.range_at_tca,
    )
    return DebrisToday(
        on_orbit=counts,
        approaches=len(others),
        closest=tuple(others[:DEBRIS_CLOSEST]),
    )
