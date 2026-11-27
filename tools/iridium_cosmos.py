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
from astropy.time import Time
from generated import REPO

from keepout.events import DEFAULT_TCA_TOLERANCE, ConjunctionEvent
from keepout.omm import load
from keepout.pc import compute_pc
from keepout.risk import DEFAULT_DECISION_CONFIG, Assessment, assess
from keepout.synthetic import Fleet, FleetSatellite, run
from keepout.units import DAY, HOUR, KM

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

    from keepout.cdm import Cdm
    from keepout.omm import Omm

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

EXCERPT_RADIUS: Final = 100 * KM
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
