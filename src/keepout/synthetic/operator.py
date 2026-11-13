"""A synthetic operator: screening, CDMs and risk for a fleet, from GP data.

:func:`run` does on public data what a conjunction assessment service does
for an operator:

1. **Element sets.** Each fleet satellite's element set is taken from the
   catalog (the newest of its sets, as screening uses); a satellite with
   none is a :class:`~keepout.errors.FleetError`.
2. **Screening.** The fleet is screened against the catalog
   (:func:`keepout.screening.screen`).
3. **Conjunctions.** At each approach's TCA, both objects get the
   covariance of the :class:`SyntheticCovarianceModel` for their own
   element sets, and their hard-body radii: a fleet satellite its own, any
   other object the fleet's ``catalog_hard_body_radius``.
4. **CDMs.** Each conjunction becomes a CDM
   (:func:`keepout.cdm.make_cdm`; ``OBJECT1`` is the fleet satellite):
   ``ORIGINATOR`` ``KEEPOUT``, ``MESSAGE_FOR`` the satellite's name,
   ``CREATION_DATE`` the given ``created`` time (by default the start of
   the window, so a run gives the same messages again),
   ``COVARIANCE_METHOD`` ``DEFAULT`` with the model's comments,
   ``MANEUVERABLE`` from the fleet (``N/A`` for catalog objects),
   ``UNKNOWN`` for a name or international designator the element set
   lacks,
   and the screening window and volume.
5. **Risk.** Each CDM is assessed as its recipient would assess it: its Pc
   is computed from the message
   (:meth:`keepout.events.ConjunctionEvent.observations`) and the event is
   assessed (:func:`keepout.risk.assess`).

Every approach is an event of its own, with one CDM: two approaches of one
pair are distinct local minima of their range, however close in time. With
a single Pc the trend is undetermined, so the decision rests on the Pc's
risk level and the time from ``created`` to TCA. Two fleet satellites
approaching each other give two events, one for each as ``OBJECT1``, as a
screening service sends each satellite's operator its own CDM.

The covariances are an **assumption** (see
:mod:`keepout.synthetic.covariance_model`): the messages say so, and so must
anything shown from them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from keepout._validation import require_single_time
from keepout.ccsds.message import UNKNOWN
from keepout.cdm import (
    ORIGINATOR,
    Cdm,
    CovarianceMethod,
    Maneuverable,
    ObjectMetadata,
    ObjectType,
    make_cdm,
)
from keepout.errors import FleetError
from keepout.events import ConjunctionEvent
from keepout.omm import Omm, deduplicate
from keepout.pc import AUTO, PcComputation
from keepout.risk import (
    DEFAULT_DECISION_CONFIG,
    DEFAULT_THRESHOLDS,
    Action,
    Assessment,
    DecisionConfig,
    PcObservation,
    Thresholds,
    assess,
)
from keepout.screening import (
    DEFAULT_DURATION,
    DEFAULT_VOLUME,
    CloseApproach,
    ScreeningResult,
    ScreeningVolume,
    VolumeRule,
    screen,
)
from keepout.synthetic.covariance_model import SyntheticCovarianceModel

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from astropy.time import Time

    from keepout.covariance import Policy
    from keepout.synthetic.fleet import Fleet

__all__ = [
    "DEFAULT_MODEL",
    "HEADER_COMMENTS",
    "OperatorEvent",
    "OperatorResult",
    "run",
]

DEFAULT_MODEL: Final = SyntheticCovarianceModel()
"""The covariance model unless told otherwise: the published parameters
with the default safety factor."""

HEADER_COMMENTS: Final = (
    "keepout synthetic operator: screening of public GP data with SGP4",
    "Covariances are assumed (a model, not orbit determination)",
)
"""Header comments of the operator's CDMs."""


@dataclass(frozen=True, eq=False, slots=True)
class OperatorEvent:
    """One close approach of a fleet satellite, as a CDM and its assessment.

    Parameters
    ----------
    approach
        The approach screening found (the fleet satellite is the primary).
    event
        The event of its one CDM.
    observation
        keepout's Pc of the CDM (or why it could not be computed).
    assessment
        The event's risk assessment.
    """

    approach: CloseApproach
    event: ConjunctionEvent
    observation: PcObservation
    assessment: Assessment

    @property
    def cdm(self) -> Cdm:
        """The CDM of the approach."""
        return self.event.latest

    @property
    def action(self) -> Action:
        """The recommended action."""
        return self.assessment.decision.action


@dataclass(frozen=True, eq=False, slots=True)
class OperatorResult:
    """What a synthetic operator run found and decided.

    Parameters
    ----------
    fleet
        The fleet.
    screening
        The screening of the fleet against the catalog, with the element
        sets it left out.
    events
        One event per approach, in the screening's order (chronological).
    model
        The covariance model used.
    created
        ``CREATION_DATE`` of every CDM.
    """

    fleet: Fleet
    screening: ScreeningResult
    events: tuple[OperatorEvent, ...]
    model: SyntheticCovarianceModel
    created: Time

    @property
    def cdms(self) -> tuple[Cdm, ...]:
        """The CDM of every event, in order."""
        return tuple(event.cdm for event in self.events)


@dataclass(frozen=True, slots=True, kw_only=True)
class _RunContext:
    """How each approach becomes an event."""

    fleet: Fleet
    newest_by_number: Mapping[int, Omm]
    screening: ScreeningResult
    model: SyntheticCovarianceModel
    created: Time
    originator: str
    compute: PcComputation
    policy: Policy
    combined_hard_body_radius: float | None
    thresholds: Thresholds
    decision_config: DecisionConfig


def run(
    fleet: Fleet,
    catalog: Iterable[Omm],
    start: Time,
    duration: float = DEFAULT_DURATION,
    volume: ScreeningVolume | VolumeRule = DEFAULT_VOLUME,
    *,
    model: SyntheticCovarianceModel = DEFAULT_MODEL,
    workers: int | None = 1,
    created: Time | None = None,
    originator: str = ORIGINATOR,
    compute: PcComputation = AUTO,
    policy: Policy = "repair",
    combined_hard_body_radius: float | None = None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    decision_config: DecisionConfig = DEFAULT_DECISION_CONFIG,
) -> OperatorResult:
    """A fleet screened against a catalog, a CDM issued per approach, each assessed.

    See the module's description for the steps.

    Parameters
    ----------
    fleet
        The satellites to protect.
    catalog
        Element sets of the objects to screen them against; must hold the
        fleet satellites' own.
    start, duration, volume
        The window and the screening volume, as for
        :func:`keepout.screening.screen`.
    model
        The covariance model of every object.
    workers
        Number of processes to screen in, or ``None`` for the default (see
        :func:`~keepout.screening.screen`: a script calling with more than 1
        needs an ``if __name__ == "__main__":`` guard).
    created
        ``CREATION_DATE`` of the CDMs, and so the time their risk is judged
        at; ``start`` if ``None``.
    originator
        ``ORIGINATOR`` of the CDMs.
    compute
        Pc method and options, as for
        :meth:`keepout.events.ConjunctionEvent.observations`; ``"auto"``
        keeps the 2-D Pc for the model's position-only covariances.
    policy, combined_hard_body_radius
        Covariance policy and combined radius [m], as for
        :meth:`keepout.events.ConjunctionEvent.observations`; ``None``: the
        sum of the CDM's radii, from the fleet.
    thresholds, decision_config
        Risk thresholds and decision point, as for
        :func:`keepout.risk.assess`.

    Returns
    -------
    OperatorResult
        The screening, one event (CDM, Pc, assessment) per approach, the
        model and the creation time.

    Raises
    ------
    FleetError
        If a fleet satellite has no element set in the catalog.
    ValueError
        If ``start`` or ``created`` is not one time, or a screening argument
        is out of range (as :func:`~keepout.screening.screen`).

    Examples
    --------
    ```python
    from astropy.time import Time

    from keepout import omm
    from keepout.synthetic import load_fleet, run

    # Sample files in the repository.
    fleet = load_fleet("examples/synthetic-operator/fleet.toml")
    catalog = omm.load("tests/screening/data/celestrak-public.csv.gz")
    start = Time("2026-10-06T00:00:00", scale="utc")
    result = run(fleet, catalog, start, duration=86400.0)  # one day
    for item in result.events:
        pair = (item.approach.primary_number, item.approach.secondary_number)
        print(*pair, f"{item.observation.pc:.1e}", item.action.label)
    # 25544 58180 1.4e-30 ignore
    # 38771 35434 1.5e-17 ignore
    # 25544 58180 4.3e-59 ignore
    # 38771 31882 4.3e-33 ignore
    ```
    """
    elements = tuple(catalog)
    newest = deduplicate(elements).by_number
    missing = [number for number in fleet.catalog_numbers if number not in newest]
    if missing:
        raise FleetError(
            "no element set in the catalog for fleet satellite(s) "
            f"{', '.join(map(str, missing))}"
        )
    created = start if created is None else created
    require_single_time(created, "created")
    primaries = [newest[number] for number in fleet.catalog_numbers]
    screening = screen(primaries, elements, start, duration, volume, workers=workers)
    context = _RunContext(
        fleet=fleet,
        newest_by_number=newest,
        screening=screening,
        model=model,
        created=created,
        originator=originator,
        compute=compute,
        policy=policy,
        combined_hard_body_radius=combined_hard_body_radius,
        thresholds=thresholds,
        decision_config=decision_config,
    )
    events = tuple(_event(approach, context) for approach in screening.approaches)
    return OperatorResult(fleet, screening, events, model, created)


def _metadata(
    omm: Omm, context: _RunContext, tca: Time
) -> tuple[ObjectMetadata, float]:
    """What the CDM says about an object, and its hard-body radius."""
    satellite = context.fleet.find(omm.catalog_number)
    if satellite is None:
        name = omm.object_name
        maneuverable = Maneuverable.NOT_APPLICABLE
        object_type = None
        radius = context.fleet.catalog_hard_body_radius
    else:
        name = satellite.name or omm.object_name
        maneuverable = Maneuverable.YES if satellite.maneuverable else Maneuverable.NO
        object_type = ObjectType.PAYLOAD
        radius = satellite.hard_body_radius
    metadata = ObjectMetadata(
        name or UNKNOWN,
        omm.international_designator or UNKNOWN,
        object_type=object_type,
        maneuverable=maneuverable,
        covariance_method=CovarianceMethod.DEFAULT,
        covariance_comments=context.model.comments(omm, tca),
    )
    return metadata, radius


def _event(approach: CloseApproach, context: _RunContext) -> OperatorEvent:
    """The CDM of an approach and its assessment."""
    tca = approach.tca
    primary = context.newest_by_number[approach.primary_number]
    secondary = context.newest_by_number[approach.secondary_number]
    primary_metadata, primary_radius = _metadata(primary, context, tca)
    secondary_metadata, secondary_radius = _metadata(secondary, context, tca)
    model = context.model
    conjunction = approach.to_conjunction(
        primary_covariance=model.covariance(primary, tca),
        secondary_covariance=model.covariance(secondary, tca),
        primary_radius=primary_radius,
        secondary_radius=secondary_radius,
    )
    screening = context.screening
    cdm = make_cdm(
        conjunction,
        primary_metadata,
        secondary_metadata,
        created=context.created,
        originator=context.originator,
        message_for=primary_metadata.name,
        screening_period=(screening.start, screening.end),
        screening_volume=screening.volumes[approach.primary_number],
        comments=HEADER_COMMENTS,
    )
    event = ConjunctionEvent.from_cdms([cdm])
    (observation,) = event.observations(
        compute=context.compute,
        policy=context.policy,
        combined_hard_body_radius=context.combined_hard_body_radius,
    )
    assessment = assess(
        [observation],
        thresholds=context.thresholds,
        decision_config=context.decision_config,
    )
    return OperatorEvent(approach, event, observation, assessment)
