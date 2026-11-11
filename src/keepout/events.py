"""A conjunction event: the CDMs of one encounter of two objects."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, NamedTuple

from keepout._defaults import TCA_TOLERANCE
from keepout._validation import is_probability, require_non_negative
from keepout.cdm.model import Cdm
from keepout.covariance import Policy
from keepout.errors import EventError, HardBodyRadiusError, KeepoutError
from keepout.pc import AUTO, PcComputation
from keepout.risk.decision import Assessment
from keepout.risk.observation import PcObservation
from keepout.units import elapsed_seconds

if TYPE_CHECKING:
    from astropy.time import Time

__all__ = [
    "DEFAULT_TCA_TOLERANCE",
    "EVENT_ID_LENGTH",
    "AssessedEvent",
    "ConjunctionEvent",
    "EventBounds",
    "group_events",
    "stable_event_id",
    "time_to_tca",
    "without_hard_body_radius",
]

DEFAULT_TCA_TOLERANCE: Final = TCA_TOLERANCE
"""Largest spread [s] of the TCAs of one event's CDMs. Successive CDMs of an
encounter move its TCA by milliseconds to seconds; two encounters of the same
pair are at least a large fraction of an orbit apart."""


def time_to_tca(cdm: Cdm) -> float:
    """Seconds from the CDM's creation to its TCA (negative after TCA)."""
    return elapsed_seconds(cdm.creation_date, cdm.tca)


def without_hard_body_radius(cdms: Iterable[Cdm]) -> tuple[str, ...]:
    """The ``MESSAGE_ID`` of each CDM that lacks ``HBR`` for an object, in
    the order given: their Pc needs a combined hard-body radius from
    elsewhere.

    Parameters
    ----------
    cdms
        The CDMs.
    """
    return tuple(
        cdm.message_id
        for cdm in cdms
        if cdm.object1.hard_body_radius is None or cdm.object2.hard_body_radius is None
    )


EVENT_ID_LENGTH: Final = 12
"""Hexadecimal characters of an event id: at 100,000 events, two share one
with a chance of about one in six million."""


def stable_event_id(message_id: str) -> str:
    """The id of the event whose first CDM has this ``MESSAGE_ID``: the
    first :data:`EVENT_ID_LENGTH` hexadecimal characters of its SHA-256.

    The same CDMs give the same id, with a store or without, on any machine.
    """
    digest = hashlib.sha256(message_id.encode("utf-8")).hexdigest()
    return digest[:EVENT_ID_LENGTH]


def _pair(cdm: Cdm) -> tuple[str, str]:
    return cdm.object1.object_id, cdm.object2.object_id


def _reported(cdm: Cdm) -> tuple[float | None, tuple[str, ...]]:
    """The originator's Pc, if it is a probability; else a note saying why not."""
    reported = cdm.collision_probability
    if reported is None or is_probability(reported):
        return reported, ()
    return None, (f"reported COLLISION_PROBABILITY {reported} is not in [0, 1]",)


@dataclass(frozen=True, eq=False, slots=True)
class ConjunctionEvent:
    """The CDMs of one encounter, from the oldest to the latest.

    Parameters
    ----------
    event_id
        Identifier of the event.
    cdms
        The CDMs, in any order; kept sorted by ``CREATION_DATE`` (then
        ``MESSAGE_ID``).
    tca_tolerance
        Largest spread [s] of the CDMs' TCAs.

    Raises
    ------
    ValueError
        If ``event_id`` is empty or there are no CDMs.
    EventError
        If a ``MESSAGE_ID`` repeats, the CDMs are about different objects
        (``OBJECT1`` and ``OBJECT2`` by designator, in that order), their TCAs
        spread over more than ``tca_tolerance``, or they give different
        ``CONJUNCTION_ID`` values.
    """

    event_id: str
    cdms: tuple[Cdm, ...]
    tca_tolerance: float = DEFAULT_TCA_TOLERANCE

    def __post_init__(self) -> None:
        if not self.cdms:
            raise ValueError("an event needs at least one CDM")
        if not self.event_id:
            raise ValueError("event_id must not be empty")
        require_non_negative(self.tca_tolerance, "tca_tolerance")
        first = self.cdms[0]
        ordered = tuple(
            sorted(
                self.cdms,
                key=lambda cdm: (
                    elapsed_seconds(first.creation_date, cdm.creation_date),
                    cdm.message_id,
                ),
            )
        )
        object.__setattr__(self, "cdms", ordered)
        counts = Counter(cdm.message_id for cdm in ordered)
        repeated = sorted(i for i, n in counts.items() if n > 1)
        if repeated:
            raise EventError(f"MESSAGE_ID repeated: {', '.join(repeated)}")
        pairs = {_pair(cdm) for cdm in ordered}
        if len(pairs) > 1:
            listed = ", ".join(f"{a}/{b}" for a, b in sorted(pairs))
            raise EventError(f"CDMs of one event must share both objects, got {listed}")
        offsets = [elapsed_seconds(first.tca, cdm.tca) for cdm in ordered]
        spread = max(offsets) - min(offsets)
        if spread > self.tca_tolerance:
            raise EventError(
                f"TCAs spread over {spread:.3g} s, more than tca_tolerance "
                f"({self.tca_tolerance:g} s)"
            )
        conjunction_ids = {
            cdm.conjunction_id for cdm in ordered if cdm.conjunction_id is not None
        }
        if len(conjunction_ids) > 1:
            raise EventError(
                "CDMs of one event give different CONJUNCTION_ID values: "
                f"{', '.join(sorted(conjunction_ids))}"
            )

    @classmethod
    def from_cdms(
        cls,
        cdms: Iterable[Cdm],
        *,
        event_id: str | None = None,
        tca_tolerance: float = DEFAULT_TCA_TOLERANCE,
    ) -> ConjunctionEvent:
        """The event of ``cdms``, named by ``event_id`` or after its CDMs.

        Parameters
        ----------
        cdms
            The CDMs, in any order.
        event_id
            Identifier; if ``None``, the :func:`stable_event_id` of the first
            CDM by ``CREATION_DATE`` (then ``MESSAGE_ID``).
        tca_tolerance
            Largest spread [s] of the CDMs' TCAs.

        Raises
        ------
        ValueError
            As :class:`ConjunctionEvent`.
        """
        messages = tuple(cdms)
        if event_id is None and messages:
            reference = messages[0].creation_date
            first = min(
                messages,
                key=lambda m: (
                    elapsed_seconds(reference, m.creation_date),
                    m.message_id,
                ),
            )
            event_id = stable_event_id(first.message_id)
        return cls(event_id or "", messages, tca_tolerance)

    @property
    def primary_id(self) -> str:
        """``OBJECT_DESIGNATOR`` of ``OBJECT1``."""
        return _pair(self.cdms[0])[0]

    @property
    def secondary_id(self) -> str:
        """``OBJECT_DESIGNATOR`` of ``OBJECT2``."""
        return _pair(self.cdms[0])[1]

    @property
    def latest(self) -> Cdm:
        """The most recently created CDM."""
        return self.cdms[-1]

    @property
    def assumed_covariance(self) -> bool:
        """Whether a CDM of the event has an assumed covariance
        (:attr:`~keepout.cdm.Cdm.assumed_covariance`): its Pc rests on a model.
        """
        return any(cdm.assumed_covariance for cdm in self.cdms)

    def observations(
        self,
        *,
        compute: PcComputation = AUTO,
        policy: Policy = "repair",
        combined_hard_body_radius: float | None = None,
    ) -> tuple[PcObservation, ...]:
        """The Pc keepout computes for every CDM, oldest first.

        Each CDM's Pc is computed by ``compute``. In ``"auto"`` mode a slow
        encounter can take up to a minute per CDM, while
        ``compute=computation("foster")`` takes milliseconds. A CDM whose Pc
        cannot be computed (a :class:`~keepout.errors.KeepoutError`: a
        covariance or geometry problem, no convergence) gives an observation
        with the error instead of stopping the others. The originator's Pc is
        kept in ``reported_pc``.

        Parameters
        ----------
        compute
            The Pc method and its options, from
            :func:`~keepout.pc.computation`; ``"auto"`` by default. Its
            :attr:`~keepout.pc.PcComputation.label` is the source of a failed
            CDM's observation.
        policy
            Covariance policy; see :func:`keepout.covariance.make_covariance`.
        combined_hard_body_radius
            Combined radius [m]; if ``None``, the sum of both objects'
            ``HBR`` in each CDM.

        Returns
        -------
        tuple of PcObservation
            One observation per CDM, in the order of :attr:`cdms`. Each holds
            the time from the CDM's creation to TCA, keepout's Pc or the
            error, the originator's Pc and the diagnostics; pass them to
            :func:`~keepout.risk.assess`.

        Raises
        ------
        HardBodyRadiusError
            If ``combined_hard_body_radius`` is ``None`` and a CDM lacks ``HBR`` for an
            object (checked before any Pc is computed).
        """
        if combined_hard_body_radius is None:
            missing = without_hard_body_radius(self.cdms)
            if missing:
                raise HardBodyRadiusError(
                    f"no HBR for both objects in {', '.join(missing)}: pass "
                    "combined_hard_body_radius= (in metres)"
                )
        return tuple(
            _observe(cdm, compute, policy, combined_hard_body_radius)
            for cdm in self.cdms
        )


def _observe(
    cdm: Cdm,
    compute: PcComputation,
    policy: Policy,
    combined_hard_body_radius: float | None,
) -> PcObservation:
    """The observation of one CDM (see :meth:`ConjunctionEvent.observations`)."""
    reported, notes = _reported(cdm)
    when = time_to_tca(cdm)
    assumed = cdm.assumed_covariance
    try:
        conjunction = cdm.conjunction(
            combined_hard_body_radius=combined_hard_body_radius, policy=policy
        )
        result = compute(conjunction)
    except KeepoutError as exc:
        return PcObservation(
            time_to_tca=when,
            source=compute.label,
            error=f"{type(exc).__name__}: {exc}",
            message_id=cdm.message_id,
            reported_pc=reported,
            diagnostics=notes,
            assumed_covariance=assumed,
        )
    return PcObservation.from_result(
        result,
        time_to_tca=when,
        message_id=cdm.message_id,
        reported_pc=reported,
        assumed_covariance=assumed,
        notes=notes,
    )


class AssessedEvent(NamedTuple):
    """A conjunction event and its risk assessment
    (:func:`keepout.risk.assess` of its observations)."""

    event: ConjunctionEvent
    assessment: Assessment


@dataclass(frozen=True, slots=True)
class EventBounds:
    """What decides whether a CDM joins an event: the event's objects, the
    earliest and latest TCA of its CDMs and its ``CONJUNCTION_ID``.

    A CDM joins an event (:meth:`admits`) when it is about the same objects
    (``OBJECT1``, ``OBJECT2`` by designator), the TCAs with its own spread
    over no more than the tolerance, and neither gives a ``CONJUNCTION_ID``
    the other contradicts. :func:`group_events` and the conjunction store
    both group by this rule, one CDM at a time.

    Parameters
    ----------
    pair
        ``OBJECT_DESIGNATOR`` of ``OBJECT1`` and ``OBJECT2``.
    first_tca, last_tca
        The earliest and latest TCA of the event's CDMs.
    conjunction_id
        The ``CONJUNCTION_ID`` its CDMs give (CDM 2.0); ``None`` if none does.
    """

    pair: tuple[str, str]
    first_tca: Time
    last_tca: Time
    conjunction_id: str | None = None

    @classmethod
    def of(cls, cdm: Cdm) -> EventBounds:
        """The bounds of the event of ``cdm`` alone."""
        return cls(_pair(cdm), cdm.tca, cdm.tca, cdm.conjunction_id)

    @property
    def spread(self) -> float:
        """The spread [s] of the TCAs of the event's CDMs."""
        return elapsed_seconds(self.first_tca, self.last_tca)

    def including(self, cdm: Cdm) -> EventBounds:
        """The bounds of the event with ``cdm`` added, whether it fits or not."""
        first, last = self.first_tca, self.last_tca
        if elapsed_seconds(first, cdm.tca) < 0.0:
            first = cdm.tca
        if elapsed_seconds(last, cdm.tca) > 0.0:
            last = cdm.tca
        return EventBounds(
            self.pair, first, last, self.conjunction_id or cdm.conjunction_id
        )

    def admits(self, cdm: Cdm, *, tca_tolerance: float = DEFAULT_TCA_TOLERANCE) -> bool:
        """Whether ``cdm`` joins the event: same objects, the TCAs with its
        own within ``tca_tolerance`` [s], no other ``CONJUNCTION_ID``."""
        given = (self.conjunction_id, cdm.conjunction_id)
        return (
            _pair(cdm) == self.pair
            and (None in given or given[0] == given[1])
            and self.including(cdm).spread <= tca_tolerance
        )


def group_events(
    cdms: Iterable[Cdm], *, tca_tolerance: float = DEFAULT_TCA_TOLERANCE
) -> list[ConjunctionEvent]:
    """CDMs grouped into events by the rule of :class:`EventBounds`.

    The CDMs are taken in ``CREATION_DATE`` order (then by ``MESSAGE_ID``),
    as an operator receives them and a conjunction store ingests them, each
    joining the event that admits it, or starting a new one: a CDM whose TCA
    would spread the event's over more than ``tca_tolerance``, or whose
    ``CONJUNCTION_ID`` differs from the event's, starts another event.
    Events are named as in :meth:`ConjunctionEvent.from_cdms`.

    Parameters
    ----------
    cdms
        CDMs about any objects, in any order.
    tca_tolerance
        Largest spread [s] of the TCAs of one event.

    Returns
    -------
    list[ConjunctionEvent]
        The events, by their first TCA (then by objects).

    Raises
    ------
    ValueError
        If ``tca_tolerance`` is invalid.
    EventError
        If a CDM fits more than one event (it gives no ``CONJUNCTION_ID``,
        and two events of different ones are within ``tca_tolerance``), or
        a ``MESSAGE_ID`` is repeated within an event.
    """
    require_non_negative(tca_tolerance, "tca_tolerance")
    messages = list(cdms)
    if not messages:
        return []
    reference = messages[0].tca
    created = messages[0].creation_date
    bounds: list[EventBounds] = []
    members: list[list[Cdm]] = []
    by_pair: dict[tuple[str, str], list[int]] = {}
    for cdm in sorted(
        messages,
        key=lambda m: (elapsed_seconds(created, m.creation_date), m.message_id),
    ):
        indices = by_pair.setdefault(_pair(cdm), [])
        fitting = [
            i for i in indices if bounds[i].admits(cdm, tca_tolerance=tca_tolerance)
        ]
        if len(fitting) > 1:
            raise EventError(f"CDM {cdm.message_id} fits more than one event")
        if fitting:
            (index,) = fitting
            bounds[index] = bounds[index].including(cdm)
            members[index].append(cdm)
        else:
            indices.append(len(bounds))
            bounds.append(EventBounds.of(cdm))
            members.append([cdm])
    order = sorted(
        range(len(bounds)),
        key=lambda i: (elapsed_seconds(reference, bounds[i].first_tca), bounds[i].pair),
    )
    return [
        ConjunctionEvent.from_cdms(members[i], tca_tolerance=tca_tolerance)
        for i in order
    ]
