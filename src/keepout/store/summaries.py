"""What the store reads out: the events list of the dashboard (a page of
stored events, each summed up), an event's detail, and the jobs of the
queue and of the manoeuvre planner.

:meth:`keepout.store.Store.summaries` reads a page of events
(:class:`SummaryPage`) in a fixed number of queries, whatever its length
(:func:`keepout.store.queries.read_page`): the count, the page's events,
their CDMs with the Pc cached for the settings, the latest CDM of each, and
their active jobs. An event's latest CDM is parsed only when its Pc rests on
an assumed covariance, whose provenance its header comments tell
(:func:`keepout.risk.provenance`). Each event's risk is
:func:`~keepout.risk.assess` of the Pc observations cached for it: none is
computed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import keepout.cdm
from keepout.risk import assess, provenance
from keepout.settings import settings_from_data
from keepout.store.codec import observation_from_json

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from astropy.time import Time

    from keepout.events import ConjunctionEvent
    from keepout.maneuver import ManeuverOutcome
    from keepout.risk import Assessment, PcObservation, Provenance, Thresholds
    from keepout.settings import Settings
    from keepout.store import models
    from keepout.store.models import EventId, JobKind, JobStatus, MessageId

__all__ = [
    "MAX_LIMIT",
    "EventDetail",
    "EventRows",
    "EventSummary",
    "Job",
    "LatestCdm",
    "PlanDetail",
    "PlanJobs",
    "SummaryPage",
    "assessed",
    "check_page",
    "complete_of",
    "decoded",
    "summarize",
]

MAX_LIMIT: Final = 200
"""Most events a page holds."""


@dataclass(frozen=True, slots=True)
class EventSummary:
    """A stored event as the events list shows it.

    Parameters
    ----------
    event_id
        The store's id of the event.
    object1, object2
        The objects' designators (``OBJECT_DESIGNATOR``).
    tca
        The TCA of the event's latest CDM, in UTC. The list filters and
        orders on the event's latest TCA instead (``events.tca_max``, the
        latest of its CDMs' TCAs); the two differ by at most
        :data:`~keepout.events.DEFAULT_TCA_TOLERANCE`, the spread of an
        event's TCAs.
    cdm_count
        How many CDMs the store holds of it.
    latest_creation
        The ``CREATION_DATE`` of its latest CDM, in UTC.
    latest
        The Pc observation of its latest CDM cached for the settings;
        ``None`` if it is not computed.
    assessment
        The risk of the event, assessed from the observations of all its
        CDMs; ``None`` if any of them is not computed.
    provenance
        What the latest Pc rests on; ``None`` if it is not computed.
    active_job
        The kind of the oldest queued or running job of the event; ``None``
        if there is none.
    """

    event_id: EventId
    object1: str
    object2: str
    tca: Time
    cdm_count: int
    latest_creation: Time
    latest: PcObservation | None
    assessment: Assessment | None
    provenance: Provenance | None
    active_job: JobKind | None


@dataclass(frozen=True, slots=True)
class EventDetail:
    """A stored event as its page shows it, read in one transaction.

    Parameters
    ----------
    event
        The event and its CDMs.
    observations
        Each CDM's ``MESSAGE_ID`` and its Pc observation cached for the
        settings (``None`` if it is not computed), oldest CDM first.
    refresh_jobs
        The ids of the event's queued or running ``refresh`` jobs for the
        settings, oldest first.
    """

    event: ConjunctionEvent
    observations: tuple[tuple[MessageId, PcObservation | None], ...]
    refresh_jobs: tuple[int, ...]

    @property
    def missing(self) -> tuple[MessageId, ...]:
        """The CDMs whose Pc is not computed, oldest first."""
        return tuple(
            message_id
            for message_id, observation in self.observations
            if observation is None
        )

    @property
    def complete(self) -> tuple[PcObservation, ...] | None:
        """Every CDM's observation, oldest first; ``None`` if any is not
        computed."""
        return complete_of([item for _, item in self.observations])

    def assessment(self, thresholds: Thresholds) -> Assessment | None:
        """The event's risk with ``thresholds`` (:func:`assessed`); ``None``
        if any Pc is not computed."""
        return assessed([item for _, item in self.observations], thresholds)


def decoded(
    payloads: Iterable[Mapping[str, object] | None],
) -> tuple[PcObservation | None, ...]:
    """The observations of cached payloads, in order; ``None`` where there
    is no payload (the Pc is not computed).

    Raises
    ------
    StoreError
        If a payload is not an observation.
    """
    return tuple(
        None if payload is None else observation_from_json(payload)
        for payload in payloads
    )


def complete_of(
    observations: Sequence[PcObservation | None],
) -> tuple[PcObservation, ...] | None:
    """The observations of an event's CDMs if every one is computed, else
    ``None``: an event is assessed only from all of them."""
    known = tuple(item for item in observations if item is not None)
    return known if len(known) == len(observations) else None


def assessed(
    observations: Sequence[PcObservation | None], thresholds: Thresholds
) -> Assessment | None:
    """The risk of an event from its CDMs' observations (oldest first) with
    ``thresholds``; ``None`` unless every one is computed
    (:func:`complete_of`), the events list and an event's page alike."""
    complete = complete_of(observations)
    return None if complete is None else assess(complete, thresholds=thresholds)


@dataclass(frozen=True, slots=True)
class SummaryPage:
    """A page of the events list.

    Parameters
    ----------
    items
        The page's events, by latest TCA (then by id).
    total
        How many events the list has, on every page.
    offset
        How many events of the list come before the page.
    limit
        Most events the page holds.
    """

    items: tuple[EventSummary, ...]
    total: int
    offset: int
    limit: int


@dataclass(frozen=True, slots=True)
class LatestCdm:
    """The latest CDM of an event, as stored."""

    tca: Time
    creation_date: Time
    content: str


@dataclass(frozen=True, slots=True)
class EventRows:
    """What the store holds of a listed event, as
    :func:`~keepout.store.queries.read_page` reads it for :func:`summarize`.

    Internal to :mod:`keepout.store`.
    """

    event_id: EventId
    object1: str
    object2: str
    payloads: Sequence[Mapping[str, object] | None]
    """The Pc cached on each of its CDMs, oldest first; ``None`` where there
    is none."""
    latest: LatestCdm
    active_job: JobKind | None


def _summary(rows: EventRows, thresholds: Thresholds) -> EventSummary:
    """The summary of a listed event.

    Raises
    ------
    StoreError
        If a cached observation cannot be read.
    CdmParseError
        If its latest CDM, parsed for an assumed covariance, no longer parses.
    """
    observations = decoded(rows.payloads)
    latest = observations[-1]
    assessment = assessed(observations, thresholds)
    badge = (
        None
        if latest is None
        else provenance(
            latest, lambda: keepout.cdm.loads(rows.latest.content).header.comments
        )
    )
    return EventSummary(
        event_id=rows.event_id,
        object1=rows.object1,
        object2=rows.object2,
        tca=rows.latest.tca,
        cdm_count=len(observations),
        latest_creation=rows.latest.creation_date,
        latest=latest,
        assessment=assessment,
        provenance=badge,
        active_job=rows.active_job,
    )


def summarize(
    events: Iterable[EventRows], thresholds: Thresholds
) -> tuple[EventSummary, ...]:
    """The summaries of listed events, each assessed from its cached
    observations, in the same order.

    Raises
    ------
    StoreError
        If a cached observation cannot be read.
    CdmParseError
        If a latest CDM, parsed for an assumed covariance, no longer parses.
    """
    return tuple(_summary(rows, thresholds) for rows in events)


def check_page(offset: int, limit: int) -> None:
    """Check a page's bounds: ``offset`` not negative, ``limit`` from 1 to
    :data:`MAX_LIMIT`.

    Raises
    ------
    ValueError
        If either is out of its bounds.
    """
    if not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit must be from 1 to {MAX_LIMIT}, got {limit}")
    if offset < 0:
        raise ValueError(f"offset must not be negative, got {offset}")


@dataclass(frozen=True, slots=True, eq=False)
class Job:
    """A job of the queue (:class:`~keepout.store.JobQueue`), as it was when
    read.

    Two jobs compare by identity (:class:`~keepout.settings.Settings` do):
    compare :attr:`id` and :attr:`claims`.
    """

    id: int
    """The job's number, unique in the store's queue."""
    kind: JobKind
    """What it computes: a manoeuvre plan, or the missing Pc."""
    event_id: str
    """The store's id of the event."""
    message_id: str
    """The CDM the computation is pinned to: the event's latest when
    queued."""
    key: str
    """The key its result is cached under."""
    settings: Settings
    """The operator's settings the job computes with."""
    status: JobStatus
    """Where the job is: queued, running, succeeded, failed or cancelled."""
    attempt: int
    """How many attempts it cost: a claim adds one, a release on a stop
    takes it back."""
    claims: int
    """How many times it was claimed, never lowered: a worker's writes name
    the claim they belong to."""
    worker: str | None
    """The worker that holds it, or held it when it succeeded, failed or was
    cancelled. Cleared when it goes back to the queue or its lease lapses."""
    created_at: Time
    """When it was queued, in UTC."""
    started_at: Time | None
    """When a worker last claimed it, in UTC; ``None`` while it waits in the
    queue (also after going back to it)."""
    finished_at: Time | None
    """When it succeeded, failed or was cancelled, in UTC; ``None`` while it
    is active."""
    error: str | None
    """Why it failed."""

    @classmethod
    def of(cls, row: models.Job) -> Job:
        """The job of a row.

        Parameters
        ----------
        row
            The job's row in the store.

        Returns
        -------
        Job
            The job.
        """
        return cls(
            id=row.id,
            kind=row.kind,
            event_id=row.event_id,
            message_id=row.message_id,
            key=row.key,
            settings=settings_from_data(row.params),
            status=row.status,
            attempt=row.attempt,
            claims=row.claims,
            worker=row.worker,
            created_at=row.created_at,
            started_at=row.started_at,
            finished_at=row.finished_at,
            error=row.error,
        )


@dataclass(frozen=True, slots=True)
class PlanJobs:
    """An event's manoeuvre jobs: the active ones, and the one of a set of
    settings (:meth:`~keepout.store.JobQueue.plan_jobs`)."""

    active: tuple[Job, ...]
    """Every manoeuvre job of the event still queued or running, whatever
    its settings, newest first."""
    current: Job | None
    """The job of the settings on the event's latest CDM: the queued or
    running one, else the newest (ended); ``None`` if there is none."""


@dataclass(frozen=True, slots=True)
class PlanDetail:
    """A stored event as its manoeuvre planner shows it, read in one
    transaction (:meth:`~keepout.store.Store.plan_detail`)."""

    event: ConjunctionEvent
    """The event and its CDMs."""
    message_id: MessageId
    """The CDM a plan of the event is cached on: its latest."""
    key: str
    """The plan's :func:`~keepout.store.codec.maneuver_key`: the settings
    and the catalog of that CDM."""
    outcome: ManeuverOutcome | None
    """The plan cached there, of :attr:`event`; ``None`` if none is."""
    jobs: PlanJobs
    """The event's manoeuvre jobs, and the one of the settings."""
