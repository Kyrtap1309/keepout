"""The events list of the dashboard: one page of stored events, each summed up.

:meth:`keepout.store.Store.summaries` reads a page of events
(:class:`SummaryPage`) in a fixed number of queries, whatever its length:
the count, the page's events, their CDMs with the Pc cached for the
settings, the latest CDM of each, and their active jobs. Only the latest CDM
of an event is parsed, for its Pc's provenance. Each event's risk is
:func:`~keepout.risk.assess` of the Pc observations cached for it: none is
computed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from sqlalchemy import and_, func, select

import keepout.cdm
from keepout.figures.provenance import provenance
from keepout.risk import assess
from keepout.store import models
from keepout.store.codec import observation_from_json
from keepout.store.models import JobKind, JobStatus

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from astropy.time import Time
    from sqlalchemy import ColumnElement
    from sqlalchemy.orm import Session

    from keepout.events import ConjunctionEvent
    from keepout.figures.provenance import Provenance
    from keepout.risk import Assessment, PcObservation, Thresholds
    from keepout.store.models import MessageId

__all__ = [
    "MAX_LIMIT",
    "EventDetail",
    "EventSummary",
    "PageRows",
    "SummaryPage",
    "active_refresh_in",
    "check_page",
    "complete_of",
    "decoded",
    "read_page",
]

MAX_LIMIT: Final = 200
"""Most events a page holds."""

_ACTIVE: Final = tuple(status for status in JobStatus if status.is_active)
"""The statuses of a job that has not ended."""


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

    event_id: str
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
class _Latest:
    """The latest CDM of an event, as stored."""

    tca: Time
    creation_date: Time
    content: str


@dataclass(frozen=True, slots=True)
class _EventRows:
    """What the store holds of a listed event."""

    event_id: str
    object1: str
    object2: str
    payloads: Sequence[Mapping[str, object] | None]
    """The Pc cached on each of its CDMs, oldest first; ``None`` where there
    is none."""
    latest: _Latest
    active_job: JobKind | None


@dataclass(frozen=True, slots=True)
class PageRows:
    """A page of the events list as read from the store, nothing parsed."""

    events: tuple[_EventRows, ...]
    total: int
    offset: int
    limit: int

    def page(self, thresholds: Thresholds) -> SummaryPage:
        """The page, each event assessed from its cached observations.

        Parameters
        ----------
        thresholds
            The thresholds of the risk levels.

        Returns
        -------
        SummaryPage
            The page.

        Raises
        ------
        StoreError
            If a cached observation cannot be read.
        CdmParseError
            If a latest CDM no longer parses (a store edited by hand).
        """
        return SummaryPage(
            tuple(_summary(rows, thresholds) for rows in self.events),
            self.total,
            self.offset,
            self.limit,
        )


def _summary(rows: _EventRows, thresholds: Thresholds) -> EventSummary:
    """The summary of a listed event.

    Raises
    ------
    StoreError
        If a cached observation cannot be read.
    CdmParseError
        If its latest CDM no longer parses.
    """
    observations = decoded(rows.payloads)
    latest = observations[-1]
    complete = complete_of(observations)
    assessment = None if complete is None else assess(complete, thresholds=thresholds)
    badge = (
        None
        if latest is None
        else provenance(latest, keepout.cdm.loads(rows.latest.content))
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


def check_page(offset: int, limit: int) -> None:
    """Check the bounds of a page.

    Parameters
    ----------
    offset
        How many events come before the page: not negative.
    limit
        Most events the page holds: 1 to :data:`MAX_LIMIT`.

    Raises
    ------
    ValueError
        If either is out of its bounds.
    """
    if not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit must be from 1 to {MAX_LIMIT}, got {limit}")
    if offset < 0:
        raise ValueError(f"offset must not be negative, got {offset}")


def read_page(
    session: Session,
    key: str,
    *,
    as_of: Time | None,
    offset: int,
    limit: int,
) -> PageRows:
    """A page of the events list, read in the caller's transaction.

    Internal to :mod:`keepout.store`: :meth:`~keepout.store.Store.summaries`
    reads it, then parses and assesses it outside the transaction.

    Parameters
    ----------
    session
        A session of the store.
    key
        The :func:`~keepout.store.codec.result_key` of the observations.
    as_of
        Only the events whose latest TCA is not before it; ``None``: all.
    offset
        How many events come before the page.
    limit
        Most events the page holds.

    Returns
    -------
    PageRows
        The page's rows.
    """
    listed: list[ColumnElement[bool]] = []
    if as_of is not None:
        listed.append(models.Event.tca_max >= as_of)
    total = session.scalar(
        select(func.count()).select_from(models.Event).where(*listed)
    )
    events = session.execute(
        select(models.Event.id, models.Event.object1, models.Event.object2)
        .where(*listed)
        .order_by(models.Event.tca_max, models.Event.id)
        .offset(offset)
        .limit(limit)
    ).all()
    ids = [event_id for event_id, _, _ in events]
    payloads = _payloads(session, ids, key)
    latest = _latest(session, [cdms[-1][0] for cdms in payloads.values()])
    jobs = _active_jobs(session, ids)
    return PageRows(
        tuple(
            _EventRows(
                event_id=event_id,
                object1=object1,
                object2=object2,
                payloads=[payload for _, payload in payloads[event_id]],
                latest=latest[payloads[event_id][-1][0]],
                active_job=jobs.get(event_id),
            )
            for event_id, object1, object2 in events
        ),
        total or 0,
        offset,
        limit,
    )


def _payloads(
    session: Session, event_ids: Sequence[str], key: str
) -> dict[str, list[tuple[str, Mapping[str, object] | None]]]:
    """The CDMs of each event, oldest first, by ``MESSAGE_ID``, with the
    payload cached on each under ``key`` (``None`` where there is none)."""
    cached = and_(
        models.Result.message_id == models.Cdm.message_id,
        models.Result.key == key,
    )
    rows: Sequence[tuple[str, str, Mapping[str, object] | None]] = session.execute(
        select(models.Cdm.event_id, models.Cdm.message_id, models.Result.payload)
        .outerjoin(models.Result, cached)
        .where(models.Cdm.event_id.in_(event_ids))
        .order_by(models.Cdm.event_id, models.Cdm.creation_date, models.Cdm.message_id)
    ).all()
    by_event: dict[str, list[tuple[str, Mapping[str, object] | None]]] = {}
    for event_id, message_id, payload in rows:
        by_event.setdefault(event_id, []).append((message_id, payload))
    return by_event


def _latest(session: Session, message_ids: Sequence[str]) -> dict[str, _Latest]:
    """The TCA, creation and text of each CDM of ``message_ids``."""
    rows = session.execute(
        select(
            models.Cdm.message_id,
            models.Cdm.tca,
            models.Cdm.creation_date,
            models.Cdm.content,
        ).where(models.Cdm.message_id.in_(message_ids))
    ).all()
    return {
        message_id: _Latest(tca, creation_date, content)
        for message_id, tca, creation_date, content in rows
    }


def active_refresh_in(session: Session, event_id: str, key: str) -> tuple[int, ...]:
    """The ids of an event's queued or running ``refresh`` jobs whose result
    is cached under ``key``, oldest first, in the caller's transaction."""
    return tuple(
        session.scalars(
            select(models.Job.id)
            .where(
                models.Job.event_id == event_id,
                models.Job.kind == JobKind.REFRESH,
                models.Job.key == key,
                models.Job.status.in_(_ACTIVE),
            )
            .order_by(models.Job.id)
        )
    )


def _active_jobs(session: Session, event_ids: Sequence[str]) -> dict[str, JobKind]:
    """The kind of the oldest queued or running job of each event that has
    one."""
    rows = session.execute(
        select(models.Job.event_id, models.Job.kind)
        .where(models.Job.event_id.in_(event_ids), models.Job.status.in_(_ACTIVE))
        .order_by(models.Job.id.desc())
    ).all()
    # Newest first: the oldest is written last.
    return dict(rows)
