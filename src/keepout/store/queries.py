"""The store's reads and writes in a caller's session.

Internal to :mod:`keepout.store`: :class:`~keepout.store.Store` and
:class:`~keepout.store.JobQueue` put several of these in one transaction.
None of them begins, commits or validates more than it says: a function
that takes an event's id reads nothing of an event that is not stored,
unless it says it raises.

Predicates (:func:`result_under`, :func:`cached`, :func:`cdms_of`,
:func:`active`) are pure: they build a condition and run no query.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Final, TypeVarTuple, cast

from astropy.time import Time
from sqlalchemy import and_, exists, func, or_, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import aliased

from keepout.errors import StoreError
from keepout.sources.catalog import CatalogFileRef, CatalogReference
from keepout.store import models
from keepout.store.codec import maneuver_key
from keepout.store.computing import parsed_event
from keepout.store.models import ACTIVE_STATUSES, JobKind, MessageId
from keepout.store.summaries import EventRows, Job, LatestCdm, PlanJobs

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence
    from pathlib import Path

    from sqlalchemy import ColumnElement, Select
    from sqlalchemy.orm import Session

    from keepout._types import JsonValue
    from keepout.events import ConjunctionEvent
    from keepout.settings import Settings
    from keepout.store.models import EventId

__all__ = [
    "NEWEST_FIRST",
    "active",
    "active_job_in",
    "active_refresh_in",
    "cached",
    "cached_payload_in",
    "catalog_in",
    "cdms_of",
    "insert_results_in",
    "is_cached_in",
    "latest_in",
    "maneuver_slot_in",
    "message_ids_in",
    "no_event",
    "oldest_first",
    "payloads_in",
    "pinned_in",
    "plan_jobs_in",
    "read_page",
    "require_events",
    "require_latest_in",
    "result_under",
    "save_maneuver_in",
    "texts_in",
]

_Ts = TypeVarTuple("_Ts")

NEWEST_FIRST: Final = (models.Job.created_at.desc(), models.Job.id.desc())
"""The order of jobs listed newest first."""


def no_event(event_id: str) -> str:
    """The message of an event that is not stored."""
    return f"no event {event_id} in the store"


# --- The cache ---------------------------------------------------------------


def result_under(key: str) -> ColumnElement[bool]:
    """A CDM's result cached under ``key``: the condition that joins
    :class:`~keepout.store.models.Result` to :class:`~keepout.store.models.Cdm`."""
    return and_(
        models.Result.message_id == models.Cdm.message_id,
        models.Result.key == key,
    )


def cached(key: str) -> ColumnElement[bool]:
    """The CDMs whose result is cached under ``key``."""
    return exists().where(result_under(key))


def is_cached_in(session: Session, message_id: str, key: str) -> bool:
    """Whether a result is cached on ``message_id`` under ``key``."""
    found = exists().where(models.Cdm.message_id == message_id, cached(key))
    return bool(session.scalar(select(found)))


def cached_payload_in(
    session: Session, message_id: str, key: str
) -> dict[str, JsonValue] | None:
    """The result cached on ``message_id`` under ``key``; ``None`` if none is."""
    return session.scalar(
        select(models.Result.payload).where(
            models.Result.message_id == message_id, models.Result.key == key
        )
    )


def insert_results_in(
    session: Session,
    key: str,
    payloads: Sequence[tuple[str, Mapping[str, JsonValue]]],
) -> int:
    """Write results under ``key``, each a CDM's ``MESSAGE_ID`` and its
    payload, but those of CDMs not stored (e.g. purged meanwhile) and those
    already cached; how many it wrote."""
    given = {message_id for message_id, _ in payloads}
    stored = set(
        session.scalars(
            select(models.Cdm.message_id).where(models.Cdm.message_id.in_(given))
        )
    )
    created_at = Time.now()
    rows: list[dict[str, object]] = [
        {
            "message_id": message_id,
            "key": key,
            "payload": dict(payload),
            "created_at": created_at,
        }
        for message_id, payload in payloads
        if message_id in stored
    ]
    if not rows:
        return 0
    insert = sqlite_insert(models.Result).on_conflict_do_nothing(
        index_elements=[models.Result.message_id, models.Result.key]
    )
    return session.connection().execute(insert, rows).rowcount


def save_maneuver_in(
    session: Session, message_id: str, key: str, payload: Mapping[str, JsonValue]
) -> bool:
    """Cache a manoeuvre outcome's JSON on a CDM under its
    :func:`~keepout.store.codec.maneuver_key`; whether it was written (not if
    the CDM is not stored, or an outcome is cached there under ``key``)."""
    return insert_results_in(session, key, [(message_id, payload)]) == 1


# --- Events and their CDMs ---------------------------------------------------


def require_events(session: Session, event_ids: Collection[str]) -> None:
    """Check that the events are stored.

    Raises
    ------
    StoreError
        If one is not, naming the first by id.
    """
    known = set(
        session.scalars(select(models.Event.id).where(models.Event.id.in_(event_ids)))
    )
    unknown = sorted(set(event_ids) - known)
    if unknown:
        raise StoreError(no_event(unknown[0]))


def cdms_of(event_id: str, *, until: str | None = None) -> ColumnElement[bool]:
    """The CDMs of an event, up to ``until`` (its ``MESSAGE_ID``) in
    (``CREATION_DATE``, ``MESSAGE_ID``) order if given; none if ``until`` is
    not stored."""
    of_event = models.Cdm.event_id == event_id
    if until is None:
        return of_event
    pinned = aliased(models.Cdm)
    created = (
        select(pinned.creation_date).where(pinned.message_id == until).scalar_subquery()
    )
    return and_(
        of_event,
        or_(
            models.Cdm.creation_date < created,
            and_(models.Cdm.creation_date == created, models.Cdm.message_id <= until),
        ),
    )


def oldest_first(
    query: Select[*_Ts], cdms: ColumnElement[bool], *, lacking: str | None = None
) -> Select[*_Ts]:
    """``query`` of ``cdms``, by ``CREATION_DATE`` then ``MESSAGE_ID``; only
    those without a result under ``lacking`` if given."""
    if lacking is not None:
        query = query.where(~cached(lacking))
    return query.where(cdms).order_by(models.Cdm.creation_date, models.Cdm.message_id)


def message_ids_in(
    session: Session,
    event_id: str,
    *,
    until: str | None = None,
    lacking: str | None = None,
) -> tuple[MessageId, ...]:
    """The ``MESSAGE_ID`` of an event's CDMs, oldest first: up to ``until``
    and without a result under ``lacking`` if given (:func:`oldest_first`)."""
    cdms = cdms_of(event_id, until=until)
    query = oldest_first(select(models.Cdm.message_id), cdms, lacking=lacking)
    return tuple(session.scalars(query))


def texts_in(session: Session, cdms: ColumnElement[bool]) -> Sequence[str]:
    """The stored texts of ``cdms``."""
    return session.scalars(select(models.Cdm.content).where(cdms)).all()


def latest_in(session: Session, event_id: str) -> MessageId | None:
    """The ``MESSAGE_ID`` of an event's latest CDM, by ``CREATION_DATE``
    then ``MESSAGE_ID``; ``None`` if the event is not stored (every stored
    event has a CDM)."""
    return session.scalar(
        select(models.Cdm.message_id)
        .where(cdms_of(event_id))
        .order_by(models.Cdm.creation_date.desc(), models.Cdm.message_id.desc())
        .limit(1)
    )


def require_latest_in(session: Session, event_id: str) -> MessageId:
    """The ``MESSAGE_ID`` of an event's latest CDM (:func:`latest_in`).

    Raises
    ------
    StoreError
        If the event is not stored.
    """
    latest = latest_in(session, event_id)
    if latest is None:
        raise StoreError(no_event(event_id))
    return latest


def catalog_in(
    session: Session, base: Path, message_id: str
) -> CatalogReference | None:
    """The catalog a stored CDM was made from, its paths under ``base`` (the
    store's directory); ``None`` if it was made from none."""
    # The outer join's file is None for a catalog of no file, which the
    # column's type does not say.
    rows = cast(
        "Sequence[tuple[str, models.CatalogFile | None]]",
        session.execute(
            select(models.Catalog.label, models.CatalogFile)
            .select_from(models.Cdm)
            .join(models.Catalog, models.Cdm.catalog_id == models.Catalog.id)
            .outerjoin(
                models.CatalogFile, models.CatalogFile.catalog_id == models.Catalog.id
            )
            .where(models.Cdm.message_id == message_id)
            .order_by(models.CatalogFile.id)
        ).all(),
    )
    if not rows:
        return None
    return CatalogReference(
        rows[0][0],
        tuple(
            CatalogFileRef(
                file.name,
                base / file.path,
                file.sha256,
                url=file.url,
                retrieved=file.retrieved,
            )
            for _, file in rows
            if file is not None
        ),
    )


def maneuver_slot_in(
    session: Session, base: Path, event_id: str, settings: Settings
) -> tuple[MessageId, str] | None:
    """Where a plan of a stored event with ``settings`` is cached: the
    event's latest CDM and the plan's key; ``None`` if the event is not
    stored.

    Raises
    ------
    ValueError
        If a settings file cannot hold ``settings``, as
        :func:`~keepout.store.codec.maneuver_key`.
    """
    latest = latest_in(session, event_id)
    if latest is None:
        return None
    return latest, maneuver_key(settings, catalog_in(session, base, latest))


def pinned_in(
    session: Session, base: Path, event_id: str, message_id: str | None
) -> tuple[ConjunctionEvent, CatalogReference | None]:
    """A stored event as it was at one of its CDMs (``None``: its latest),
    and that CDM's catalog.

    Raises
    ------
    StoreError
        If there is no such event, or ``message_id`` is not a CDM of it.
    CdmParseError
        If a stored CDM no longer parses (a store edited by hand).
    EventError
        If the stored CDMs no longer make an event (likewise).
    """
    if message_id is None:
        pinned = require_latest_in(session, event_id)
    else:
        found = session.scalar(
            select(models.Cdm.event_id).where(models.Cdm.message_id == message_id)
        )
        if found != event_id:
            require_events(session, [event_id])
            raise StoreError(f"no CDM {message_id} in event {event_id}")
        pinned = MessageId(message_id)
    texts = texts_in(session, cdms_of(event_id, until=message_id))
    return parsed_event(event_id, texts), catalog_in(session, base, pinned)


def payloads_in(
    session: Session, event_ids: Collection[str], key: str
) -> dict[EventId, list[tuple[MessageId, Mapping[str, object] | None]]]:
    """The CDMs of each stored event of ``event_ids``, oldest first, by
    ``MESSAGE_ID``, with the payload cached on each under ``key``: ``None``
    where there is none (the outer join's ``NULL``, which the column's type
    does not say)."""
    rows: Sequence[tuple[EventId, MessageId, Mapping[str, object] | None]] = (
        session.execute(
            select(models.Cdm.event_id, models.Cdm.message_id, models.Result.payload)
            .outerjoin(models.Result, result_under(key))
            .where(models.Cdm.event_id.in_(event_ids))
            .order_by(
                models.Cdm.event_id, models.Cdm.creation_date, models.Cdm.message_id
            )
        ).all()
    )
    by_event: defaultdict[
        EventId, list[tuple[MessageId, Mapping[str, object] | None]]
    ] = defaultdict(list)
    for event_id, message_id, payload in rows:
        by_event[event_id].append((message_id, payload))
    return dict(by_event)


def read_page(
    session: Session,
    key: str,
    *,
    as_of: Time | None,
    offset: int,
    limit: int,
) -> tuple[tuple[EventRows, ...], int]:
    """A page of the events list (:meth:`~keepout.store.Store.summaries`):
    the rows of its events, with their Pc cached under ``key``, and how
    many events the list has; only the events whose latest TCA is not before
    ``as_of`` if given."""
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
    payloads = payloads_in(session, ids, key)
    latest = _latest_cdms(session, [cdms[-1][0] for cdms in payloads.values()])
    jobs = _active_kinds(session, ids)
    rows = tuple(
        EventRows(
            event_id=event_id,
            object1=object1,
            object2=object2,
            payloads=[payload for _, payload in payloads[event_id]],
            latest=latest[payloads[event_id][-1][0]],
            active_job=jobs.get(event_id),
        )
        for event_id, object1, object2 in events
    )
    return rows, total or 0


def _latest_cdms(
    session: Session, message_ids: Sequence[MessageId]
) -> dict[MessageId, LatestCdm]:
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
        message_id: LatestCdm(tca, creation_date, content)
        for message_id, tca, creation_date, content in rows
    }


# --- Jobs --------------------------------------------------------------------


def active(
    kind: JobKind | None = None,
    *,
    event_ids: Collection[str],
    key: str | None = None,
) -> ColumnElement[bool]:
    """The queued or running jobs of ``event_ids``; only those of ``kind``
    and ``key`` if given."""
    condition = and_(
        models.Job.event_id.in_(event_ids), models.Job.status.in_(ACTIVE_STATUSES)
    )
    if kind is not None:
        condition = and_(condition, models.Job.kind == kind)
    if key is not None:
        condition = and_(condition, models.Job.key == key)
    return condition


def active_refresh_in(session: Session, event_id: str, key: str) -> tuple[int, ...]:
    """The ids of an event's queued or running ``refresh`` jobs whose result
    is cached under ``key``, oldest first."""
    which = active(JobKind.REFRESH, event_ids=[event_id], key=key)
    return tuple(
        session.scalars(select(models.Job.id).where(which).order_by(models.Job.id))
    )


def _active_kinds(
    session: Session, event_ids: Sequence[EventId]
) -> dict[EventId, JobKind]:
    """The kind of the oldest queued or running job of each event that has
    one."""
    rows = session.execute(
        select(models.Job.event_id, models.Job.kind)
        .where(active(event_ids=event_ids))
        .order_by(models.Job.id.desc())
    ).all()
    # Newest first: the oldest is written last.
    return dict(rows)


def _jobs_of(kind: JobKind, message_id: str, key: str) -> Select[models.Job]:
    """The jobs of ``kind``, ``message_id`` and ``key``."""
    return select(models.Job).where(
        models.Job.kind == kind,
        models.Job.message_id == message_id,
        models.Job.key == key,
    )


def active_job_in(
    session: Session, kind: JobKind, message_id: str, key: str
) -> Job | None:
    """The oldest queued or running job of ``kind``, ``message_id`` and
    ``key``."""
    row = session.scalars(
        _jobs_of(kind, message_id, key)
        .where(models.Job.status.in_(ACTIVE_STATUSES))
        .order_by(models.Job.id)
        .limit(1)
    ).one_or_none()
    return None if row is None else Job.of(row)


def plan_jobs_in(
    session: Session, event_id: str, message_id: str, key: str
) -> PlanJobs:
    """An event's manoeuvre jobs: those still queued or running, and the one
    of ``message_id`` (its latest CDM) and ``key`` (the plan's
    :func:`~keepout.store.codec.maneuver_key`), active or else newest."""
    maneuver = JobKind.MANEUVER
    current = active_job_in(session, maneuver, message_id, key)
    if current is None:
        newest = session.scalars(
            _jobs_of(maneuver, message_id, key).order_by(*NEWEST_FIRST).limit(1)
        ).one_or_none()
        current = None if newest is None else Job.of(newest)
    rows = session.scalars(
        select(models.Job)
        .where(active(maneuver, event_ids=[event_id]))
        .order_by(*NEWEST_FIRST)
    )
    return PlanJobs(tuple(Job.of(row) for row in rows), current)
