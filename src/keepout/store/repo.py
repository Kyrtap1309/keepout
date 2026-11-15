"""The conjunction store: a SQLite database of events, CDMs and results.

CDMs are only added (:meth:`Store.ingest`): a message is stored once, by its
``MESSAGE_ID``, and given again it is reported as already stored. Each CDM
joins the stored event of its encounter, by the rule of
:func:`keepout.events.group_events` (:class:`~keepout.events.EventBounds`):
same objects (``OBJECT1``, ``OBJECT2`` by designator), TCAs within
:data:`~keepout.events.DEFAULT_TCA_TOLERANCE`, no other ``CONJUNCTION_ID``.
An event keeps the id its first CDM gave it
(:func:`~keepout.events.stable_event_id`), however many CDMs follow.

Each CDM's Pc observation is cached (:meth:`Store.refresh`) under the key of
the settings it was computed with
(:func:`~keepout.store.codec.result_key`); an event's manoeuvre outcome
(:meth:`Store.maneuver`) is cached on its latest CDM under the key of the
operator's settings and the catalog
(:func:`~keepout.store.codec.maneuver_key`).
"""

from __future__ import annotations

import functools
import hashlib
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Self, TypeVarTuple

import astropy.units as u
from astropy.time import Time
from sqlalchemy import and_, exists, func, or_, select
from sqlalchemy import delete as delete_rows
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import DatabaseError, IntegrityError
from sqlalchemy.orm import Session, selectinload

import keepout.cdm
from keepout.errors import HardBodyRadiusError, StoreError
from keepout.events import (
    DEFAULT_TCA_TOLERANCE,
    AssessedEvent,
    ConjunctionEvent,
    EventBounds,
    missing_hard_body_radius,
    stable_event_id,
    without_hard_body_radius,
)
from keepout.risk import assess
from keepout.sources.catalog import CatalogFileRef, CatalogReference
from keepout.store import models
from keepout.store.codec import (
    maneuver_key,
    observation_to_json,
    result_key,
)
from keepout.store.models import EventId, MessageId
from keepout.store.summaries import (
    EventDetail,
    SummaryPage,
    active_refresh_in,
    check_page,
    decoded,
    payloads_in,
    read_page,
    result_under,
    summarize,
)
from keepout.units import elapsed_seconds

if TYPE_CHECKING:
    from collections.abc import (
        Collection,
        Generator,
        Iterable,
        Iterator,
        Mapping,
        Sequence,
    )
    from types import TracebackType

    from sqlalchemy import ColumnElement, Engine, Select

    from keepout._types import JsonValue
    from keepout.cdm import Cdm, CdmFormat
    from keepout.maneuver import ManeuverOutcome
    from keepout.risk import PcObservation
    from keepout.settings import Settings
    from keepout.store.jobs import PlanDetail

__all__ = ["WRITE", "CdmInput", "IngestReport", "RefreshReport", "Store"]

WRITE: Final = "keepout_write"
"""Execution option of a write transaction: it starts with ``BEGIN IMMEDIATE``
(see :func:`keepout.store.db.engine_for`)."""

_COPY_NAME: Final = re.compile(r"([0-9a-f]{64})(?:\.|$)")
"""The name of a catalog copy: its SHA-256, then the original's suffixes."""

_BATCH_SIZE: Final = 100
"""Most observations :meth:`Store.refresh` writes in one transaction, and
most CDM texts it reads in one."""

_BATCH_SECONDS: Final = 1.0
"""Longest :meth:`Store.refresh` computes before writing what it computed."""

_GIVE_HARD_BODY_RADIUS: Final = "give the combined hard-body radius in the settings"
"""The hint of a Pc that lacks a hard-body radius."""

_WINDOW: Final = (DEFAULT_TCA_TOLERANCE + 1.0) * u.s
"""How far from a TCA the database searches for the events it may join: the
tolerance and a second of slack; the exact test follows."""

_PARSED_CDMS: Final = 1024
"""How many stored CDMs the process keeps parsed (:func:`_parsed_cdm`): the
CDMs of some 50 events of 20 CDMs, about 25 MB with their texts."""


_Ts = TypeVarTuple("_Ts")


def _now() -> Time:
    """The current time, for the rows' ``created_at`` and ``ingested_at``."""
    return Time.now()


def _digest(cdm: Cdm) -> str:
    """What makes two messages the same: the SHA-256 (hex) of the KVN."""
    return hashlib.sha256(keepout.cdm.kvn.dumps(cdm).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CdmInput:
    """A CDM to ingest, with its text as read and where it came from.

    Parameters
    ----------
    cdm
        The message.
    text
        Its text, KVN or XML: what the store keeps.
    format
        The text's format.
    origin
        Where it came from: the file it was read from, or ``"screen"``.
    """

    cdm: Cdm
    text: str
    format: CdmFormat
    origin: str

    @classmethod
    def read(cls, path: Path) -> CdmInput:
        """The CDM of a KVN or XML file (UTF-8, gzip-compressed or not),
        its origin the path.

        Parameters
        ----------
        path
            The CDM file, KVN or XML; its path is the origin.

        Returns
        -------
        CdmInput
            The parsed CDM, its text, format and origin.

        Raises
        ------
        CdmParseError
            If the file is broken gzip data, not UTF-8 text or not a CDM.
        OSError
            If it cannot be read.
        """
        return cls.parse(keepout.cdm.read_text(path), str(path))

    @classmethod
    def parse(cls, text: str, origin: str) -> CdmInput:
        """The CDM of a text, KVN or XML.

        Parameters
        ----------
        text
            The message's text, as read (decoded, without a byte-order mark).
        origin
            Where it came from, e.g. the file it was read from.

        Returns
        -------
        CdmInput
            The parsed CDM, its text, format and origin.

        Raises
        ------
        CdmParseError
            If the text is not a CDM.
        """
        return cls(keepout.cdm.loads(text), text, keepout.cdm.format_of(text), origin)

    @classmethod
    def of(cls, cdm: Cdm, origin: str) -> CdmInput:
        """A CDM made in memory, e.g. by a screening, kept as its KVN.

        Parameters
        ----------
        cdm
            The message, e.g. from :func:`~keepout.cdm.make_cdm`.
        origin
            Where it came from, e.g. ``"screen"``.

        Returns
        -------
        CdmInput
            The CDM, its KVN text and its origin.
        """
        return cls(cdm, keepout.cdm.kvn.dumps(cdm), "kvn", origin)


@dataclass(frozen=True, slots=True)
class IngestReport:
    """What an ingest did, in the order the messages were given.

    Parameters
    ----------
    added
        The ``MESSAGE_ID`` of each message stored.
    already_stored
        The ``MESSAGE_ID`` of each message given that was stored before
        (or earlier in the same ingest), with the same content.
    events
        The id of each event the messages given belong to.
    new_events
        The id of each of those events that the ingest created.
    """

    added: tuple[MessageId, ...]
    already_stored: tuple[MessageId, ...]
    events: tuple[EventId, ...]
    new_events: tuple[EventId, ...]


@dataclass(frozen=True, slots=True)
class RefreshReport:
    """What a refresh did.

    Parameters
    ----------
    computed
        How many Pc observations it computed and cached.
    skipped
        The ``MESSAGE_ID`` of each CDM left without a Pc because it lacks
        ``HBR`` and the settings give no hard-body radius (only when every
        event is refreshed), oldest event first.
    """

    computed: int
    skipped: tuple[MessageId, ...] = ()


def _unique(items: Sequence[EventId]) -> tuple[EventId, ...]:
    """``items`` without repeats, in the order they first come."""
    return tuple(dict.fromkeys(items))


def _bounds(event: models.Event) -> EventBounds:
    """The :class:`~keepout.events.EventBounds` of a stored event."""
    return EventBounds(
        (event.object1, event.object2),
        event.tca_min,
        event.tca_max,
        event.conjunction_id,
    )


def _new_event_id(session: Session, message_id: str) -> EventId:
    """The id of a new event whose first CDM has this ``MESSAGE_ID``: its
    :func:`~keepout.events.stable_event_id`, or, if another event has that,
    the stable id of ``"<MESSAGE_ID>#<n>"`` for the first free ``n``."""
    candidate = EventId(stable_event_id(message_id))
    attempt = 0
    while session.get(models.Event, candidate) is not None:
        attempt += 1
        candidate = EventId(stable_event_id(f"{message_id}#{attempt}"))
    return candidate


def _event_of(session: Session, cdm: Cdm) -> tuple[models.Event, bool]:
    """The event that ``cdm`` joins, by the rule of
    :class:`~keepout.events.EventBounds`, updated for it, and whether it is
    new.

    Raises
    ------
    StoreError
        If it fits more than one stored event.
    """
    # The candidates: the same objects, a TCA near enough.
    query = select(models.Event).where(
        models.Event.object1 == cdm.object1.object_id,
        models.Event.object2 == cdm.object2.object_id,
        models.Event.tca_max >= cdm.tca - _WINDOW,
        models.Event.tca_min <= cdm.tca + _WINDOW,
    )
    fitting = [event for event in session.scalars(query) if _bounds(event).admits(cdm)]
    if len(fitting) > 1:
        ids = ", ".join(sorted(event.id for event in fitting))
        raise StoreError(f"CDM {cdm.message_id} fits more than one event: {ids}")
    if fitting:
        (event,) = fitting
        bounds = _bounds(event).including(cdm)
        event.tca_min, event.tca_max = bounds.first_tca, bounds.last_tca
        event.conjunction_id = bounds.conjunction_id
        return event, False
    event = models.Event(
        id=_new_event_id(session, cdm.message_id),
        object1=cdm.object1.object_id,
        object2=cdm.object2.object_id,
        conjunction_id=cdm.conjunction_id,
        tca_min=cdm.tca,
        tca_max=cdm.tca,
        created_at=_now(),
    )
    session.add(event)
    return event, True


def _catalog_row(reference: CatalogReference, base: Path) -> models.Catalog:
    """The row of a catalog reference, its files' paths relative to ``base``
    (the store's directory)."""
    return models.Catalog(
        label=reference.label,
        files=[
            models.CatalogFile(
                name=file.name,
                path=file.path.relative_to(base).as_posix(),
                url=file.url,
                retrieved=file.retrieved,
                sha256=file.sha256,
            )
            for file in reference.files
        ],
    )


def insert_results_in(
    session: Session,
    key: str,
    payloads: Sequence[tuple[str, Mapping[str, JsonValue]]],
) -> int:
    """Write results under ``key`` in the caller's write transaction, each a
    CDM's ``MESSAGE_ID`` and its payload, but those of CDMs not stored (e.g.
    purged meanwhile) and those already cached.

    Internal to :mod:`keepout.store`: a refresh job writes its observations
    and its own end in one transaction.

    Parameters
    ----------
    session
        A session of :meth:`Store.write`.
    key
        The key of the results.
    payloads
        Each result: its CDM's ``MESSAGE_ID`` and its JSON.

    Returns
    -------
    int
        How many it wrote.
    """
    given = {message_id for message_id, _ in payloads}
    stored = set(
        session.scalars(
            select(models.Cdm.message_id).where(models.Cdm.message_id.in_(given))
        )
    )
    rows: list[dict[str, object]] = [
        {
            "message_id": message_id,
            "key": key,
            "payload": dict(payload),
            "created_at": _now(),
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


def _write_results(
    session: Session, key: str, observations: Sequence[PcObservation]
) -> int:
    """Write observations under ``key``, but those of CDMs not stored (e.g.
    purged meanwhile) and those already cached; how many it wrote."""
    return insert_results_in(
        session,
        key,
        [
            (observation.message_id, observation_to_json(observation))
            for observation in observations
            if observation.message_id is not None
        ],
    )


def save_maneuver_in(
    session: Session, message_id: str, key: str, payload: Mapping[str, JsonValue]
) -> bool:
    """Cache a manoeuvre outcome's JSON on a CDM, in the caller's write
    transaction (:meth:`Store.save_maneuver` makes one of its own).

    Internal to :mod:`keepout.store`: a job that plans a manoeuvre writes
    its result and its own end in one transaction.

    Parameters
    ----------
    session
        A session of :meth:`Store.write`.
    message_id
        The ``MESSAGE_ID`` of the CDM the outcome was planned on.
    key
        The outcome's :func:`~keepout.store.codec.maneuver_key`: the settings and
        catalog it was planned with.
    payload
        The outcome's JSON, from :func:`~keepout.store.maneuver_codec.outcome_to_json`.

    Returns
    -------
    bool
        Whether it was written: not if the CDM is not stored (e.g. purged
        meanwhile) or an outcome is already cached on it under ``key``.
    """
    return insert_results_in(session, key, [(message_id, payload)]) == 1


def _maneuver_key_in(
    session: Session, base: Path, message_id: str, settings: Settings
) -> str:
    """The key of a plan with ``settings`` on a stored CDM: its
    :func:`~keepout.store.codec.maneuver_key` with the CDM's catalog.

    Raises
    ------
    ValueError
        If a settings file cannot hold ``settings``.
    """
    return maneuver_key(settings, catalog_in(session, base, message_id))


def maneuver_slot_in(
    session: Session, base: Path, event_id: str, settings: Settings
) -> tuple[MessageId, str]:
    """Where a plan of a stored event with ``settings`` is cached, in the
    caller's transaction: the event's latest CDM and the plan's key.

    Internal to :mod:`keepout.store`: the cache and the job queue look up
    the same slot.

    Parameters
    ----------
    session
        A session of the store.
    base
        The directory of the store file.
    event_id
        The store's id of the event.
    settings
        The operator's settings of the plan.

    Raises
    ------
    StoreError
        If there is no such event.
    ValueError
        If a settings file cannot hold ``settings``, as
        :func:`~keepout.store.codec.maneuver_key`.
    """
    message_id = latest_in(session, event_id)
    return message_id, _maneuver_key_in(session, base, message_id, settings)


def pinned_in(
    session: Session, base: Path, event_id: str, message_id: str | None
) -> tuple[ConjunctionEvent, CatalogReference | None]:
    """A stored event as it was at one of its CDMs, and that CDM's catalog,
    in the caller's transaction.

    Internal to :mod:`keepout.store`: a plan reads both in one transaction.

    Parameters
    ----------
    session
        A session of the store.
    base
        The directory of the store file.
    event_id
        The store's id of the event.
    message_id
        The ``MESSAGE_ID`` of one of its CDMs: the event without the CDMs
        after it in (``CREATION_DATE``, ``MESSAGE_ID``) order. ``None``: its
        latest CDM.

    Raises
    ------
    StoreError
        If there is no such event, or ``message_id`` is not a CDM of it.
    CdmParseError
        If a stored CDM no longer parses (a store edited by hand).
    EventError
        If the stored CDMs no longer make an event (likewise).
    """
    cdms = _cdms_of(session, event_id, message_id)
    latest = _latest_message_id(session, cdms)
    event = _parsed_event(event_id, _texts(session, cdms))
    return event, catalog_in(session, base, latest)


def _paired(
    rows: Sequence[tuple[str, Mapping[str, object] | None]],
) -> tuple[tuple[MessageId, PcObservation | None], ...]:
    """Each CDM's ``MESSAGE_ID`` with its observation decoded from its
    cached payload (:func:`~keepout.store.summaries.decoded`).

    Raises
    ------
    StoreError
        If a payload is not an observation.
    """
    observations = decoded(payload for _, payload in rows)
    return tuple(
        (MessageId(message_id), observation)
        for (message_id, _), observation in zip(rows, observations, strict=True)
    )


def _cached(key: str) -> ColumnElement[bool]:
    """The CDMs whose result is cached under ``key``."""
    return exists().where(result_under(key))


def _oldest_first(
    query: Select[*_Ts],
    cdms: ColumnElement[bool],
    *,
    lacking: str | None = None,
) -> Select[*_Ts]:
    """``query`` of ``cdms``, by ``CREATION_DATE`` then ``MESSAGE_ID``; only
    those without a result under ``lacking`` if given."""
    if lacking is not None:
        query = query.where(~_cached(lacking))
    return query.where(cdms).order_by(models.Cdm.creation_date, models.Cdm.message_id)


def _lacking_hard_body_radius(settings: Settings, cdms: Iterable[Cdm]) -> list[Cdm]:
    """The CDMs of ``cdms`` without ``HBR`` for an object, if the settings
    give no hard-body radius (``cdms`` is not read if they do)."""
    if settings.combined_hard_body_radius is not None:
        return []
    return [cdm for cdm in cdms if without_hard_body_radius((cdm,))]


def _refuse_lacking_hard_body_radius(lacking: Sequence[Cdm]) -> None:
    """Refuse the Pc of CDMs that lack ``HBR`` (:func:`_lacking_hard_body_radius`).

    Raises
    ------
    HardBodyRadiusError
        If there is one.
    """
    if lacking:
        raise HardBodyRadiusError(
            missing_hard_body_radius(lacking), _GIVE_HARD_BODY_RADIUS
        )


def _observation(event_id: str, cdm: Cdm, settings: Settings) -> PcObservation:
    """The Pc observation of a CDM of an event, computed with ``settings``."""
    (observation,) = ConjunctionEvent(event_id, (cdm,)).observations(
        compute=settings.compute,
        policy=settings.policy,
        combined_hard_body_radius=settings.combined_hard_body_radius,
    )
    return observation


def _require_events(session: Session, event_ids: Collection[str]) -> None:
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
        raise StoreError(f"no event {unknown[0]} in the store")


def _cdms_of(session: Session, event_id: str, until: str | None) -> ColumnElement[bool]:
    """The CDMs of a stored event, up to ``until`` (its ``MESSAGE_ID``) in
    (``CREATION_DATE``, ``MESSAGE_ID``) order if given.

    Raises
    ------
    StoreError
        If there is no such event, or ``until`` is not a CDM of it.
    """
    _require_events(session, [event_id])
    of_event = models.Cdm.event_id == event_id
    if until is None:
        return of_event
    pinned = session.execute(
        select(models.Cdm.creation_date, models.Cdm.message_id).where(
            of_event, models.Cdm.message_id == until
        )
    ).one_or_none()
    if pinned is None:
        raise StoreError(f"no CDM {until} in event {event_id}")
    created, message_id = pinned
    return and_(
        of_event,
        or_(
            models.Cdm.creation_date < created,
            and_(
                models.Cdm.creation_date == created,
                models.Cdm.message_id <= message_id,
            ),
        ),
    )


def _latest_message_id(session: Session, cdms: ColumnElement[bool]) -> MessageId:
    """The ``MESSAGE_ID`` of the latest of ``cdms``, by ``CREATION_DATE``
    (then by ``MESSAGE_ID``); every stored event has a CDM."""
    return session.execute(
        select(models.Cdm.message_id)
        .where(cdms)
        .order_by(models.Cdm.creation_date.desc(), models.Cdm.message_id.desc())
        .limit(1)
    ).scalar_one()


def missing_in(
    session: Session, event_id: str, key: str, *, until: str | None = None
) -> tuple[MessageId, ...]:
    """The CDMs of a stored event without a result under ``key``, oldest
    first, in the caller's transaction.

    Internal to :mod:`keepout.store`: the store and the job queue look for
    the same missing Pc.

    Parameters
    ----------
    session
        A session of the store.
    event_id
        The store's id of the event.
    key
        The key of the results, e.g. a
        :func:`~keepout.store.codec.result_key`.
    until
        Only the CDMs up to this one (its ``MESSAGE_ID``), in
        (``CREATION_DATE``, ``MESSAGE_ID``) order; ``None``: all of them.

    Returns
    -------
    tuple of MessageId
        Their ``MESSAGE_ID``, by ``CREATION_DATE`` then ``MESSAGE_ID``.

    Raises
    ------
    StoreError
        If there is no such event, or ``until`` is not a CDM of it.
    """
    cdms = _cdms_of(session, event_id, until)
    query = _oldest_first(select(models.Cdm.message_id), cdms, lacking=key)
    return tuple(session.scalars(query))


def message_ids_in(
    session: Session, event_id: str, *, until: str | None = None
) -> tuple[MessageId, ...]:
    """The CDMs of a stored event, oldest first, in the caller's transaction.

    Internal to :mod:`keepout.store`: the job queue checks a refresh's
    result against them.

    Parameters
    ----------
    session
        A session of the store.
    event_id
        The store's id of the event.
    until
        Only the CDMs up to this one (its ``MESSAGE_ID``), as
        :func:`missing_in`; ``None``: all of them.

    Returns
    -------
    tuple of MessageId
        Their ``MESSAGE_ID``, by ``CREATION_DATE`` then ``MESSAGE_ID``.

    Raises
    ------
    StoreError
        If there is no such event, or ``until`` is not a CDM of it.
    """
    cdms = _cdms_of(session, event_id, until)
    return tuple(session.scalars(_oldest_first(select(models.Cdm.message_id), cdms)))


def compute_missing(
    store: Store, event_id: str, settings: Settings, *, key: str, until: str
) -> dict[MessageId, PcObservation]:
    """The Pc observations a stored event lacks under ``key``, computed with
    ``settings`` but not written.

    Internal to :mod:`keepout.store`: the computation of a refresh job
    (:func:`~keepout.store.planning.run_refresh_job`), whose worker writes
    them. The CDMs are read in one read transaction, then parsed and
    computed outside it.

    Parameters
    ----------
    store
        The open store.
    event_id
        The store's id of the event.
    settings
        How Pc is computed.
    key
        The key the observations are cached under: the
        :func:`~keepout.store.codec.result_key` of ``settings``.
    until
        Only the CDMs up to this one (its ``MESSAGE_ID``), as
        :func:`missing_in`.

    Returns
    -------
    dict
        The observation of each CDM without one, by ``MESSAGE_ID``, oldest
        first.

    Raises
    ------
    StoreError
        If there is no such event, ``until`` is not a CDM of it, or the
        database cannot be used.
    HardBodyRadiusError
        If the settings give no hard-body radius and a CDM lacks ``HBR``
        for an object (as :meth:`Store.refresh` of given events; checked
        before any Pc is computed).
    CdmParseError
        If a stored CDM no longer parses (a store edited by hand).
    """
    with store.read() as session:
        cdms_of = _cdms_of(session, event_id, until)
        query = _oldest_first(select(models.Cdm.content), cdms_of, lacking=key)
        texts = session.scalars(query).all()
    cdms = [keepout.cdm.loads(text) for text in texts]
    _refuse_lacking_hard_body_radius(_lacking_hard_body_radius(settings, cdms))
    return {
        MessageId(cdm.message_id): _observation(event_id, cdm, settings) for cdm in cdms
    }


def latest_in(session: Session, event_id: str) -> MessageId:
    """The ``MESSAGE_ID`` of an event's latest CDM, by ``CREATION_DATE``
    (then by ``MESSAGE_ID``), in the caller's transaction.

    Parameters
    ----------
    session
        A session of the store, in the caller's transaction.
    event_id
        The store's id of the event.

    Raises
    ------
    StoreError
        If there is no such event.
    """
    return _latest_message_id(session, _cdms_of(session, event_id, None))


def catalog_in(
    session: Session, base: Path, message_id: str
) -> CatalogReference | None:
    """The catalog a stored CDM was made from (as :meth:`Store.catalog`), in
    the caller's transaction.

    Internal to :mod:`keepout.store`: a job's key is that of the catalog of
    its CDM, read in the transaction that queues the job.

    Parameters
    ----------
    session
        A session of the store.
    base
        The directory of the store file: the catalog's paths are relative
        to it.
    message_id
        The ``MESSAGE_ID`` of the CDM.
    """
    catalog_id = session.scalar(
        select(models.Cdm.catalog_id).where(models.Cdm.message_id == message_id)
    )
    if catalog_id is None:
        return None
    catalog = session.get_one(
        models.Catalog, catalog_id, options=[selectinload(models.Catalog.files)]
    )
    return CatalogReference(
        catalog.label,
        tuple(
            CatalogFileRef(
                file.name,
                base / file.path,
                file.sha256,
                url=file.url,
                retrieved=file.retrieved,
            )
            for file in catalog.files
        ),
    )


def _texts(session: Session, cdms: ColumnElement[bool]) -> Sequence[str]:
    """The stored texts of ``cdms``."""
    return session.scalars(select(models.Cdm.content).where(cdms)).all()


@functools.lru_cache(maxsize=_PARSED_CDMS)
def _parsed_cdm(text: str) -> Cdm:
    """A stored CDM's text, parsed; the last :data:`_PARSED_CDMS` texts
    parsed are kept, and shared.

    A stored CDM's text never changes (a ``MESSAGE_ID`` is stored once),
    and a :class:`~keepout.cdm.Cdm` cannot be changed (frozen, its
    covariance read-only): a page shown again reads its event's CDMs
    without parsing them again.

    Raises
    ------
    CdmParseError
        If the text does not parse (a store edited by hand); not kept.
    """
    return keepout.cdm.loads(text)


def _parsed_event(event_id: str, texts: Iterable[str]) -> ConjunctionEvent:
    """The event of stored CDM texts, each parsed once in the process
    (:func:`_parsed_cdm`).

    Raises
    ------
    CdmParseError
        If a text no longer parses (a store edited by hand).
    EventError
        If the CDMs no longer make an event (likewise).
    """
    return ConjunctionEvent(event_id, tuple(_parsed_cdm(text) for text in texts))


def _cached_payload(
    session: Session, message_id: str, key: str
) -> dict[str, Any] | None:
    """The result cached on ``message_id`` under ``key``, ``None`` if none
    is."""
    return session.scalar(
        select(models.Result.payload).where(
            models.Result.message_id == message_id, models.Result.key == key
        )
    )


def _before(before: Time) -> ColumnElement[bool]:
    """The events whose latest TCA is before ``before``."""
    return models.Event.tca_max < before


def _count_before(session: Session, before: Time) -> int:
    """The number of events whose latest TCA is before ``before``."""
    counted = select(func.count()).select_from(models.Event).where(_before(before))
    return session.scalar(counted) or 0


class Store:
    """An open conjunction store; :func:`keepout.store.open_store` opens one.

    Use it as a context manager, or call :meth:`close` when done. Every
    method raises :class:`~keepout.errors.StoreError` naming the file if the
    database cannot be used: e.g. a write lock another process held past the
    busy timeout (:data:`~keepout.store.db.BUSY_TIMEOUT_MS`), a file that
    cannot be read or written.

    Parameters
    ----------
    path
        The database file; :attr:`path` is its resolved (absolute) path.
    engine
        The SQLAlchemy engine on it, from :func:`keepout.store.db.engine_for`.

    Examples
    --------
    ```python
    from pathlib import Path

    from keepout.settings import Settings
    from keepout.store import CdmInput, open_store

    with open_store("keepout.db") as store:  # a new store if missing
        # A sample CDM in the repository.
        report = store.ingest([CdmInput.read(Path("tests/cdm/data/tracss_v2.kvn"))])
        print(report.added, report.new_events)
        # ('TRACSS-TEST-0001',) ('5ede3a14c345',)
        settings = Settings(combined_hard_body_radius=20.0)  # [m]
        for event, assessment in store.assess(report.events, settings):
            action = assessment.decision.action
            print(event.event_id, assessment.level.key, action.label)
        # 5ede3a14c345 red maneuver
    ```
    """

    def __init__(self, path: Path, engine: Engine) -> None:
        self.path = Path(path).resolve()
        self.engine = engine
        self._writer = engine.execution_options(**{WRITE: True})

    @property
    def catalog_dir(self) -> Path:
        """The directory of the store's copies of the catalogs its CDMs were
        made from: ``<store file name>.catalogs`` beside the store file."""
        return self.path.with_name(f"{self.path.name}.catalogs")

    @contextmanager
    def _errors(self) -> Generator[None]:
        """Database errors as :class:`~keepout.errors.StoreError`, naming the
        file: a lock held past the busy timeout, a file that cannot be read or
        written or is no longer a database. A broken constraint stays an
        :class:`~sqlalchemy.exc.IntegrityError`: it is a bug, not the store's
        state.

        Raises
        ------
        StoreError
            On a database error but a broken constraint.
        IntegrityError
            On a broken constraint.
        """
        try:
            yield
        except IntegrityError:
            raise
        except DatabaseError as exc:
            raise StoreError(f"cannot use the store {self.path}: {exc.orig}") from exc

    @contextmanager
    def read(self) -> Generator[Session]:
        """A session that only reads.

        Its transaction starts with a plain (deferred) ``BEGIN``: readers do
        not wait for each other, nor for a writer (WAL), and see the
        database as it was at their first statement.

        Yields
        ------
        Session
            The session, rolled back and closed on exit.

        Raises
        ------
        StoreError
            If the database cannot be used, naming the file.
        """
        with self._errors(), Session(self.engine) as session:
            yield session

    @contextmanager
    def write(self) -> Generator[Session]:
        """A session that writes, in one transaction.

        The transaction starts on entry with ``BEGIN IMMEDIATE``, which takes
        the write lock at once: a second writer waits for the first to finish
        (up to :data:`~keepout.store.db.BUSY_TIMEOUT_MS`) instead of failing
        on a lock upgrade
        halfway through. Keep it short, e.g. compute Pc before it.

        Yields
        ------
        Session
            The session, committed on a normal exit and rolled back on an
            exception. Its objects stay loaded after the commit.

        Raises
        ------
        StoreError
            If the database cannot be used, naming the file: e.g. the write lock held
            by another connection past the busy timeout.
        """
        with (
            self._errors(),
            Session(self._writer, expire_on_commit=False) as session,
            session.begin(),
        ):
            session.connection()  # BEGIN IMMEDIATE now, not at the first statement
            yield session

    def ingest(
        self, messages: Sequence[CdmInput], *, catalog: CatalogReference | None = None
    ) -> IngestReport:
        """Store CDMs, each in its event; all of them or, on an error, none.

        The messages are taken in ``CREATION_DATE`` order (then by
        ``MESSAGE_ID``), as :func:`~keepout.events.group_events` takes them,
        in one write transaction. A
        message whose ``MESSAGE_ID`` is stored with the same content (the
        same KVN, whatever the format read) is reported as already stored.
        Any other joins the stored event (or one made earlier in the ingest)
        of the same ``OBJECT1`` and ``OBJECT2`` whose TCAs, with its own,
        spread over no more than
        :data:`~keepout.events.DEFAULT_TCA_TOLERANCE`, and whose
        ``CONJUNCTION_ID``, if both give one, is its own; if there is none, it
        starts a new event.

        Parameters
        ----------
        messages
            The CDMs, in any order.
        catalog
            The catalog a screening made them from, kept with the messages
            stored (not with those already stored). Its files are copied
            into :attr:`catalog_dir`, named by their SHA-256, so later
            downloads do not replace them.

        Returns
        -------
        IngestReport
            The messages stored and already stored, and their events.

        Raises
        ------
        StoreError
            If a ``MESSAGE_ID`` is stored, or given earlier, with other
            content; if a message fits more than one event (the message
            names them); if a catalog file cannot be copied; or if the
            database cannot be used.
        SourceUnavailable
            If a catalog file is gone or cannot be read.
        ChecksumMismatch
            If a catalog file changed since ``catalog`` was made.
        """
        if not messages:
            return IngestReport((), (), (), ())
        # In the order of group_events, so that the first CDM of a new event
        # names it as it does.
        reference = messages[0].cdm.creation_date
        order = sorted(
            range(len(messages)),
            key=lambda i: (
                elapsed_seconds(reference, messages[i].cdm.creation_date),
                messages[i].cdm.message_id,
            ),
        )
        added: dict[int, bool] = {}
        events: dict[int, EventId] = {}
        created: set[EventId] = set()
        with self.write() as session:
            catalog_row: models.Catalog | None = None
            for i in order:
                given = messages[i]
                message_id = MessageId(given.cdm.message_id)
                digest = _digest(given.cdm)
                stored = session.get(models.Cdm, message_id)
                if stored is not None:
                    if stored.digest != digest:
                        raise StoreError(
                            f"MESSAGE_ID {message_id} is stored with other content"
                        )
                    added[i], events[i] = False, stored.event_id
                    continue
                event, new = _event_of(session, given.cdm)
                if new:
                    created.add(event.id)
                if catalog is not None and catalog_row is None:
                    catalog_row = _catalog_row(
                        self._snapshot(catalog), self.path.parent
                    )
                session.add(
                    models.Cdm(
                        message_id=message_id,
                        event=event,
                        creation_date=given.cdm.creation_date,
                        tca=given.cdm.tca,
                        digest=digest,
                        content=given.text,
                        format=given.format,
                        origin=given.origin,
                        catalog=catalog_row,
                        ingested_at=_now(),
                    )
                )
                added[i], events[i] = True, event.id
        given_order = range(len(messages))
        touched = _unique([events[i] for i in given_order])
        ids = [MessageId(given.cdm.message_id) for given in messages]
        return IngestReport(
            added=tuple(ids[i] for i in given_order if added[i]),
            already_stored=tuple(ids[i] for i in given_order if not added[i]),
            events=touched,
            new_events=tuple(event for event in touched if event in created),
        )

    def _snapshot(self, catalog: CatalogReference) -> CatalogReference:
        """The catalog copied into :attr:`catalog_dir`.

        Raises
        ------
        StoreError
            If a copy cannot be written.
        SourceUnavailable
            If a file is gone or cannot be read.
        ChecksumMismatch
            If a file changed since the reference was made.
        """
        try:
            return catalog.snapshot(self.catalog_dir)
        except OSError as exc:
            raise StoreError(
                f"cannot copy the catalog into {self.catalog_dir}: {exc}"
            ) from exc

    def refresh(
        self, settings: Settings, *, event_ids: Collection[str] | None = None
    ) -> RefreshReport:
        """Compute and cache the Pc observations missing for ``settings``.

        Settings with the same :func:`~keepout.store.codec.result_key` share
        their observations. Each CDM is parsed just before its Pc is computed,
        outside any transaction (in ``"auto"`` mode it can take up to a
        minute). The observations are written in short transactions of 100
        observations or 1 s of computing, whichever comes first, so that an
        interrupted refresh keeps most of what it computed and other
        processes are not held up.
        One that another process cached meanwhile, or of a CDM purged
        meanwhile, is not written.

        If the settings give no hard-body radius, a CDM without ``HBR`` for
        an object has no Pc: refreshing every event skips it (and reports
        it), refreshing given events is an error.

        Parameters
        ----------
        settings
            How Pc is computed.
        event_ids
            The events to refresh; all events if ``None``.

        Returns
        -------
        RefreshReport
            How many observations it computed and cached, and the CDMs it
            skipped for lack of ``HBR``.

        Raises
        ------
        StoreError
            If an event is not in the store, or if the database cannot be used.
        HardBodyRadiusError
            If ``event_ids`` are given, the settings give no hard-body radius
            and a CDM of those events lacks ``HBR`` for an object (checked
            before any Pc is computed).
        CdmParseError
            If a stored CDM no longer parses (a store edited by hand).
        """
        key = result_key(settings)
        query = (
            select(models.Cdm.event_id, models.Cdm.message_id)
            .where(~_cached(key))
            .order_by(
                models.Cdm.event_id, models.Cdm.creation_date, models.Cdm.message_id
            )
        )
        with self.read() as session:
            if event_ids is not None:
                _require_events(session, event_ids)
                query = query.where(models.Cdm.event_id.in_(event_ids))
            missing = [(event_id, m) for event_id, m in session.execute(query)]
        parsed = (cdm for _, cdm in self._parsed(missing))
        lacking = _lacking_hard_body_radius(settings, parsed)
        if event_ids is not None:
            _refuse_lacking_hard_body_radius(lacking)
        skipped = tuple(MessageId(cdm.message_id) for cdm in lacking)
        unknown = set(skipped)
        todo = [(event_id, m) for event_id, m in missing if m not in unknown]
        written = 0
        pending: list[PcObservation] = []
        since = time.monotonic()
        for event_id, cdm in self._parsed(todo):
            pending.append(_observation(event_id, cdm, settings))
            if (
                len(pending) >= _BATCH_SIZE
                or time.monotonic() - since >= _BATCH_SECONDS
            ):
                written += self._write(key, pending)
                pending, since = [], time.monotonic()
        written += self._write(key, pending)
        return RefreshReport(written, skipped)

    def _parsed(
        self, rows: Sequence[tuple[EventId, MessageId]]
    ) -> Iterator[tuple[EventId, Cdm]]:
        """The stored CDMs of ``rows``, each parsed only when asked for: their
        texts are read :data:`_BATCH_SIZE` at a time, in short read
        transactions. A CDM purged meanwhile is left out.

        Raises
        ------
        StoreError
            If the database cannot be used.
        CdmParseError
            If a stored CDM no longer parses (a store edited by hand).
        """
        for start in range(0, len(rows), _BATCH_SIZE):
            chunk = rows[start : start + _BATCH_SIZE]
            with self.read() as session:
                found = session.execute(
                    select(models.Cdm.message_id, models.Cdm.content).where(
                        models.Cdm.message_id.in_([m for _, m in chunk])
                    )
                )
                texts = dict(found.all())
            for event_id, message_id in chunk:
                if message_id in texts:
                    yield event_id, keepout.cdm.loads(texts.pop(message_id))

    def _write(self, key: str, observations: Sequence[PcObservation]) -> int:
        """Write observations under ``key`` in one write transaction (none if
        there are none); how many it wrote (see :func:`_write_results`)."""
        if not observations:
            return 0
        with self.write() as session:
            return _write_results(session, key, observations)

    def cache(self, settings: Settings, observations: Iterable[PcObservation]) -> int:
        """Cache Pc observations computed with ``settings`` elsewhere, e.g. by
        the screening that issued the CDMs, so that :meth:`refresh` does not
        compute them again.

        Each observation is cached under its
        :attr:`~keepout.risk.PcObservation.message_id`, in one write
        transaction. One of a CDM not stored, or already cached for
        ``settings``, is not written.

        Parameters
        ----------
        settings
            The settings the observations were computed with.
        observations
            The observations, each of one stored CDM, as
            :meth:`keepout.events.ConjunctionEvent.observations` computes them.

        Returns
        -------
        int
            How many it cached.

        Raises
        ------
        StoreError
            If the database cannot be used.
        """
        return self._write(result_key(settings), list(observations))

    def assess(
        self, event_ids: Collection[str], settings: Settings
    ) -> list[AssessedEvent]:
        """Stored events, each assessed from its Pc observations for
        ``settings`` and their thresholds; the Pc the store lacks is computed
        and cached first (:meth:`refresh`).

        Parameters
        ----------
        event_ids
            The store's ids of the events to assess.
        settings
            How Pc is computed and the risk judged.

        Returns
        -------
        list[AssessedEvent]
            The events, with every CDM stored of them, by their first TCA
            (then by id).

        Raises
        ------
        StoreError
            As :meth:`refresh` and :meth:`events`.
        HardBodyRadiusError
            If the settings give no hard-body radius and a CDM lacks ``HBR``.
        """
        self.refresh(settings, event_ids=event_ids)
        return [
            AssessedEvent(
                event,
                assess(
                    self.observations(event.event_id, settings),
                    thresholds=settings.thresholds,
                ),
            )
            for event in self.events(event_ids=event_ids)
        ]

    def summaries(
        self,
        settings: Settings,
        *,
        as_of: Time,
        include_past: bool = False,
        offset: int = 0,
        limit: int = 25,
    ) -> SummaryPage:
        """A page of the stored events, each summed up
        (:mod:`keepout.store.summaries`).

        It reads the store in one read transaction and computes no Pc: an
        event whose CDM lacks its Pc for ``settings`` has no assessment.
        The list filters and orders on each event's latest TCA of any of
        its CDMs, which may differ from
        :attr:`~keepout.store.EventSummary.tca` (its latest CDM's) by at most
        :data:`~keepout.events.DEFAULT_TCA_TOLERANCE`.

        Parameters
        ----------
        settings
            The settings of the cached Pc (by their
            :func:`~keepout.store.codec.result_key`) and their thresholds.
        as_of
            The time of the list: an event whose latest TCA is before it is
            past.
        include_past
            List the past events too.
        offset
            How many events of the list come before the page: not negative.
        limit
            Most events of the page: 1 to
            :data:`~keepout.store.summaries.MAX_LIMIT`.

        Returns
        -------
        SummaryPage
            The page, its events by latest TCA (then by id), and how many
            events the list has.

        Raises
        ------
        ValueError
            If ``offset`` or ``limit`` is out of its bounds.
        StoreError
            If a cached observation cannot be read, or the database cannot
            be used.
        CdmParseError
            If the latest CDM of a listed event, parsed when its Pc rests
            on an assumed covariance, no longer parses (a store edited by
            hand).
        """
        check_page(offset, limit)
        with self.read() as session:
            events, total = read_page(
                session,
                result_key(settings),
                as_of=None if include_past else as_of,
                offset=offset,
                limit=limit,
            )
        return SummaryPage(summarize(events, settings.thresholds), total, offset, limit)

    def event(self, event_id: str) -> ConjunctionEvent:
        """A stored event, its CDMs read from their stored text.

        Parameters
        ----------
        event_id
            The store's id of the event.

        Returns
        -------
        ConjunctionEvent
            The event and its CDMs.

        Raises
        ------
        StoreError
            If there is no such event, or if the database cannot be used.
        CdmParseError
            If a stored CDM no longer parses (a store edited by hand).
        EventError
            If the stored CDMs no longer make an event (likewise).
        """
        with self.read() as session:
            texts = _texts(session, _cdms_of(session, event_id, None))
        return _parsed_event(event_id, texts)

    def events(
        self, *, event_ids: Collection[str] | None = None
    ) -> list[ConjunctionEvent]:
        """The stored events, by their first TCA (then by id).

        Parameters
        ----------
        event_ids
            The events to list (only their CDMs are read); all if ``None``.

        Returns
        -------
        list of ConjunctionEvent
            The events, by first TCA then id.

        Raises
        ------
        StoreError
            If an event of ``event_ids`` is not in the store, or if the
            database cannot be used.
        CdmParseError
            If a stored CDM of a listed event no longer parses (a store
            edited by hand).
        EventError
            If the stored CDMs of a listed event no longer make one
            (likewise).
        """
        listed = select(models.Event.id)
        if event_ids is not None:
            listed = listed.where(models.Event.id.in_(event_ids))
        with self.read() as session:
            if event_ids is not None:
                _require_events(session, event_ids)
            ids = session.scalars(
                listed.order_by(models.Event.tca_min, models.Event.id)
            ).all()
            rows = session.execute(
                select(models.Cdm.event_id, models.Cdm.content).where(
                    models.Cdm.event_id.in_(listed.scalar_subquery())
                )
            ).all()
        cdms: dict[str, list[Cdm]] = {}
        for event_id, content in rows:
            cdms.setdefault(event_id, []).append(keepout.cdm.loads(content))
        return [ConjunctionEvent(event_id, tuple(cdms[event_id])) for event_id in ids]

    def cached_observations(
        self, event_id: str, settings: Settings
    ) -> tuple[tuple[MessageId, PcObservation | None], ...]:
        """Each CDM of an event with its Pc observation cached for
        ``settings``, oldest CDM first.

        Parameters
        ----------
        event_id
            The store's id of the event.
        settings
            The settings they were computed with (by their
            :func:`~keepout.store.codec.result_key`).

        Returns
        -------
        tuple of (MessageId, PcObservation or None)
            Each CDM's ``MESSAGE_ID`` and its observation; ``None`` where
            it is not cached (:meth:`refresh` caches it).

        Raises
        ------
        StoreError
            If there is no such event, a cached observation cannot be read,
            or the database cannot be used.
        """
        with self.read() as session:
            _require_events(session, [event_id])
            key = result_key(settings)
            rows = payloads_in(session, [event_id], key)[EventId(event_id)]
        return _paired(rows)

    def observations(
        self, event_id: str, settings: Settings
    ) -> tuple[PcObservation, ...]:
        """The cached Pc observations of an event's CDMs, oldest CDM first,
        all or none (:meth:`cached_observations` gives those there are).

        Parameters
        ----------
        event_id
            The store's id of the event.
        settings
            The settings they were computed with (by their
            :func:`~keepout.store.codec.result_key`).

        Returns
        -------
        tuple of PcObservation
            One observation per CDM, oldest CDM first.

        Raises
        ------
        StoreError
            If there is no such event; if a CDM's observation is not cached
            for these settings (:meth:`refresh` caches them) or cannot be
            read; or if the database cannot be used.
        """
        pairs = self.cached_observations(event_id, settings)
        missing = [message_id for message_id, item in pairs if item is None]
        if missing:
            raise StoreError(
                f"no Pc of {', '.join(missing)} in event {event_id} for these "
                "settings: refresh the store first"
            )
        return tuple(item for _, item in pairs if item is not None)

    def detail(self, event_id: str, settings: Settings) -> EventDetail | None:
        """A stored event, read in one transaction: its CDMs, their cached
        Pc and its active refresh jobs. It computes no Pc.

        Parameters
        ----------
        event_id
            The store's id of the event.
        settings
            The settings of the cached Pc and of the refresh jobs (by their
            :func:`~keepout.store.codec.result_key`).

        Returns
        -------
        EventDetail or None
            The event; ``None`` if there is no such event.

        Raises
        ------
        StoreError
            If a cached observation cannot be read, or the database cannot
            be used.
        CdmParseError
            If a stored CDM no longer parses (a store edited by hand).
        EventError
            If the stored CDMs no longer make an event (likewise).
        """
        key = result_key(settings)
        with self.read() as session:
            if session.get(models.Event, event_id) is None:
                return None
            texts = _texts(session, models.Cdm.event_id == event_id)
            rows = payloads_in(session, [event_id], key)[EventId(event_id)]
            jobs = active_refresh_in(session, event_id, key)
        return EventDetail(_parsed_event(event_id, texts), _paired(rows), jobs)

    def plan_detail(self, event_id: str, settings: Settings) -> PlanDetail | None:
        """A stored event as its manoeuvre planner shows it, read in one
        transaction: its CDMs (each parsed once), the CDM and key its plan
        with ``settings`` is cached under (as :meth:`maneuver`), the plan
        if it is cached, and its manoeuvre jobs (as
        :meth:`~keepout.store.JobQueue.plan_jobs`). It plans nothing.

        Parameters
        ----------
        event_id
            The store's id of the event.
        settings
            The operator's settings of the plan (by their
            :func:`~keepout.store.codec.maneuver_key`).

        Returns
        -------
        PlanDetail or None
            The event, its plan and its jobs; ``None`` if there is no such
            event.

        Raises
        ------
        StoreError
            If the cached plan cannot be read, or the database cannot be
            used.
        ValueError
            If a settings file cannot hold ``settings``, as
            :func:`~keepout.store.codec.maneuver_key`.
        CdmParseError
            If a stored CDM no longer parses (a store edited by hand).
        EventError
            If the stored CDMs no longer make an event (likewise).
        """
        from keepout.store.jobs import PlanDetail, plan_jobs_in
        from keepout.store.maneuver_codec import outcome_from_json

        with self.read() as session:
            if session.get(models.Event, event_id) is None:
                return None
            latest, key = maneuver_slot_in(
                session, self.path.parent, event_id, settings
            )
            payload = _cached_payload(session, latest, key)
            texts = _texts(session, models.Cdm.event_id == event_id)
            jobs = plan_jobs_in(session, event_id, latest, key)
        event = _parsed_event(event_id, texts)
        outcome = None if payload is None else outcome_from_json(payload, event)
        return PlanDetail(event, latest, key, outcome, jobs)

    def catalog(self, event_id: str) -> CatalogReference | None:
        """The catalog the event's latest CDM was made from.

        Parameters
        ----------
        event_id
            The store's id of the event.

        Returns
        -------
        CatalogReference or None
            The catalog's files, at their copies in :attr:`catalog_dir` (the
            store keeps their paths relative to its directory, so the
            directory can move); ``None`` if the latest CDM was not made by
            a screening with a catalog.

        Raises
        ------
        StoreError
            If there is no such event, or if the database cannot be used.
        """
        with self.read() as session:
            return catalog_in(session, self.path.parent, latest_in(session, event_id))

    def maneuver(self, event_id: str, settings: Settings) -> ManeuverOutcome | None:
        """The cached manoeuvre outcome of an event: the one planned on its
        latest CDM with ``settings`` and the catalog of that CDM.

        A CDM ingested since makes it miss: the outcome is of the event as
        it was.

        Parameters
        ----------
        event_id
            The store's id of the event.
        settings
            The operator's settings of the plan (by their
            :func:`~keepout.store.codec.maneuver_key`).

        Returns
        -------
        ManeuverOutcome or None
            The outcome, with the event built from all its stored CDMs;
            ``None`` if none is cached.

        Raises
        ------
        StoreError
            If there is no such event, if the cached outcome cannot be read,
            or if the database cannot be used.
        ValueError
            If a settings file cannot hold ``settings``, as
            :func:`~keepout.store.codec.maneuver_key`.
        CdmParseError
            If a stored CDM no longer parses (a store edited by hand).
        EventError
            If the stored CDMs no longer make an event (likewise).
        """
        with self.read() as session:
            latest, key = maneuver_slot_in(
                session, self.path.parent, event_id, settings
            )
            payload = _cached_payload(session, latest, key)
            if payload is None:
                return None
            texts = _texts(session, _cdms_of(session, event_id, None))
        from keepout.store.maneuver_codec import outcome_from_json

        return outcome_from_json(payload, _parsed_event(event_id, texts))

    def save_maneuver(
        self, message_id: str, settings: Settings, outcome: ManeuverOutcome
    ) -> bool:
        """Cache a manoeuvre outcome on a CDM, in one write transaction.

        Parameters
        ----------
        message_id
            The ``MESSAGE_ID`` of the CDM the outcome was planned on: the
            latest CDM of its event.
        settings
            The operator's settings of the plan; with the CDM's catalog they
            give its :func:`~keepout.store.codec.maneuver_key`.
        outcome
            The outcome, from :func:`~keepout.maneuver.plan_event`.

        Returns
        -------
        bool
            Whether it was written: not if the CDM is not stored (e.g. purged
            meanwhile) or an outcome is already cached on it under its key.

        Raises
        ------
        StoreError
            If the database cannot be used.
        ValueError
            If a settings file cannot hold ``settings``, as
            :func:`~keepout.store.codec.maneuver_key`.
        """
        from keepout.store.maneuver_codec import outcome_to_json

        payload = outcome_to_json(outcome)
        with self.write() as session:
            key = _maneuver_key_in(session, self.path.parent, message_id, settings)
            return save_maneuver_in(session, message_id, key, payload)

    def count_before(self, before: Time) -> int:
        """The number of events whose latest TCA is before a time: those
        :meth:`purge` would delete now.

        Parameters
        ----------
        before
            The time; an event whose latest TCA is at it is not counted.

        Returns
        -------
        int
            The number of events.

        Raises
        ------
        StoreError
            If the database cannot be used.
        """
        with self.read() as session:
            return _count_before(session, before)

    def purge(self, before: Time) -> int:
        """Delete the events whose latest TCA is before a time, with their
        CDMs, cached results and jobs, and the catalogs no CDM is left made from,
        and every copy in :attr:`catalog_dir` no catalog refers to (also
        those a failed ingest left).

        Parameters
        ----------
        before
            The time; an event whose latest TCA is at it is kept.

        Returns
        -------
        int
            The number of events deleted.

        Raises
        ------
        StoreError
            If the database cannot be used, or a catalog copy cannot be
            deleted (the database is purged then).
        """
        referenced = exists().where(models.Cdm.catalog_id == models.Catalog.id)
        with self.write() as session:
            count = _count_before(session, before)
            session.execute(delete_rows(models.Event).where(_before(before)))
            # A catalog goes only once no CDM refers to it (ON DELETE RESTRICT).
            session.execute(delete_rows(models.Catalog).where(~referenced))
        self._delete_copies()
        return count

    def _delete_copies(self) -> None:
        """Delete the copies in :attr:`catalog_dir` whose digest no catalog
        refers to: those of deleted catalogs, and those a failed ingest left.

        This runs in a write transaction of its own: an ingest writes or
        reuses its copies under the same write lock, so it either comes
        first, and its row keeps the copy, or after, and writes the copy
        again. Only files named as the store names its copies
        (``<sha256>…``) are deleted.

        Raises
        ------
        StoreError
            If a copy cannot be deleted, or the database cannot be used.
        """
        if not self.catalog_dir.is_dir():
            return
        with self.write() as session:
            used = set(session.scalars(select(models.CatalogFile.sha256)))
            for path in sorted(self.catalog_dir.iterdir()):
                named = _COPY_NAME.match(path.name)
                if named is None or named[1] in used:
                    continue
                try:
                    path.unlink()
                except OSError as exc:
                    raise StoreError(
                        f"cannot delete the catalog copy {path}: {exc}"
                    ) from exc

    def close(self) -> None:
        """Close the store's connections; closing it again does nothing."""
        self.engine.dispose()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
