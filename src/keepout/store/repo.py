"""The conjunction store: a SQLite database of events, CDMs and results.

CDMs are only added (:meth:`Store.ingest`): a message is stored once, by its
``MESSAGE_ID``, and given again it is reported as already stored. Each CDM
joins the stored event of its encounter, by the rule of
:func:`keepout.events.group_events`: same objects (``OBJECT1``, ``OBJECT2``
by designator), TCAs within :data:`~keepout.events.DEFAULT_TCA_TOLERANCE`, no
other ``CONJUNCTION_ID``. An event keeps the id it was given when its first
CDM came, eight hexadecimal characters, however many CDMs follow.

Each CDM's Pc observation is cached (:meth:`Store.refresh`) under the key of
the settings it was computed with
(:attr:`~keepout.store.config.Settings.result_key`).
"""

from __future__ import annotations

import hashlib
import re
import secrets
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Self

import astropy.units as u
from astropy.time import Time
from sqlalchemy import and_, exists, false, func, or_, select
from sqlalchemy import delete as delete_rows
from sqlalchemy.exc import DatabaseError, IntegrityError
from sqlalchemy.orm import Session

import keepout.cdm
from keepout.errors import HardBodyRadiusError, StoreError
from keepout.events import (
    DEFAULT_TCA_TOLERANCE,
    ConjunctionEvent,
    without_hard_body_radius,
)
from keepout.sources.catalog import CatalogFileRef, CatalogReference
from keepout.store import models
from keepout.store.codec import observation_from_json, observation_to_json
from keepout.units import elapsed_seconds

if TYPE_CHECKING:
    from collections.abc import Collection, Generator, Sequence
    from types import TracebackType

    from sqlalchemy import ColumnElement, Engine

    from keepout.cdm import Cdm, CdmFormat
    from keepout.risk import PcObservation
    from keepout.store.config import Settings

__all__ = ["WRITE", "CdmInput", "IngestReport", "RefreshReport", "Store"]

WRITE: Final = "keepout_write"
"""Execution option of a write transaction: it starts with ``BEGIN IMMEDIATE``
(see :func:`keepout.store.db.engine_for`)."""

_COPY_NAME: Final = re.compile(r"([0-9a-f]{64})(?:\.|$)")
"""The name of a catalog copy: its SHA-256, then the original's suffixes."""

_WINDOW: Final = (DEFAULT_TCA_TOLERANCE + 1.0) * u.s
"""How far from a TCA the database searches for the events it may join: the
tolerance and a second of slack; the exact test follows."""


def _iso(time: Time) -> str:
    """A time as the database writes it: UTC ISO-8601 with microseconds."""
    return str(Time(time.utc, precision=6).isot)


def _time(text: str) -> Time:
    """The time the database wrote as ``text``."""
    return Time(text, format="isot", scale="utc")


def _now() -> str:
    """The current time as the database writes it."""
    return _iso(Time.now())


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
            The file.

        Raises
        ------
        CdmParseError
            If the file is broken gzip data, not UTF-8 text or not a CDM.
        OSError
            If it cannot be read.
        """
        text = keepout.cdm.read_text(path)
        return cls(
            keepout.cdm.loads(text), text, keepout.cdm.format_of(text), str(path)
        )

    @classmethod
    def of(cls, cdm: Cdm, origin: str) -> CdmInput:
        """A CDM made in memory, e.g. by a screening, kept as its KVN.

        Parameters
        ----------
        cdm
            The message.
        origin
            Where it came from, e.g. ``"screen"``.
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

    added: tuple[str, ...]
    already_stored: tuple[str, ...]
    events: tuple[str, ...]
    new_events: tuple[str, ...]


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
    skipped: tuple[str, ...] = ()


def _unique(items: Sequence[str]) -> tuple[str, ...]:
    """``items`` without repeats, in the order they first come."""
    return tuple(dict.fromkeys(items))


def _spread(event: models.Event, session: Session, cdm: Cdm) -> float:
    """The spread [s] of the TCAs of the event's CDMs and ``cdm``, at full
    precision, as :class:`~keepout.events.ConjunctionEvent` measures it.

    The database's TCAs are rounded to the microsecond: they find the
    event's earliest and latest CDMs, whose own TCAs are then compared.
    """
    extremes = session.scalars(
        select(models.Cdm.content).where(
            models.Cdm.event_id == event.id,
            or_(
                models.Cdm.tca == event.tca_min,
                models.Cdm.tca == event.tca_max,
            ),
        )
    )
    tcas = [cdm.tca, *(keepout.cdm.loads(text).tca for text in extremes)]
    offsets = [elapsed_seconds(tcas[0], tca) for tca in tcas]
    return max(offsets) - min(offsets)


def _new_event_id(session: Session) -> str:
    """A random event id that no stored event has."""
    while True:
        candidate = secrets.token_hex(4)
        if session.get(models.Event, candidate) is None:
            return candidate


def _event_of(session: Session, cdm: Cdm, tca: str) -> tuple[models.Event, bool]:
    """The event that ``cdm`` joins, updated for it, and whether it is new;
    ``tca`` is the CDM's TCA as the database writes it.

    Raises
    ------
    StoreError
        If it fits more than one stored event.
    """
    query = select(models.Event).where(
        models.Event.object1 == cdm.object1.object_id,
        models.Event.object2 == cdm.object2.object_id,
        models.Event.tca_max >= _iso(cdm.tca - _WINDOW),
        models.Event.tca_min <= _iso(cdm.tca + _WINDOW),
    )
    if cdm.conjunction_id is not None:
        query = query.where(
            or_(
                models.Event.conjunction_id.is_(None),
                models.Event.conjunction_id == cdm.conjunction_id,
            )
        )
    fitting = [
        event
        for event in session.scalars(query)
        if _spread(event, session, cdm) <= DEFAULT_TCA_TOLERANCE
    ]
    if len(fitting) > 1:
        ids = ", ".join(sorted(event.id for event in fitting))
        raise StoreError(f"CDM {cdm.message_id} fits more than one event: {ids}")
    if fitting:
        (event,) = fitting
        event.tca_min = min(event.tca_min, tca)
        event.tca_max = max(event.tca_max, tca)
        if event.conjunction_id is None:
            event.conjunction_id = cdm.conjunction_id
        return event, False
    event = models.Event(
        id=_new_event_id(session),
        object1=cdm.object1.object_id,
        object2=cdm.object2.object_id,
        conjunction_id=cdm.conjunction_id,
        tca_min=tca,
        tca_max=tca,
        archived=False,
        created_at=_now(),
    )
    session.add(event)
    # Before its CDM: without relationships, the session does not order the
    # inserts by foreign key.
    session.flush()
    return event, True


def _add_catalog(session: Session, reference: CatalogReference, base: Path) -> int:
    """Store a catalog reference, its files' paths relative to ``base`` (the
    store's directory); its row's id."""
    catalog = models.Catalog(label=reference.label)
    session.add(catalog)
    session.flush()
    session.add_all(
        models.CatalogFile(
            catalog_id=catalog.id,
            name=file.name,
            path=file.path.relative_to(base).as_posix(),
            url=file.url,
            retrieved=None if file.retrieved is None else _iso(file.retrieved),
            sha256=file.sha256,
        )
        for file in reference.files
    )
    return catalog.id


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


def _before(before: Time) -> ColumnElement[bool]:
    """The events whose latest TCA is before ``before``."""
    return models.Event.tca_max < _iso(before)


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

        The messages are taken in TCA order, in one write transaction. A
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
        tcas = [_iso(given.cdm.tca) for given in messages]
        order = sorted(range(len(messages)), key=lambda i: (tcas[i], i))
        added: dict[int, bool] = {}
        events: dict[int, str] = {}
        created: set[str] = set()
        with self.write() as session:
            catalog_id: int | None = None
            for i in order:
                given = messages[i]
                message_id = given.cdm.message_id
                digest = _digest(given.cdm)
                stored = session.get(models.Cdm, message_id)
                if stored is not None:
                    if stored.digest != digest:
                        raise StoreError(
                            f"MESSAGE_ID {message_id} is stored with other content"
                        )
                    added[i], events[i] = False, stored.event_id
                    continue
                event, new = _event_of(session, given.cdm, tcas[i])
                if new:
                    created.add(event.id)
                if catalog is not None and catalog_id is None:
                    catalog_id = _add_catalog(
                        session, self._snapshot(catalog), self.path.parent
                    )
                session.add(
                    models.Cdm(
                        message_id=message_id,
                        event_id=event.id,
                        creation_date=_iso(given.cdm.creation_date),
                        tca=tcas[i],
                        digest=digest,
                        content=given.text,
                        format=given.format,
                        origin=given.origin,
                        catalog_id=catalog_id,
                        ingested_at=_now(),
                    )
                )
                session.flush()
                added[i], events[i] = True, event.id
        given_order = range(len(messages))
        touched = _unique([events[i] for i in given_order])
        return IngestReport(
            added=tuple(messages[i].cdm.message_id for i in given_order if added[i]),
            already_stored=tuple(
                messages[i].cdm.message_id for i in given_order if not added[i]
            ),
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

        Settings with the same :attr:`~keepout.store.config.Settings.result_key`
        share their observations. Each observation is computed outside any
        transaction (in ``"auto"`` mode it can take up to a minute) and
        written in a short one of its own, so that an interrupted refresh
        keeps what it computed and other processes are not held up. One
        that another process cached meanwhile, or of a CDM purged meanwhile,
        is not written.

        If the settings give no hard-body radius, a CDM without ``HBR`` for
        an object has no Pc: refreshing every event skips it (and reports
        it), refreshing given events is an error.

        Parameters
        ----------
        settings
            How Pc is computed.
        event_ids
            The events to refresh; all events (archived too) if ``None``.

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
        key = settings.result_key
        cached = exists().where(
            models.Result.message_id == models.Cdm.message_id,
            models.Result.key == key,
        )
        query = (
            select(models.Cdm.event_id, models.Cdm.content)
            .where(~cached)
            .order_by(
                models.Cdm.event_id, models.Cdm.creation_date, models.Cdm.message_id
            )
        )
        with self.read() as session:
            if event_ids is not None:
                _require_events(session, event_ids)
                query = query.where(models.Cdm.event_id.in_(event_ids))
            missing = [
                (event_id, keepout.cdm.loads(content))
                for event_id, content in session.execute(query)
            ]
        skipped: tuple[str, ...] = ()
        if settings.combined_hard_body_radius is None:
            skipped = without_hard_body_radius(cdm for _, cdm in missing)
            if skipped and event_ids is not None:
                raise HardBodyRadiusError(
                    f"no HBR for both objects in {', '.join(skipped)}: give the "
                    "combined hard-body radius in the settings"
                )
        unknown = set(skipped)
        written = 0
        for event_id, cdm in missing:
            if cdm.message_id in unknown:
                continue
            (observation,) = ConjunctionEvent(event_id, (cdm,)).observations(
                compute=settings.compute,
                policy=settings.policy,
                combined_hard_body_radius=settings.combined_hard_body_radius,
            )
            with self.write() as session:
                if session.get(models.Cdm, cdm.message_id) is None:
                    continue
                done = session.scalar(
                    select(models.Result.id).where(
                        models.Result.message_id == cdm.message_id,
                        models.Result.key == key,
                    )
                )
                if done is not None:
                    continue
                session.add(
                    models.Result(
                        message_id=cdm.message_id,
                        key=key,
                        payload=observation_to_json(observation),
                        created_at=_now(),
                    )
                )
            written += 1
        return RefreshReport(written, skipped)

    def event(self, event_id: str) -> ConjunctionEvent:
        """A stored event, its CDMs read from their stored text.

        Parameters
        ----------
        event_id
            The store's id of the event.

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
            _require_events(session, [event_id])
            texts = session.scalars(
                select(models.Cdm.content).where(models.Cdm.event_id == event_id)
            ).all()
        return ConjunctionEvent(
            event_id, tuple(keepout.cdm.loads(text) for text in texts)
        )

    def events(
        self,
        *,
        include_archived: bool = False,
        event_ids: Collection[str] | None = None,
    ) -> list[ConjunctionEvent]:
        """The stored events, by their first TCA (then by id).

        Parameters
        ----------
        include_archived
            Whether archived events are listed too.
        event_ids
            The events to list (only their CDMs are read); all if ``None``.

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
        if not include_archived:
            listed = listed.where(models.Event.archived == false())
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

    def observations(
        self, event_id: str, settings: Settings
    ) -> tuple[PcObservation, ...]:
        """The cached Pc observations of an event's CDMs, oldest CDM first.

        Parameters
        ----------
        event_id
            The store's id of the event.
        settings
            The settings they were computed with (by their
            :attr:`~keepout.store.config.Settings.result_key`).

        Raises
        ------
        StoreError
            If there is no such event; if a CDM's observation is not cached
            for these settings (:meth:`refresh` caches them) or cannot be
            read; or if the database cannot be used.
        """
        joined = and_(
            models.Result.message_id == models.Cdm.message_id,
            models.Result.key == settings.result_key,
        )
        with self.read() as session:
            _require_events(session, [event_id])
            rows = session.execute(
                select(models.Cdm.message_id, models.Result.payload)
                .outerjoin(models.Result, joined)
                .where(models.Cdm.event_id == event_id)
                .order_by(models.Cdm.creation_date, models.Cdm.message_id)
            ).all()
        missing = [message_id for message_id, payload in rows if payload is None]
        if missing:
            raise StoreError(
                f"no Pc of {', '.join(missing)} in event {event_id} for these "
                "settings: refresh the store first"
            )
        return tuple(observation_from_json(payload) for _, payload in rows)

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
            _require_events(session, [event_id])
            catalog_id = session.scalar(
                select(models.Cdm.catalog_id)
                .where(models.Cdm.event_id == event_id)
                .order_by(models.Cdm.creation_date.desc(), models.Cdm.message_id.desc())
                .limit(1)
            )
            if catalog_id is None:
                return None
            label = session.execute(
                select(models.Catalog.label).where(models.Catalog.id == catalog_id)
            ).scalar_one()
            files = session.scalars(
                select(models.CatalogFile)
                .where(models.CatalogFile.catalog_id == catalog_id)
                .order_by(models.CatalogFile.id)
            ).all()
        return CatalogReference(
            label,
            tuple(
                CatalogFileRef(
                    file.name,
                    self.path.parent / file.path,
                    file.sha256,
                    url=file.url,
                    retrieved=None if file.retrieved is None else _time(file.retrieved),
                )
                for file in files
            ),
        )

    def count_before(self, before: Time) -> int:
        """The number of events (archived too) whose latest TCA is before a
        time: those :meth:`purge` would delete now.

        Parameters
        ----------
        before
            The time; an event whose latest TCA is at it is not counted.

        Raises
        ------
        StoreError
            If the database cannot be used.
        """
        with self.read() as session:
            return _count_before(session, before)

    def purge(self, before: Time) -> int:
        """Delete the events whose latest TCA is before a time, with their
        CDMs and cached results, and the catalogs no CDM is left made from,
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
