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

What it is given and reports is in :mod:`keepout.store.inputs`; its reads
in a session, shared with the job queue, in :mod:`keepout.store.queries`.
"""

from __future__ import annotations

import hashlib
import re
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Final, Self

import astropy.units as u
from astropy.time import Time
from sqlalchemy import delete as delete_rows
from sqlalchemy import exists, func, select
from sqlalchemy.exc import DatabaseError, IntegrityError
from sqlalchemy.orm import Session

import keepout.cdm
from keepout.errors import StoreError
from keepout.events import (
    DEFAULT_TCA_TOLERANCE,
    AssessedEvent,
    ConjunctionEvent,
    EventBounds,
    stable_event_id,
)
from keepout.risk import assess
from keepout.store import computing, models
from keepout.store.codec import maneuver_key, observation_to_json, result_key
from keepout.store.computing import (
    batches,
    lacking_hard_body_radius,
    observation_of,
    parsed_cdm,
    parsed_event,
    refuse_lacking_hard_body_radius,
)
from keepout.store.inputs import IngestReport, RefreshReport
from keepout.store.models import EventId, MessageId
from keepout.store.queries import (
    active_refresh_in,
    cached,
    cached_payload_in,
    catalog_in,
    cdms_of,
    insert_results_in,
    maneuver_slot_in,
    no_event,
    payloads_in,
    plan_jobs_in,
    read_page,
    require_events,
    require_latest_in,
    save_maneuver_in,
    texts_in,
)
from keepout.store.summaries import (
    EventDetail,
    PlanDetail,
    SummaryPage,
    check_page,
    complete_of,
    decoded,
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

    from sqlalchemy import ColumnElement, Engine

    from keepout.cdm import Cdm
    from keepout.maneuver import ManeuverOutcome
    from keepout.risk import PcObservation
    from keepout.settings import Settings
    from keepout.sources.catalog import CatalogReference
    from keepout.store.inputs import CdmInput

__all__ = ["WRITE", "Store"]

WRITE: Final = "keepout_write"
"""Execution option of a write transaction: it starts with ``BEGIN IMMEDIATE``
(see :func:`keepout.store.db.engine_for`)."""

_COPY_NAME: Final = re.compile(r"([0-9a-f]{64})(?:\.|$)")
"""The name of a catalog copy: its SHA-256, then the original's suffixes."""

_WINDOW: Final = (DEFAULT_TCA_TOLERANCE + 1.0) * u.s
"""How far from a TCA the database searches for the events it may join: the
tolerance and a second of slack; the exact test follows."""


def _digest(cdm: Cdm) -> str:
    """What makes two messages the same: the SHA-256 (hex) of the KVN."""
    return hashlib.sha256(keepout.cdm.kvn.dumps(cdm).encode("utf-8")).hexdigest()


def _unique(items: Iterable[EventId]) -> tuple[EventId, ...]:
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
    """The event that ``cdm`` joins (:class:`~keepout.events.EventBounds`),
    updated for it, and whether it is new.

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
        raise StoreError(
            f"CDM {cdm.message_id} fits more than one event: {ids}",
            message_id=cdm.message_id,
        )
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
        created_at=Time.now(),
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


def _texts_by_event(
    session: Session, event_ids: Collection[str] | None
) -> tuple[Sequence[EventId], dict[EventId, list[str]]]:
    """The ids of the stored events of ``event_ids`` (all if ``None``), by
    first TCA then id, and the texts of each one's CDMs."""
    listed = select(models.Event.id)
    if event_ids is not None:
        listed = listed.where(models.Event.id.in_(event_ids))
    ids = session.scalars(listed.order_by(models.Event.tca_min, models.Event.id)).all()
    rows = session.execute(
        select(models.Cdm.event_id, models.Cdm.content).where(
            models.Cdm.event_id.in_(listed.scalar_subquery())
        )
    ).all()
    texts: dict[EventId, list[str]] = {}
    for event_id, content in rows:
        texts.setdefault(event_id, []).append(content)
    return ids, texts


def _all_observations(
    event_id: str, rows: Sequence[tuple[MessageId, Mapping[str, object] | None]]
) -> tuple[PcObservation, ...]:
    """The observations of an event's CDMs and their cached payloads, all or
    none.

    Raises
    ------
    StoreError
        If a CDM's observation is not cached, or cannot be read.
    """
    observations = complete_of(decoded(payload for _, payload in rows))
    if observations is None:
        missing = ", ".join(
            message_id for message_id, payload in rows if payload is None
        )
        raise StoreError(
            f"no Pc of {missing} in event {event_id} for these settings: "
            "refresh the store first"
        )
    return observations


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
        """Database errors as :class:`~keepout.errors.StoreError` naming the
        file, but a broken constraint (a bug, not the store's state).

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
        in one write transaction. A message whose ``MESSAGE_ID`` is stored
        with the same content (the same KVN, whatever the format read) is
        reported as already stored. Any other joins the stored event (or one
        made earlier in the ingest) of the same ``OBJECT1`` and ``OBJECT2``
        whose TCAs, with its own, spread over no more than
        :data:`~keepout.events.DEFAULT_TCA_TOLERANCE`, and whose
        ``CONJUNCTION_ID``, if both give one, is its own; if there is none,
        it starts a new event.

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
        # Each message's event, and whether the ingest stored it.
        outcomes: dict[int, tuple[EventId, bool]] = {}
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
                            f"MESSAGE_ID {message_id} is stored with other content",
                            message_id=message_id,
                        )
                    outcomes[i] = (stored.event_id, False)
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
                        ingested_at=Time.now(),
                    )
                )
                outcomes[i] = (event.id, True)
        given_order = [
            (MessageId(given.cdm.message_id), *outcomes[i])
            for i, given in enumerate(messages)
        ]
        touched = _unique(event_id for _, event_id, _ in given_order)
        return IngestReport(
            added=tuple(message_id for message_id, _, new in given_order if new),
            already_stored=tuple(
                message_id for message_id, _, new in given_order if not new
            ),
            events=touched,
            new_events=tuple(event_id for event_id in touched if event_id in created),
        )

    def _snapshot(self, catalog: CatalogReference) -> CatalogReference:
        """The catalog copied into :attr:`catalog_dir`.

        Raises
        ------
        StoreError
            If a copy cannot be written.
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
        their observations. The CDMs' texts are read 100 at a time, in short
        read transactions, and each CDM is parsed just before its Pc is
        computed, outside any transaction (in ``"auto"`` mode it can take up
        to a minute): a refresh of the whole store holds some 100 texts at a
        time, not every one. The observations are written in short
        transactions of 100 observations or 1 s of computing, whichever
        comes first, so that an interrupted refresh keeps most of what it
        computed and other processes are not held up. One that another
        process cached meanwhile, or of a CDM purged meanwhile, is not
        written.

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
            .where(~cached(key))
            .order_by(
                models.Cdm.event_id, models.Cdm.creation_date, models.Cdm.message_id
            )
        )
        with self.read() as session:
            if event_ids is not None:
                require_events(session, event_ids)
                query = query.where(models.Cdm.event_id.in_(event_ids))
            missing = session.execute(query).all()
        parsed = (cdm for _, cdm in self._parsed(missing))
        lacking = lacking_hard_body_radius(settings, parsed)
        if event_ids is not None:
            refuse_lacking_hard_body_radius(lacking)
        skipped = tuple(MessageId(cdm.message_id) for cdm in lacking)
        without_hbr = set(skipped)
        todo = [(event_id, m) for event_id, m in missing if m not in without_hbr]
        computed = (
            observation_of(event_id, cdm, settings)
            for event_id, cdm in self._parsed(todo)
        )
        written = sum(self._write(key, batch) for batch in batches(computed))
        return RefreshReport(written, skipped)

    def _parsed(
        self, rows: Sequence[tuple[EventId, MessageId]]
    ) -> Iterator[tuple[EventId, Cdm]]:
        """The stored CDMs of ``rows`` but those purged meanwhile, each parsed
        when asked for, read a batch at a time.

        Raises
        ------
        StoreError
            If the database cannot be used.
        """
        size = computing.BATCH_SIZE
        for start in range(0, len(rows), size):
            chunk = rows[start : start + size]
            with self.read() as session:
                found = session.execute(
                    select(models.Cdm.message_id, models.Cdm.content).where(
                        models.Cdm.message_id.in_([m for _, m in chunk])
                    )
                )
                texts = dict(found.all())
            for event_id, message_id in chunk:
                if message_id in texts:
                    yield event_id, parsed_cdm(texts.pop(message_id))

    def _write(self, key: str, observations: Sequence[PcObservation]) -> int:
        """Write observations under ``key`` in one write transaction (none if
        there are none), but those of CDMs not stored and those cached; how
        many it wrote."""
        payloads = [
            (observation.message_id, observation_to_json(observation))
            for observation in observations
            if observation.message_id is not None
        ]
        if not payloads:
            return 0
        with self.write() as session:
            return insert_results_in(session, key, payloads)

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
            As :meth:`refresh`; or if a Pc is not cached after it (a CDM
            ingested meanwhile) or cannot be read.
        HardBodyRadiusError
            If the settings give no hard-body radius and a CDM lacks ``HBR``.
        CdmParseError
            If a stored CDM no longer parses (a store edited by hand).
        EventError
            If the stored CDMs of an event no longer make one (likewise).
        """
        self.refresh(settings, event_ids=event_ids)
        with self.read() as session:
            ids, texts = _texts_by_event(session, event_ids)
            payloads = payloads_in(session, ids, result_key(settings))
        return [
            AssessedEvent(
                parsed_event(event_id, texts[event_id]),
                assess(
                    _all_observations(event_id, payloads[event_id]),
                    thresholds=settings.thresholds,
                ),
            )
            for event_id in ids
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
            texts = texts_in(session, cdms_of(event_id))
        if not texts:  # every stored event has a CDM
            raise StoreError(no_event(event_id))
        return parsed_event(event_id, texts)

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
        with self.read() as session:
            if event_ids is not None:
                require_events(session, event_ids)
            ids, texts = _texts_by_event(session, event_ids)
        return [parsed_event(event_id, texts[event_id]) for event_id in ids]

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
            texts = texts_in(session, cdms_of(event_id))
            if not texts:  # every stored event has a CDM
                return None
            rows = payloads_in(session, [event_id], key)[EventId(event_id)]
            jobs = active_refresh_in(session, event_id, key)
        observations = decoded(payload for _, payload in rows)
        paired = tuple(zip((m for m, _ in rows), observations, strict=True))
        return EventDetail(parsed_event(event_id, texts), paired, jobs)

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
        # Imported here: keepout.store loads no manoeuvre code until a plan
        # is read or written.
        from keepout.store.maneuver_codec import outcome_from_json

        with self.read() as session:
            slot = maneuver_slot_in(session, self.path.parent, event_id, settings)
            if slot is None:
                return None
            latest, key = slot
            payload = cached_payload_in(session, latest, key)
            texts = texts_in(session, cdms_of(event_id))
            jobs = plan_jobs_in(session, event_id, latest, key)
        event = parsed_event(event_id, texts)
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
            latest = require_latest_in(session, event_id)
            return catalog_in(session, self.path.parent, latest)

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
        from keepout.store.maneuver_codec import outcome_from_json

        with self.read() as session:
            slot = maneuver_slot_in(session, self.path.parent, event_id, settings)
            if slot is None:
                raise StoreError(no_event(event_id))
            payload = cached_payload_in(session, *slot)
            if payload is None:
                return None
            texts = texts_in(session, cdms_of(event_id))
        return outcome_from_json(payload, parsed_event(event_id, texts))

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
            catalog = catalog_in(session, self.path.parent, message_id)
            key = maneuver_key(settings, catalog)
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
        """Delete the copies in :attr:`catalog_dir` (named ``<sha256>…``)
        whose digest no catalog refers to.

        It runs under the write lock: an ingest writes or reuses its copies
        under it too, so its row keeps the copy, or it writes it again.

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
