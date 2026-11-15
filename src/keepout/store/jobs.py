"""The job queue of a store: manoeuvre plans and missing Pc computed in the
background.

A job is of a kind (:class:`~keepout.store.models.JobKind`). A caller
submits an event and its settings: a manoeuvre plan (:meth:`JobQueue.submit`)
or the Pc its CDMs lack (:meth:`JobQueue.submit_refresh`). A
worker claims the oldest queued job (:meth:`JobQueue.claim`) and holds it
with a lease it renews (:meth:`JobQueue.heartbeat`). It ends the job
(:meth:`JobQueue.finish`, :meth:`JobQueue.fail`) or gives it back
(:meth:`JobQueue.release`, :meth:`JobQueue.abandon`).

A job is pinned to its event's latest CDM at submission, so a CDM ingested
later does not change what it computes. A plan is cached on that CDM, under
:func:`~keepout.store.codec.maneuver_key`. A refresh computes the Pc of the
CDMs up to it that lack one, each cached on its CDM under
:func:`~keepout.store.codec.result_key`. Submission is idempotent. A cached
result means no job. A queued or running plan of the same CDM and key is
returned instead of a second one. A refresh is one per event and key: a
queued one is pinned again to the event's latest CDM, and a running one is
followed by one queued at most (:meth:`JobQueue.submit_refresh`).

Every write a worker makes checks that the job is still its own: same id,
``running``, and :attr:`Job.claims` unchanged. ``False`` means the job was
cancelled, purged, or given back and claimed again. The worker then drops
its result. A lapsed lease or a lost child (:meth:`JobQueue.abandon`) costs
an attempt. After :data:`MAX_ATTEMPTS` the job fails. A release on a stop
(:meth:`JobQueue.release`) costs none.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final, TypedDict, assert_never

import astropy.units as u
from astropy.time import Time
from sqlalchemy import ColumnElement, Select, exists, select, update

from keepout.errors import SettingsError, StoreError
from keepout.settings import Settings, settings_from_data, settings_to_data
from keepout.store import models
from keepout.store.codec import observation_from_json, result_key
from keepout.store.models import ACTIVE_STATUSES, JobKind, JobStatus, MessageId
from keepout.store.repo import (
    insert_results_in,
    latest_in,
    maneuver_slot_in,
    message_ids_in,
    missing_in,
    save_maneuver_in,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping
    from typing import Unpack

    from sqlalchemy.orm import Session

    from keepout._types import JsonValue
    from keepout.events import ConjunctionEvent
    from keepout.maneuver import ManeuverOutcome
    from keepout.settings import SettingsData
    from keepout.store.repo import Store

__all__ = [
    "LEASE_SECONDS",
    "MAX_ATTEMPTS",
    "Job",
    "JobKind",
    "JobQueue",
    "JobStatus",
    "PlanDetail",
    "PlanJobs",
    "Submission",
    "SubmissionOutcome",
]

LEASE_SECONDS: Final = 60.0
"""How long a claim or a heartbeat holds a job, in seconds."""

MAX_ATTEMPTS: Final = 2
"""How many times a job is taken before a worker that stops without an
answer fails it."""


def _times(count: int) -> str:
    """``count`` times in words: ``once``, ``twice``, ``3 times``."""
    return {1: "once", 2: "twice"}.get(count, f"{count} times")


_STOPPED: Final = f"the worker stopped {_times(MAX_ATTEMPTS)} without an answer"
"""The error of a job failed after :data:`MAX_ATTEMPTS`."""


@dataclass(frozen=True, slots=True, eq=False)
class Job:
    """A job of the queue, as it was when read.

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


class SubmissionOutcome(StrEnum):
    """What became of a submission (:meth:`JobQueue.submit`,
    :meth:`JobQueue.submit_refresh`)."""

    CACHED = "cached"
    """The result is cached already: no job."""
    EXISTING = "existing"
    """An identical job (same kind, CDM and key) was already queued or
    running; or a refresh of the event and key was queued, now pinned to
    the event's latest CDM."""
    QUEUED = "queued"
    """A job was queued."""


@dataclass(frozen=True, slots=True)
class Submission:
    """What a submission found or queued."""

    outcome: SubmissionOutcome
    """Whether the result was cached, an identical job existed, or a job was
    queued."""
    job: Job | None
    """The job queued, or the one already queued or running; ``None`` when
    the result is cached."""


@dataclass(frozen=True, slots=True)
class PlanJobs:
    """An event's manoeuvre jobs: the active ones, and the one of a set of
    settings (:meth:`JobQueue.plan_jobs`)."""

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


class _Columns(TypedDict, total=False):
    """The columns of a job that its transitions write."""

    status: JobStatus
    attempt: int | ColumnElement[int]
    worker: str | None
    lease_until: Time | None
    started_at: Time | None
    finished_at: Time | None
    error: str | None


def _ended(
    status: JobStatus,
    now: Time,
    *,
    error: str | None = None,
    forget_worker: bool = False,
) -> _Columns:
    """The columns of a job that ends now with ``status``.

    ``error`` is written only if given. The worker is kept, unless
    ``forget_worker``.
    """
    values = _Columns(status=status, finished_at=now, lease_until=None)
    if error is not None:
        values["error"] = error
    if forget_worker:
        values["worker"] = None
    return values


def _requeued(*, give_back_attempt: bool = False) -> _Columns:
    """The columns of a job back in the queue: no worker, lease or start.

    Its attempt stays counted, unless ``give_back_attempt``.
    """
    values = _Columns(
        status=JobStatus.QUEUED, worker=None, lease_until=None, started_at=None
    )
    if give_back_attempt:
        values["attempt"] = models.Job.attempt - 1
    return values


def _held(job: Job) -> ColumnElement[bool]:
    """The row of ``job`` while the claim that gave it is still its own:
    running, and claimed no more times since."""
    return (
        (models.Job.id == job.id)
        & (models.Job.status == JobStatus.RUNNING)
        & (models.Job.claims == job.claims)
    )


def _lapse(session: Session, which: ColumnElement[bool], now: Time) -> int:
    """Give back the running jobs of ``which`` whose worker stopped without
    an answer; how many it gave back.

    The attempt stays counted. A job is queued again, or failed at
    :data:`MAX_ATTEMPTS`. Its worker is cleared either way.
    """
    running = which & (models.Job.status == JobStatus.RUNNING)
    connection = session.connection()
    failed = connection.execute(
        update(models.Job)
        .where(running, models.Job.attempt >= MAX_ATTEMPTS)
        .values(**_ended(JobStatus.FAILED, now, error=_STOPPED, forget_worker=True))
    )
    queued = connection.execute(
        update(models.Job)
        .where(running, models.Job.attempt < MAX_ATTEMPTS)
        .values(**_requeued())
    )
    return failed.rowcount + queued.rowcount


def _lapsed(now: Time) -> ColumnElement[bool]:
    """The jobs whose lease lapsed before ``now``."""
    return models.Job.lease_until < now


def _claimable(session: Session, now: Time) -> bool:
    """Whether a claim at ``now`` has anything to do: a queued job, or a
    running one whose lease lapsed."""
    waiting = (models.Job.status == JobStatus.QUEUED) | (
        (models.Job.status == JobStatus.RUNNING) & _lapsed(now)
    )
    return bool(session.scalar(select(exists().where(waiting))))


def _claim_oldest(
    session: Session, worker: str, now: Time, lease: float
) -> models.Job | None:
    """The oldest queued job, now ``running`` for ``worker`` with one more
    attempt and claim; ``None`` if none is queued."""
    oldest = (
        select(models.Job.id)
        .where(models.Job.status == JobStatus.QUEUED)
        .order_by(models.Job.created_at, models.Job.id)
        .limit(1)
        .scalar_subquery()
    )
    return session.scalars(
        update(models.Job)
        .where(models.Job.id == oldest)
        .values(
            status=JobStatus.RUNNING,
            attempt=models.Job.attempt + 1,
            claims=models.Job.claims + 1,
            worker=worker,
            started_at=now,
            lease_until=now + lease * u.s,
        )
        .returning(models.Job)
        .execution_options(synchronize_session=False)
    ).one_or_none()


def _fail_unreadable(
    session: Session, row: models.Job, exc: SettingsError, now: Time
) -> None:
    """Fail a claimed job whose settings cannot be read: it could never be
    computed, and must not block the queue."""
    error = f"cannot read its settings: {exc}"
    session.connection().execute(
        update(models.Job)
        .where(models.Job.id == row.id)
        .values(**_ended(JobStatus.FAILED, now, error=error))
    )


def _is_cached(session: Session, message_id: str, key: str) -> bool:
    """Whether a result is cached on ``message_id`` under ``key``."""
    return bool(
        session.scalar(
            select(
                exists().where(
                    models.Result.message_id == message_id,
                    models.Result.key == key,
                )
            )
        )
    )


def _observations_in(
    session: Session, job: Job, payload: Mapping[str, JsonValue]
) -> list[tuple[str, dict[str, JsonValue]]]:
    """The observations of a refresh job's result, each a CDM's
    ``MESSAGE_ID`` and its observation's JSON, checked against the CDMs of
    its event up to its own in the caller's transaction.

    Raises
    ------
    StoreError
        If the result does not map CDMs of the event up to the job's to the
        JSON of their own observations (a bug of its computation).
    """
    observations = payload.get("observations")
    if not isinstance(observations, dict):
        raise StoreError(
            f"a refresh job's result must hold its observations, got {sorted(payload)}"
        )
    cdms = set(message_ids_in(session, job.event_id, until=job.message_id))
    found: list[tuple[str, dict[str, JsonValue]]] = []
    for message_id, data in observations.items():
        if message_id not in cdms:
            raise StoreError(
                f"a refresh job's result holds {message_id}, not a CDM of its "
                "event up to its own"
            )
        if not isinstance(data, dict):
            raise StoreError(
                f"a refresh job's observation of {message_id} must be an object"
            )
        try:
            observation = observation_from_json(data)
        except StoreError as exc:
            raise StoreError(
                f"a refresh job's observation of {message_id}: {exc}"
            ) from exc
        if observation.message_id != message_id:
            raise StoreError(
                f"a refresh job's observation of {message_id} is of "
                f"{observation.message_id}"
            )
        found.append((message_id, data))
    return found


def _jobs_of(kind: JobKind, message_id: str, key: str) -> Select[models.Job]:
    """The jobs of ``kind``, ``message_id`` and ``key``."""
    return select(models.Job).where(
        models.Job.kind == kind,
        models.Job.message_id == message_id,
        models.Job.key == key,
    )


def _active_job(
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


_NEWEST_FIRST: Final = (models.Job.created_at.desc(), models.Job.id.desc())
"""The order of jobs listed newest first."""


def _newest_job(
    session: Session, kind: JobKind, message_id: str, key: str
) -> Job | None:
    """The newest job of ``kind``, ``message_id`` and ``key``, whatever its
    status."""
    row = session.scalars(
        _jobs_of(kind, message_id, key).order_by(*_NEWEST_FIRST).limit(1)
    ).one_or_none()
    return None if row is None else Job.of(row)


def _queued(
    session: Session,
    *,
    kind: JobKind,
    event_id: str,
    message_id: str,
    key: str,
    params: SettingsData,
    now: Time,
) -> Submission:
    """The submission of a job: the active one of the same kind, CDM and key
    if there is one, else a job queued now."""
    found = _active_job(session, kind, message_id, key)
    if found is not None:
        return Submission(SubmissionOutcome.EXISTING, found)
    return _queue_new(
        session,
        kind=kind,
        event_id=event_id,
        message_id=message_id,
        key=key,
        params=params,
        now=now,
    )


def _queue_new(
    session: Session,
    *,
    kind: JobKind,
    event_id: str,
    message_id: str,
    key: str,
    params: SettingsData,
    now: Time,
) -> Submission:
    """The submission of a job queued now."""
    row = models.Job(
        kind=kind,
        event_id=event_id,
        message_id=message_id,
        key=key,
        params=params,
        status=JobStatus.QUEUED,
        attempt=0,
        claims=0,
        created_at=now,
    )
    session.add(row)
    session.flush()
    session.refresh(row)  # the times as stored, as job() reads them
    return Submission(SubmissionOutcome.QUEUED, Job.of(row))


def _refresh_queued(
    session: Session,
    *,
    event_id: str,
    message_id: str,
    key: str,
    params: SettingsData,
    now: Time,
) -> Submission:
    """The submission of a refresh of an event pinned to its latest CDM
    ``message_id``: the queued one of the event and key, pinned again to
    ``message_id``; else the running one if pinned to it; else a refresh
    queued now."""
    rows = session.scalars(
        select(models.Job)
        .where(
            models.Job.kind == JobKind.REFRESH,
            models.Job.event_id == event_id,
            models.Job.key == key,
            models.Job.status.in_(ACTIVE_STATUSES),
        )
        .order_by(models.Job.id)
    ).all()
    for row in rows:
        if row.status is JobStatus.QUEUED:
            # No worker holds it: it computes up to the latest CDM instead.
            row.message_id = MessageId(message_id)
            session.flush()
            return Submission(SubmissionOutcome.EXISTING, Job.of(row))
    for row in rows:
        if row.message_id == message_id:
            return Submission(SubmissionOutcome.EXISTING, Job.of(row))
    return _queue_new(
        session,
        kind=JobKind.REFRESH,
        event_id=event_id,
        message_id=message_id,
        key=key,
        params=params,
        now=now,
    )


def plan_jobs_in(
    session: Session, event_id: str, message_id: str, key: str
) -> PlanJobs:
    """An event's manoeuvre jobs, in the caller's transaction: those still
    queued or running, and the one of ``message_id`` and ``key``.

    Internal to :mod:`keepout.store`: the queue and the store's planner
    read (:meth:`~keepout.store.Store.plan_detail`) read the same jobs.

    Parameters
    ----------
    session
        A session of the store.
    event_id
        The store's id of the event.
    message_id
        The event's latest CDM, where its plan is cached.
    key
        The plan's :func:`~keepout.store.codec.maneuver_key`.

    Returns
    -------
    PlanJobs
        The jobs.
    """
    maneuver = JobKind.MANEUVER
    current = _active_job(session, maneuver, message_id, key)
    if current is None:
        current = _newest_job(session, maneuver, message_id, key)
    rows = session.scalars(
        select(models.Job)
        .where(
            models.Job.event_id == event_id,
            models.Job.status.in_(ACTIVE_STATUSES),
            models.Job.kind == maneuver,
        )
        .order_by(*_NEWEST_FIRST)
    )
    return PlanJobs(tuple(Job.of(row) for row in rows), current)


class JobQueue:
    """The job queue of a store.

    Parameters
    ----------
    store
        The store.
    now
        The clock of the jobs' times and leases.
    """

    def __init__(self, store: Store, *, now: Callable[[], Time] = Time.now) -> None:
        self.store = store
        self._now = now

    def submit(self, event_id: str, settings: Settings) -> Submission:
        """Queue a manoeuvre plan of a stored event, unless its result is
        cached or an identical job is queued or running.

        One write transaction: the event's latest CDM, its catalog, the
        cache and the queue are read where the job is written.

        Parameters
        ----------
        event_id
            The store's id of the event.
        settings
            The operator's settings of the plan.

        Returns
        -------
        Submission
            What became of the submission, and its job if there is one.

        Raises
        ------
        StoreError
            If there is no such event, or if the database cannot be used.
        ValueError
            If a settings file cannot hold ``settings``, as
            :func:`~keepout.settings.settings_to_data`.
        """
        params = settings_to_data(settings)
        now = self._now()
        with self.store.write() as session:
            message_id, key = maneuver_slot_in(
                session, self.store.path.parent, event_id, settings
            )
            if _is_cached(session, message_id, key):
                return Submission(SubmissionOutcome.CACHED, None)
            return _queued(
                session,
                kind=JobKind.MANEUVER,
                event_id=event_id,
                message_id=message_id,
                key=key,
                params=params,
                now=now,
            )

    def submit_refresh(self, event_id: str, settings: Settings) -> Submission:
        """Queue the computation of the Pc a stored event lacks for
        ``settings``, unless none lacks or a refresh of the event and
        settings is queued, or running on the event's latest CDM.

        The job is pinned to the event's latest CDM: it computes the Pc of
        that CDM and the older ones that lack one when a worker runs it, not
        of those ingested later. A refresh is one per event and key, as long
        as one is queued: a CDM ingested meanwhile pins the queued one again
        to the new latest CDM (it is ``existing``), so two workers never
        take two refreshes that compute the same Pc. A running refresh keeps
        its CDM; a CDM ingested meanwhile queues one refresh after it, the
        only queued one. One write transaction, as :meth:`submit`.

        Parameters
        ----------
        event_id
            The store's id of the event.
        settings
            How Pc is computed: the observations are cached under their
            :func:`~keepout.store.codec.result_key`.

        Returns
        -------
        Submission
            What became of the submission, and its job if there is one:
            ``cached`` when every CDM up to the latest has its Pc;
            ``existing`` with the queued refresh, pinned to the latest CDM,
            or the running one already pinned to it.

        Raises
        ------
        StoreError
            If there is no such event, or if the database cannot be used.
        ValueError
            If a settings file cannot hold ``settings``, as
            :func:`~keepout.settings.settings_to_data`.
        """
        params = settings_to_data(settings)
        key = result_key(settings)
        now = self._now()
        with self.store.write() as session:
            message_id = latest_in(session, event_id)
            if not missing_in(session, event_id, key, until=message_id):
                return Submission(SubmissionOutcome.CACHED, None)
            return _refresh_queued(
                session,
                event_id=event_id,
                message_id=message_id,
                key=key,
                params=params,
                now=now,
            )

    def claim(self, worker: str, *, lease: float = LEASE_SECONDS) -> Job | None:
        """Take the oldest queued job, after giving back those whose lease
        lapsed.

        It reads first, and takes the write lock only if a job is queued or
        a lease lapsed. The claim is one write transaction, so a job is
        claimed by one worker only.

        Parameters
        ----------
        worker
            The name of the worker taking it.
        lease
            How long it holds the job without a heartbeat, in seconds.

        Returns
        -------
        Job or None
            The job, ``running`` with one more attempt; ``None`` if none is
            queued. A queued job whose settings cannot be read (a store
            written by hand) is failed instead, and the next one taken.

        Raises
        ------
        StoreError
            If the database cannot be used.
        """
        now = self._now()
        # An idle worker calls this every few seconds: read first, and take
        # the write lock only if there is something to write.
        with self.store.read() as session:
            if not _claimable(session, now):
                return None
        with self.store.write() as session:
            _lapse(session, _lapsed(now), now)
            while (row := _claim_oldest(session, worker, now, lease)) is not None:
                try:
                    return Job.of(row)
                except SettingsError as exc:
                    _fail_unreadable(session, row, exc, now)
            return None

    def _update_if_held(self, job: Job, **values: Unpack[_Columns]) -> bool:
        """Write ``values`` to ``job`` if it is still held as claimed, in one
        write transaction; whether it was."""
        with self.store.write() as session:
            return self._update_if_held_in(session, job, **values)

    @staticmethod
    def _update_if_held_in(
        session: Session, job: Job, **values: Unpack[_Columns]
    ) -> bool:
        """Write ``values`` to ``job`` if it is still held as claimed;
        whether it was."""
        changed = session.connection().execute(
            update(models.Job).where(_held(job)).values(**values)
        )
        return changed.rowcount == 1

    def heartbeat(self, job: Job, *, lease: float = LEASE_SECONDS) -> bool:
        """Renew the lease of a job a worker holds.

        Parameters
        ----------
        job
            The job, as claimed.
        lease
            How long it holds the job from now, in seconds.

        Returns
        -------
        bool
            Whether the job is still the worker's: not if it was cancelled,
            taken by another worker or purged.

        Raises
        ------
        StoreError
            If the database cannot be used.
        """
        return self._update_if_held(job, lease_until=self._now() + lease * u.s)

    def finish(self, job: Job, payload: Mapping[str, JsonValue]) -> bool:
        """End a job a worker holds as succeeded and cache its result, in one
        write transaction.

        A plan is cached on the job's CDM; a refresh caches each observation
        on its CDM. A result already cached under its key (e.g. by a
        synchronous run meanwhile) is kept.

        The result of a refresh is decoded in the same transaction: it must
        map CDMs of the event up to the job's to their own observations.

        Parameters
        ----------
        job
            The job, as claimed.
        payload
            Its result: of a plan, the outcome's
            :func:`~keepout.store.maneuver_codec.outcome_to_json`; of a
            refresh, ``{"observations": {MESSAGE_ID: observation}}``, each
            observation's :func:`~keepout.store.codec.observation_to_json`.

        Returns
        -------
        bool
            Whether the job was still the worker's (as :meth:`heartbeat`);
            if not, nothing is written.

        Raises
        ------
        StoreError
            If the result of a refresh cannot be cached (nothing is written:
            fail the job, :meth:`fail`), or if the database cannot be used.
        """
        now = self._now()
        with self.store.write() as session:
            if not self._update_if_held_in(
                session, job, **_ended(JobStatus.SUCCEEDED, now)
            ):
                return False
            match job.kind:
                case JobKind.REFRESH:
                    observations = _observations_in(session, job, payload)
                    insert_results_in(session, job.key, observations)
                case JobKind.MANEUVER:
                    save_maneuver_in(session, job.message_id, job.key, payload)
                case _:
                    assert_never(job.kind)
            return True

    def fail(self, job: Job, message: str) -> bool:
        """End a job a worker holds as failed.

        Parameters
        ----------
        job
            The job, as claimed.
        message
            Why it failed.

        Returns
        -------
        bool
            Whether the job was still the worker's (as :meth:`heartbeat`).

        Raises
        ------
        StoreError
            If the database cannot be used.
        """
        return self._update_if_held(
            job, **_ended(JobStatus.FAILED, self._now(), error=message)
        )

    def release(self, job: Job) -> bool:
        """Give a job a worker holds back to the queue without counting the
        attempt, e.g. when the worker is stopped.

        Parameters
        ----------
        job
            The job, as claimed.

        Returns
        -------
        bool
            Whether the job was still the worker's (as :meth:`heartbeat`).

        Raises
        ------
        StoreError
            If the database cannot be used.
        """
        return self._update_if_held(job, **_requeued(give_back_attempt=True))

    def abandon(self, job: Job) -> bool:
        """Give back a job a worker holds whose computation stopped without
        an answer, as a lapsed lease does: queued again with the attempt
        counted, or failed at :data:`MAX_ATTEMPTS`.

        Parameters
        ----------
        job
            The job, as claimed.

        Returns
        -------
        bool
            Whether the job was still the worker's (as :meth:`heartbeat`).

        Raises
        ------
        StoreError
            If the database cannot be used.
        """
        now = self._now()
        with self.store.write() as session:
            return _lapse(session, _held(job), now) == 1

    def cancel(self, job_id: int) -> Job:
        """Cancel a queued or running job; its worker finds it no longer
        its own at its next heartbeat.

        Parameters
        ----------
        job_id
            The job's id.

        Returns
        -------
        Job
            The job, ``cancelled``.

        Raises
        ------
        StoreError
            If there is no such job, if it has already ended, or if the
            database cannot be used.
        """
        now = self._now()
        with self.store.write() as session:
            row = session.scalars(
                update(models.Job)
                .where(models.Job.id == job_id, models.Job.status.in_(ACTIVE_STATUSES))
                .values(**_ended(JobStatus.CANCELLED, now))
                .returning(models.Job)
                .execution_options(synchronize_session=False)
            ).one_or_none()
            if row is not None:
                return Job.of(row)
            status = session.scalar(
                select(models.Job.status).where(models.Job.id == job_id)
            )
        if status is None:
            raise StoreError(f"no job {job_id} in the store")
        raise StoreError(f"job {job_id} has already ended: {status}")

    def jobs(self, *, status: Collection[JobStatus] | None = None) -> list[Job]:
        """The jobs, newest first.

        Parameters
        ----------
        status
            Only the jobs of these statuses; ``None``: every job.

        Returns
        -------
        list of Job
            The jobs, by ``created_at`` then id, newest first.

        Raises
        ------
        StoreError
            If the database cannot be used.
        """
        query = select(models.Job).order_by(*_NEWEST_FIRST)
        if status is not None:
            query = query.where(models.Job.status.in_(list(status)))
        with self.store.read() as session:
            return [Job.of(row) for row in session.scalars(query)]

    def plan_jobs(self, event_id: str, settings: Settings) -> PlanJobs | None:
        """An event's manoeuvre jobs, read in one transaction: those still
        queued or running, and the one of ``settings`` on the event's latest
        CDM (as :meth:`submit` would find or queue it).

        Parameters
        ----------
        event_id
            The store's id of the event.
        settings
            The operator's settings of the plan (by their
            :func:`~keepout.store.codec.maneuver_key`).

        Returns
        -------
        PlanJobs or None
            The jobs; ``None`` if there is no such event.

        Raises
        ------
        StoreError
            If the database cannot be used.
        ValueError
            If a settings file cannot hold ``settings``, as
            :func:`~keepout.store.codec.maneuver_key`.
        """
        with self.store.read() as session:
            if session.get(models.Event, event_id) is None:
                return None
            message_id, key = maneuver_slot_in(
                session, self.store.path.parent, event_id, settings
            )
            return plan_jobs_in(session, event_id, message_id, key)

    def job(self, job_id: int) -> Job:
        """A job, by id.

        Parameters
        ----------
        job_id
            The job's id.

        Returns
        -------
        Job
            The job, as it is now.

        Raises
        ------
        StoreError
            If there is no such job, or if the database cannot be used.
        """
        with self.store.read() as session:
            row = session.get(models.Job, job_id)
            if row is None:
                raise StoreError(f"no job {job_id} in the store")
            return Job.of(row)

    def jobs_by_id(self, job_ids: Collection[int]) -> list[Job]:
        """Jobs, by id, read in one transaction.

        Parameters
        ----------
        job_ids
            The jobs' ids.

        Returns
        -------
        list of Job
            The jobs, as they are now, in the order of ``job_ids``.

        Raises
        ------
        StoreError
            If a job is not in the store (the first such one is named), or if
            the database cannot be used.
        """
        with self.store.read() as session:
            rows = session.scalars(
                select(models.Job).where(models.Job.id.in_(set(job_ids)))
            )
            found = {row.id: Job.of(row) for row in rows}
        for job_id in job_ids:
            if job_id not in found:
                raise StoreError(f"no job {job_id} in the store")
        return [found[job_id] for job_id in job_ids]
