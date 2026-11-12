"""The job queue of a store: manoeuvre plans computed in the background.

A caller submits an event and its settings (:meth:`JobQueue.submit`). A
worker claims the oldest queued job (:meth:`JobQueue.claim`) and holds it
with a lease it renews (:meth:`JobQueue.heartbeat`). It ends the job
(:meth:`JobQueue.finish`, :meth:`JobQueue.fail`) or gives it back
(:meth:`JobQueue.release`, :meth:`JobQueue.abandon`).

A job is pinned to its event's latest CDM at submission. Its result is
cached there, under :func:`~keepout.store.codec.maneuver_key`, so a CDM
ingested later does not change what it computes. Submission is idempotent.
A cached result means no job. A queued or running job of the same CDM and
key is returned instead of a second one.

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
from typing import TYPE_CHECKING, Final, TypedDict

import astropy.units as u
from astropy.time import Time
from sqlalchemy import ColumnElement, exists, select, update

from keepout.errors import SettingsError, StoreError
from keepout.settings import Settings, settings_from_data, settings_to_data
from keepout.store import models
from keepout.store.models import JobStatus
from keepout.store.repo import maneuver_slot_in, save_maneuver_in

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping
    from typing import Unpack

    from sqlalchemy.orm import Session

    from keepout._types import JsonValue
    from keepout.store.repo import Store

__all__ = [
    "LEASE_SECONDS",
    "MAX_ATTEMPTS",
    "Job",
    "JobQueue",
    "JobStatus",
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

_ACTIVE: Final = tuple(status for status in JobStatus if status.is_active)
"""The statuses of a job that has not ended."""


@dataclass(frozen=True, slots=True, eq=False)
class Job:
    """A job of the queue, as it was when read.

    Two jobs compare by identity (:class:`~keepout.settings.Settings` do):
    compare :attr:`id` and :attr:`claims`.
    """

    id: int
    event_id: str
    message_id: str
    """The CDM the computation is pinned to: the event's latest when
    queued."""
    key: str
    """The key its result is cached under."""
    settings: Settings
    status: JobStatus
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
    started_at: Time | None
    finished_at: Time | None
    error: str | None
    """Why it failed."""

    @classmethod
    def of(cls, row: models.Job) -> Job:
        """The job of a row."""
        return cls(
            id=row.id,
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
    """What became of a submission (:meth:`JobQueue.submit`)."""

    CACHED = "cached"
    """The result is cached already: no job."""
    EXISTING = "existing"
    """An identical job (same CDM and key) was already queued or running."""
    QUEUED = "queued"
    """A job was queued."""


@dataclass(frozen=True, slots=True)
class Submission:
    """What a submission found or queued."""

    outcome: SubmissionOutcome
    job: Job | None
    """The job queued, or the one already queued or running; ``None`` when
    the result is cached."""


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


def _active_job(session: Session, message_id: str, key: str) -> Job | None:
    """The oldest queued or running job of ``message_id`` and ``key``."""
    row = session.scalars(
        select(models.Job)
        .where(
            models.Job.message_id == message_id,
            models.Job.key == key,
            models.Job.status.in_(_ACTIVE),
        )
        .order_by(models.Job.id)
        .limit(1)
    ).one_or_none()
    return None if row is None else Job.of(row)


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
            found = _active_job(session, message_id, key)
            if found is not None:
                return Submission(SubmissionOutcome.EXISTING, found)
            row = models.Job(
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
        """End a job a worker holds as succeeded and cache its result on the
        job's CDM, in one write transaction.

        A result already cached under its key (e.g. by a synchronous run) is
        kept.

        Parameters
        ----------
        job
            The job, as claimed.
        payload
            Its result: the outcome's
            :func:`~keepout.store.maneuver_codec.outcome_to_json`.

        Returns
        -------
        bool
            Whether the job was still the worker's (as :meth:`heartbeat`);
            if not, nothing is written.

        Raises
        ------
        StoreError
            If the database cannot be used.
        """
        now = self._now()
        with self.store.write() as session:
            held = self._update_if_held_in(
                session, job, **_ended(JobStatus.SUCCEEDED, now)
            )
            if held:
                save_maneuver_in(session, job.message_id, job.key, payload)
            return held

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
                .where(models.Job.id == job_id, models.Job.status.in_(_ACTIVE))
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
        query = select(models.Job).order_by(
            models.Job.created_at.desc(), models.Job.id.desc()
        )
        if status is not None:
            query = query.where(models.Job.status.in_(list(status)))
        with self.store.read() as session:
            return [Job.of(row) for row in session.scalars(query)]

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
