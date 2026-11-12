"""The job queue of a store: manoeuvre plans computed in the background.

A caller submits an event and its settings (:meth:`JobQueue.submit`); a
worker claims the oldest queued job (:meth:`JobQueue.claim`), holds it with a
lease it renews (:meth:`JobQueue.heartbeat`) and ends it
(:meth:`JobQueue.finish`, :meth:`JobQueue.fail`) or gives it back
(:meth:`JobQueue.release`, :meth:`JobQueue.abandon`).

A job is pinned to its event's latest CDM at submission and its result is
cached there, under :func:`~keepout.store.maneuver_codec.maneuver_key`: a CDM
ingested later does not change what it computes. Submission is idempotent: a
cached result means no job, and a queued or running job of the same CDM and
key is returned instead of a second one.

A worker's every write about its job holds only if the job is still held
by the claim it made (same id, ``running``, same :attr:`Job.claims`, a count
of claims that never goes down): ``False`` means it was cancelled, released
or given back and claimed again, or purged with its event, and the worker
drops what it computed. A lapsed lease, or a worker that stops without an
answer (:meth:`JobQueue.abandon`), costs an attempt: the job is queued
again, or failed after :data:`MAX_ATTEMPTS`. A release
(:meth:`JobQueue.release`) gives the attempt back, but never the claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import astropy.units as u
from astropy.time import Time
from sqlalchemy import exists, select, update
from sqlalchemy.orm import Session

from keepout.errors import SettingsError, StoreError
from keepout.settings import Settings, settings_from_data, settings_to_data
from keepout.store import models
from keepout.store.maneuver_codec import MANEUVER_KIND, maneuver_key
from keepout.store.models import JobStatus
from keepout.store.repo import catalog_in, latest_in, save_maneuver_in

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping

    from sqlalchemy import ColumnElement

    from keepout._types import JsonValue
    from keepout.store.repo import Store

__all__ = [
    "LEASE_SECONDS",
    "MAX_ATTEMPTS",
    "Job",
    "JobQueue",
    "JobStatus",
    "Submission",
]

LEASE_SECONDS: Final = 60.0
"""How long a claim or a heartbeat holds a job, in seconds."""

MAX_ATTEMPTS: Final = 2
"""How many times a job is taken before a worker that stops without an
answer fails it."""

_STOPPED: Final = "the worker stopped twice without an answer"
"""The error of a job failed after :data:`MAX_ATTEMPTS`."""

_ACTIVE: Final = (JobStatus.QUEUED, JobStatus.RUNNING)
"""The statuses of a job that has not ended."""


@dataclass(frozen=True, slots=True, eq=False)
class Job:
    """A job of the queue, as it was when read.

    Two jobs compare by identity (:class:`~keepout.settings.Settings` do):
    compare :attr:`id` and :attr:`claims`.
    """

    id: int
    kind: str
    """What is computed: ``"maneuver"``."""
    event_id: str
    message_id: str
    """The CDM the computation is pinned to: the event's latest when
    queued."""
    key: str
    """The key its result is cached under."""
    settings: Settings
    status: JobStatus
    attempt: int
    """How many tries it cost: a claim counts one, a release gives it back."""
    claims: int
    """How many times it was claimed, never lowered: a worker's writes name
    the claim they belong to."""
    worker: str | None
    """The worker that holds it, or that finished or failed it."""
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
            kind=row.kind,
            event_id=row.event_id,
            message_id=row.message_id,
            key=row.key,
            settings=settings_from_data(row.params),
            status=JobStatus(row.status),
            attempt=row.attempt,
            claims=row.claims,
            worker=row.worker,
            created_at=row.created_at,
            started_at=row.started_at,
            finished_at=row.finished_at,
            error=row.error,
        )


@dataclass(frozen=True, slots=True)
class Submission:
    """What a submission found or queued."""

    job: Job | None
    """The job queued, or the one already queued or running; ``None`` when
    :attr:`cached`."""
    cached: bool
    """The result is cached already: no job."""
    existing: bool
    """An identical job (same CDM and key) was already queued or running."""


def _held(job: Job) -> ColumnElement[bool]:
    """The row of ``job`` while the claim that gave it is still its own:
    running, and claimed no more times since."""
    return (
        (models.Job.id == job.id)
        & (models.Job.status == JobStatus.RUNNING)
        & (models.Job.claims == job.claims)
    )


def _lapse(session: Session, which: ColumnElement[bool], now: Time) -> int:
    """Give back running jobs whose worker stopped without an answer: queued
    again with the attempt counted, or failed at :data:`MAX_ATTEMPTS`;
    worker and lease cleared. How many it gave back."""
    running = which & (models.Job.status == JobStatus.RUNNING)
    connection = session.connection()
    failed = connection.execute(
        update(models.Job)
        .where(running, models.Job.attempt >= MAX_ATTEMPTS)
        .values(
            status=JobStatus.FAILED,
            error=_STOPPED,
            finished_at=now,
            worker=None,
            lease_until=None,
        )
    )
    queued = connection.execute(
        update(models.Job)
        .where(running, models.Job.attempt < MAX_ATTEMPTS)
        .values(status=JobStatus.QUEUED, worker=None, lease_until=None, started_at=None)
    )
    return failed.rowcount + queued.rowcount


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
            message_id = latest_in(session, event_id)
            catalog = catalog_in(session, self.store.path.parent, message_id)
            key = maneuver_key(settings, catalog)
            cached = session.scalar(
                select(
                    exists().where(
                        models.Result.message_id == message_id,
                        models.Result.key == key,
                    )
                )
            )
            if cached:
                return Submission(None, cached=True, existing=False)
            found = session.scalars(
                select(models.Job)
                .where(
                    models.Job.kind == MANEUVER_KIND,
                    models.Job.message_id == message_id,
                    models.Job.key == key,
                    models.Job.status.in_(_ACTIVE),
                )
                .order_by(models.Job.id)
                .limit(1)
            ).one_or_none()
            if found is not None:
                return Submission(Job.of(found), cached=False, existing=True)
            row = models.Job(
                kind=MANEUVER_KIND,
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
            return Submission(Job.of(row), cached=False, existing=False)

    def claim(self, worker: str, *, lease: float = LEASE_SECONDS) -> Job | None:
        """Take the oldest queued job, after giving back those whose lease
        lapsed.

        One write transaction: a job is claimed by one worker only.

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
        with self.store.write() as session:
            _lapse(session, models.Job.lease_until < now, now)
            oldest = (
                select(models.Job.id)
                .where(models.Job.status == JobStatus.QUEUED)
                .order_by(models.Job.created_at, models.Job.id)
                .limit(1)
                .scalar_subquery()
            )
            while True:
                row = session.scalars(
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
                if row is None:
                    return None
                try:
                    return Job.of(row)
                except SettingsError as exc:
                    # Never computable: failed, not left to block the queue.
                    session.connection().execute(
                        update(models.Job)
                        .where(models.Job.id == row.id)
                        .values(
                            status=JobStatus.FAILED,
                            error=f"cannot read its settings: {exc}",
                            finished_at=now,
                            lease_until=None,
                        )
                    )

    def _change(self, job: Job, **values: object) -> bool:
        """Write ``values`` to ``job`` if it is still held as claimed, in one
        write transaction; whether it was."""
        with self.store.write() as session:
            return self._change_in(session, job, **values)

    @staticmethod
    def _change_in(session: Session, job: Job, **values: object) -> bool:
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
        return self._change(job, lease_until=self._now() + lease * u.s)

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
            held = self._change_in(
                session,
                job,
                status=JobStatus.SUCCEEDED,
                finished_at=now,
                lease_until=None,
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
        return self._change(
            job,
            status=JobStatus.FAILED,
            error=message,
            finished_at=self._now(),
            lease_until=None,
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
        return self._change(
            job,
            status=JobStatus.QUEUED,
            attempt=models.Job.attempt - 1,
            worker=None,
            lease_until=None,
            started_at=None,
        )

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
                .values(status=JobStatus.CANCELLED, finished_at=now, lease_until=None)
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
