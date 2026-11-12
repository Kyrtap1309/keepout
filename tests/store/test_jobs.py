"""The job queue: submission, claim, lease, attempts and cancel."""

import multiprocessing
import threading
import time

import astropy.units as u
import pytest
from astropy.time import Time
from sqlalchemy import select, update
from store_builders import FOSTER, count_rows, message

from keepout.errors import StoreError
from keepout.pc import PcComputation
from keepout.settings import Settings, settings_to_data
from keepout.store import Store, models, open_store
from keepout.store.codec import maneuver_key
from keepout.store.jobs import (
    LEASE_SECONDS,
    MAX_ATTEMPTS,
    JobQueue,
    JobStatus,
    SubmissionOutcome,
)
from keepout.store.repo import save_maneuver_in

PAYLOAD = {"ok": True}


class Clock:
    """A clock that moves only when told."""

    def __init__(self) -> None:
        self.time = Time("2026-11-12T12:00:00", scale="utc")

    def __call__(self) -> Time:
        return self.time

    def advance(self, seconds: float) -> None:
        self.time = self.time + seconds * u.s


@pytest.fixture
def event_id(store: Store) -> str:
    (stored,) = store.ingest(
        [message("A"), message("B", tca=1.0, created=-60.0)]
    ).events
    return stored


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def queue(store: Store, clock: Clock) -> JobQueue:
    return JobQueue(store, now=clock)


def test_submit_then_claim(queue: JobQueue, event_id: str, clock: Clock) -> None:
    submission = queue.submit(event_id, FOSTER)
    assert submission.outcome is SubmissionOutcome.QUEUED
    assert submission.job is not None
    assert submission.job.status is JobStatus.QUEUED
    assert submission.job.attempt == 0
    assert submission.job.claims == 0
    assert submission.job.worker is None
    assert submission.job.created_at.isot == clock.time.isot
    stored = queue.job(submission.job.id)
    assert submission.job.created_at.isot == stored.created_at.isot
    assert submission.job.created_at.scale == stored.created_at.scale == "utc"
    assert submission.job.key == maneuver_key(FOSTER, None)
    assert settings_to_data(submission.job.settings) == settings_to_data(FOSTER)
    job = queue.claim("w1")
    assert job is not None
    assert job.id == submission.job.id
    assert job.status is JobStatus.RUNNING
    assert job.attempt == 1
    assert job.claims == 1
    assert job.worker == "w1"
    assert job.started_at is not None
    assert job.started_at.isot == clock.time.isot
    assert settings_to_data(job.settings) == settings_to_data(FOSTER)
    assert queue.claim("w2") is None


def test_claim_on_an_empty_queue(queue: JobQueue, event_id: str) -> None:
    assert queue.claim("w1") is None


def test_an_idle_claim_does_not_take_the_write_lock(
    queue: JobQueue, store: Store, event_id: str
) -> None:
    # A claim that would write waits for this writer (BUSY_TIMEOUT_MS).
    with store.write():
        started = time.monotonic()
        assert queue.claim("w1") is None
        assert time.monotonic() - started < 1.0


def test_the_oldest_job_is_claimed_first(
    queue: JobQueue, event_id: str, clock: Clock
) -> None:
    first = queue.submit(event_id, FOSTER).job
    clock.advance(1.0)
    second = queue.submit(event_id, Settings()).job
    assert first is not None
    assert second is not None
    claimed = [queue.claim("w1"), queue.claim("w1")]
    assert [job.id for job in claimed if job is not None] == [first.id, second.id]


def test_a_job_that_cannot_be_read_is_failed_and_skipped(
    queue: JobQueue, store: Store, event_id: str, clock: Clock
) -> None:
    bad = queue.submit(event_id, FOSTER).job
    clock.advance(1.0)
    good = queue.submit(event_id, Settings()).job
    assert bad is not None
    assert good is not None
    with store.write() as session:
        session.execute(
            update(models.Job)
            .where(models.Job.id == bad.id)
            .values(params={"pc": {"method": "nope"}})
        )
    claimed = queue.claim("w1")
    assert claimed is not None
    assert claimed.id == good.id
    with store.read() as session:
        row = session.get(models.Job, bad.id)
        assert row is not None
        assert row.status == JobStatus.FAILED
        assert row.error is not None
        assert row.error.startswith("cannot read its settings: ")
        assert "pc.method" in row.error
        assert row.finished_at is not None
        assert row.finished_at.isot == clock.time.isot
        assert row.lease_until is None
    assert queue.claim("w1") is None


def test_submit_is_idempotent(queue: JobQueue, store: Store, event_id: str) -> None:
    first = queue.submit(event_id, FOSTER)
    second = queue.submit(event_id, FOSTER)
    assert second.outcome is SubmissionOutcome.EXISTING
    assert first.job is not None
    assert second.job is not None
    assert second.job.id == first.job.id
    job = queue.claim("w1")
    assert job is not None
    running = queue.submit(event_id, FOSTER)
    assert running.outcome is SubmissionOutcome.EXISTING
    assert running.job is not None
    assert running.job.id == job.id
    assert running.job.status is JobStatus.RUNNING
    assert queue.finish(job, PAYLOAD)
    third = queue.submit(event_id, FOSTER)
    assert third.outcome is SubmissionOutcome.CACHED
    assert third.job is None
    assert len(queue.jobs()) == 1


def test_a_failed_job_is_submitted_again(queue: JobQueue, event_id: str) -> None:
    queue.submit(event_id, FOSTER)
    job = queue.claim("w1")
    assert job is not None
    assert queue.fail(job, "no burn")
    again = queue.submit(event_id, FOSTER)
    assert again.outcome is SubmissionOutcome.QUEUED
    assert again.job is not None
    assert again.job.id != job.id


def test_submit_pins_the_latest_cdm(
    queue: JobQueue, store: Store, event_id: str
) -> None:
    latest = store.event(event_id).latest.message_id
    submission = queue.submit(event_id, FOSTER)
    assert submission.job is not None
    assert submission.job.message_id == latest == "B"
    assert submission.job.event_id == event_id
    store.ingest([message("C", tca=2.0, created=-30.0)])
    assert store.event(event_id).latest.message_id == "C"
    assert queue.job(submission.job.id).message_id == "B"


def test_submit_an_unknown_event(queue: JobQueue, event_id: str) -> None:
    with pytest.raises(StoreError, match="no event 12345678 in the store"):
        queue.submit("12345678", FOSTER)
    assert queue.jobs() == []


def test_submit_settings_a_file_cannot_hold(queue: JobQueue, event_id: str) -> None:
    settings = Settings(compute=PcComputation("montecarlo", options={"seed": 1}))
    with pytest.raises(ValueError, match="options"):
        queue.submit(event_id, settings)
    assert queue.jobs() == []


def test_expired_lease_requeues_then_fails(
    queue: JobQueue, event_id: str, clock: Clock
) -> None:
    queue.submit(event_id, FOSTER)
    first = queue.claim("w1")
    assert first is not None
    clock.advance(LEASE_SECONDS - 1.0)
    assert queue.claim("w2") is None
    clock.advance(2.0)
    second = queue.claim("w2")
    assert second is not None
    assert second.id == first.id
    assert second.attempt == 2
    assert second.worker == "w2"
    clock.advance(LEASE_SECONDS + 1.0)
    assert queue.claim("w3") is None
    failed = queue.job(first.id)
    assert failed.status is JobStatus.FAILED
    assert failed.error == "the worker stopped twice without an answer"
    assert failed.worker is None
    assert failed.finished_at is not None
    assert failed.attempt == MAX_ATTEMPTS


def test_abandon_requeues_then_fails(queue: JobQueue, event_id: str) -> None:
    queue.submit(event_id, FOSTER)
    first = queue.claim("w1")
    assert first is not None
    assert queue.abandon(first)
    requeued = queue.job(first.id)
    assert requeued.status is JobStatus.QUEUED
    assert requeued.attempt == 1
    assert requeued.worker is None
    assert not queue.abandon(first)
    second = queue.claim("w1")
    assert second is not None
    assert second.attempt == 2
    assert queue.abandon(second)
    failed = queue.job(first.id)
    assert failed.status is JobStatus.FAILED
    assert failed.error == "the worker stopped twice without an answer"
    assert failed.worker is None
    assert not queue.abandon(second)


def test_heartbeat_extends_the_lease(
    queue: JobQueue, event_id: str, clock: Clock
) -> None:
    queue.submit(event_id, FOSTER)
    job = queue.claim("w1")
    assert job is not None
    clock.advance(LEASE_SECONDS - 10.0)
    assert queue.heartbeat(job)
    clock.advance(LEASE_SECONDS - 10.0)  # past the first lease, not the second
    assert queue.claim("w2") is None
    held = queue.job(job.id)
    assert held.status is JobStatus.RUNNING
    assert held.worker == "w1"
    assert held.attempt == 1
    assert queue.heartbeat(job, lease=1.0)
    clock.advance(2.0)
    reclaimed = queue.claim("w2")
    assert reclaimed is not None
    assert reclaimed.id == job.id


def test_heartbeat_after_cancel_is_false(queue: JobQueue, event_id: str) -> None:
    queue.submit(event_id, FOSTER)
    job = queue.claim("w1")
    assert job is not None
    cancelled = queue.cancel(job.id)
    assert cancelled.status is JobStatus.CANCELLED
    assert cancelled.finished_at is not None
    assert not queue.heartbeat(job)
    assert not queue.finish(job, PAYLOAD)
    assert not queue.fail(job, "no burn")
    assert not queue.release(job)
    assert queue.job(job.id).status is JobStatus.CANCELLED


def test_finish_after_reclaim_is_false(
    queue: JobQueue, store: Store, event_id: str, clock: Clock
) -> None:
    queue.submit(event_id, FOSTER)
    stale = queue.claim("w1")
    assert stale is not None
    clock.advance(LEASE_SECONDS + 1.0)
    fresh = queue.claim("w2")
    assert fresh is not None
    assert (stale.attempt, fresh.attempt) == (1, 2)
    assert not queue.heartbeat(stale)
    assert not queue.finish(stale, PAYLOAD)
    assert not queue.fail(stale, "no burn")
    assert not queue.release(stale)
    assert count_rows(store, models.Result) == 0
    assert queue.job(fresh.id).status is JobStatus.RUNNING
    assert queue.finish(fresh, PAYLOAD)
    done = queue.job(fresh.id)
    assert done.status is JobStatus.SUCCEEDED
    assert done.finished_at is not None
    assert done.error is None
    assert count_rows(store, models.Result) == 1
    assert not queue.finish(fresh, PAYLOAD)


def test_finish_saves_the_result_on_the_pinned_cdm(
    queue: JobQueue, store: Store, event_id: str
) -> None:
    queue.submit(event_id, FOSTER)
    store.ingest([message("C", tca=2.0, created=-30.0)])
    job = queue.claim("w1")
    assert job is not None
    assert queue.finish(job, PAYLOAD)
    with store.read() as session:
        result = session.scalars(select(models.Result)).one()
    assert (result.message_id, result.key, result.payload) == (
        "B",
        maneuver_key(FOSTER, None),
        PAYLOAD,
    )


def test_finish_keeps_a_result_cached_meanwhile(
    queue: JobQueue, store: Store, event_id: str
) -> None:
    # E.g. by a synchronous run of the same settings: the job still ends.
    queue.submit(event_id, FOSTER)
    job = queue.claim("w1")
    assert job is not None
    with store.write() as session:
        assert save_maneuver_in(session, job.message_id, job.key, {"first": True})
    assert queue.finish(job, PAYLOAD)
    assert queue.job(job.id).status is JobStatus.SUCCEEDED
    with store.read() as session:
        assert session.scalars(select(models.Result.payload)).all() == [{"first": True}]


def test_fail_keeps_the_message(queue: JobQueue, event_id: str) -> None:
    queue.submit(event_id, FOSTER)
    job = queue.claim("w1")
    assert job is not None
    assert queue.fail(job, "no burn")
    failed = queue.job(job.id)
    assert failed.status is JobStatus.FAILED
    assert failed.error == "no burn"
    assert failed.attempt == 1


def test_release_does_not_count_an_attempt(queue: JobQueue, event_id: str) -> None:
    queue.submit(event_id, FOSTER)
    job = queue.claim("w1")
    assert job is not None
    assert queue.release(job)
    released = queue.job(job.id)
    assert released.status is JobStatus.QUEUED
    assert released.attempt == 0
    assert released.worker is None
    assert not queue.release(job)
    again = queue.claim("w2")
    assert again is not None
    assert again.attempt == 1
    assert again.claims == 2


def test_stale_write_after_release_and_reclaim_is_false(
    queue: JobQueue, store: Store, event_id: str
) -> None:
    queue.submit(event_id, FOSTER)
    stale = queue.claim("w1")
    assert stale is not None
    assert queue.release(stale)
    fresh = queue.claim("w2")
    assert fresh is not None
    assert (stale.attempt, fresh.attempt) == (1, 1)
    assert (stale.claims, fresh.claims) == (1, 2)
    assert not queue.heartbeat(stale)
    assert not queue.finish(stale, PAYLOAD)
    assert not queue.fail(stale, "no burn")
    assert not queue.release(stale)
    assert not queue.abandon(stale)
    held = queue.job(fresh.id)
    assert held.status is JobStatus.RUNNING
    assert held.worker == "w2"
    assert held.attempt == 1
    assert count_rows(store, models.Result) == 0
    assert queue.finish(fresh, PAYLOAD)
    assert queue.job(fresh.id).status is JobStatus.SUCCEEDED
    assert count_rows(store, models.Result) == 1


def test_cancel_a_queued_job(queue: JobQueue, event_id: str) -> None:
    submission = queue.submit(event_id, FOSTER)
    assert submission.job is not None
    assert queue.cancel(submission.job.id).status is JobStatus.CANCELLED
    assert queue.claim("w1") is None


def test_cancel_a_finished_job_is_refused(queue: JobQueue, event_id: str) -> None:
    queue.submit(event_id, FOSTER)
    job = queue.claim("w1")
    assert job is not None
    assert queue.finish(job, PAYLOAD)
    with pytest.raises(StoreError, match=f"job {job.id} has already ended: succeeded"):
        queue.cancel(job.id)
    assert queue.job(job.id).status is JobStatus.SUCCEEDED


def test_an_unknown_job(queue: JobQueue, event_id: str) -> None:
    with pytest.raises(StoreError, match="no job 7 in the store"):
        queue.cancel(7)
    with pytest.raises(StoreError, match="no job 7 in the store"):
        queue.job(7)


def test_jobs_are_listed_newest_first(
    queue: JobQueue, event_id: str, clock: Clock
) -> None:
    first = queue.submit(event_id, FOSTER).job
    clock.advance(1.0)
    second = queue.submit(event_id, Settings()).job
    assert first is not None
    assert second is not None
    assert [job.id for job in queue.jobs()] == [second.id, first.id]
    claimed = queue.claim("w1")
    assert claimed is not None
    assert claimed.id == first.id
    running = queue.jobs(status=[JobStatus.RUNNING])
    assert [job.id for job in running] == [first.id]
    either = queue.jobs(status={JobStatus.QUEUED, JobStatus.RUNNING})
    assert [job.id for job in either] == [second.id, first.id]
    assert queue.jobs(status=[JobStatus.FAILED]) == []


def _claim(path: str, start: threading.Barrier, worker: str) -> int | None:
    """Claim once, when both processes are ready; the id claimed, if any."""
    with open_store(path) as store:
        queue = JobQueue(store)
        start.wait()
        job = queue.claim(worker)
    return None if job is None else job.id


@pytest.mark.timeout(60)
def test_two_processes_claim_one_job(store: Store, event_id: str) -> None:
    submission = JobQueue(store).submit(event_id, FOSTER)
    assert submission.job is not None
    context = multiprocessing.get_context("spawn")
    with context.Manager() as manager, context.Pool(2) as pool:
        start = manager.Barrier(2)
        claimed = pool.starmap(
            _claim, [(str(store.path), start, worker) for worker in ("w1", "w2")]
        )
    assert sorted(claimed, key=lambda claim: claim is None) == [
        submission.job.id,
        None,
    ]
    job = JobQueue(store).job(submission.job.id)
    assert job.status is JobStatus.RUNNING
    assert job.attempt == 1
