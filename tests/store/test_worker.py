"""The worker: a job computed in a child process, its result written by the
supervisor; time limit, cancel, crashes, stop and a killed worker."""

import contextlib
import dataclasses
import logging
import multiprocessing
import os
import shutil
import signal
import threading
import time
from collections.abc import Callable, Generator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any, Final, TypedDict

import pytest
from astropy.time import Time
from builders import SMALL_GRID
from cli_builders import operator_kvn
from sqlalchemy import select
from store_builders import FOSTER, count_rows, message
from worker_jobs import (
    crash,
    crash_once,
    exit_when_told,
    pids_then_sleep,
    quick,
    raise_bug,
    raise_keepout,
    ready_then_sleep,
    sleep_briefly,
    sleep_forever,
    sleep_then_quick,
    small_grid_maneuver,
    unpicklable,
    with_grandchild,
    workers_given,
)

import keepout.cdm
import keepout.maneuver.outcome
from keepout._text import duration_words
from keepout._workers import ONE_THREAD
from keepout.maneuver import plan_event
from keepout.pc import computation
from keepout.settings import Settings
from keepout.sources.catalog import CatalogReference
from keepout.store import CdmInput, Store, models
from keepout.store.jobs import Job, JobQueue, JobStatus, SubmissionOutcome
from keepout.store.maneuver_codec import outcome_from_json, outcome_to_json
from keepout.store.planning import run_maneuver_job
from keepout.store.worker import (
    TIME_LIMIT_SECONDS,
    _Answer,
    _Failure,
    _next_step,
    _send,
    _Step,
    run_worker,
)

# Every test here runs a worker or its child: a regression fails, not hangs.
pytestmark = pytest.mark.timeout(60)

SAMPLE_CSV = Path(__file__).parents[1] / "omm" / "data" / "celestrak_sample.csv"


class _Timing(TypedDict):
    heartbeat: float
    poll: float


FAST: Final = _Timing(heartbeat=0.2, poll=0.1)
"""Heartbeats and polls fast enough for a test."""


@pytest.fixture
def path(store: Store) -> str:
    return str(store.path)


@pytest.fixture
def job(store: Store) -> Job:
    """A queued job of an event of one CDM."""
    (event_id,) = store.ingest([message("A")]).events
    submitted = JobQueue(store).submit(event_id, FOSTER).job
    assert submitted is not None
    return submitted


def _until(condition: Callable[[], bool], timeout: float = 5.0) -> None:
    """Wait for ``condition``; fail after ``timeout`` seconds."""
    end = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.02)


def _running(store: Store, job: Job) -> Callable[[], bool]:
    return lambda: JobQueue(store).job(job.id).status is JobStatus.RUNNING


@contextmanager
def _in_thread(path: str, **options: Any) -> Generator[Future[int]]:
    """``run_worker(path, **options)`` in a thread; its future. Its
    ``stop`` (one of its own unless given) is set on the way out, so that a
    failed test does not wait for the thread forever."""
    stop = options.setdefault("stop", threading.Event())
    with ThreadPoolExecutor(1) as pool:
        try:
            yield pool.submit(run_worker, path, **options)
        finally:
            stop.set()


def test_runs_a_job_and_caches_its_result(
    store: Store, path: str, job: Job, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="keepout.worker")
    assert run_worker(path, compute=quick, until_idle=True, **FAST) == 1
    done = JobQueue(store).job(job.id)
    assert done.status is JobStatus.SUCCEEDED
    assert done.attempt == 1
    assert done.worker is not None
    assert done.worker.endswith(f":{os.getpid()}")
    with store.read() as session:
        result = session.scalars(select(models.Result)).one()
        assert (result.message_id, result.key) == (job.message_id, job.key)
        assert result.payload == {"ok": True}
    assert f"claimed job {job.id}" in caplog.text
    assert f"job {job.id} succeeded" in caplog.text
    assert multiprocessing.active_children() == []


def test_the_computation_is_given_the_workers(
    store: Store, path: str, job: Job
) -> None:
    assert run_worker(path, compute=workers_given, workers=3, until_idle=True) == 1
    with store.read() as session:
        assert session.scalars(select(models.Result.payload)).one() == {"workers": 3}


def test_a_cdm_stored_after_the_job_misses_its_result(
    store: Store, path: str, job: Job
) -> None:
    # The job plans the event as it was when queued: its result is cached on
    # that CDM, and the event's newer latest CDM misses it.
    store.ingest([message("B", tca=1.0, created=-60.0)])
    assert store.event(job.event_id).latest.message_id == "B"
    assert run_worker(path, compute=quick, until_idle=True, **FAST) == 1
    assert JobQueue(store).job(job.id).status is JobStatus.SUCCEEDED
    with store.read() as session:
        cached = session.execute(
            select(models.Result.message_id, models.Result.key)
        ).all()
    assert [tuple(row) for row in cached] == [("A", job.key)]
    assert store.maneuver(job.event_id, FOSTER) is None
    again = JobQueue(store).submit(job.event_id, FOSTER)
    assert again.outcome is SubmissionOutcome.QUEUED
    assert again.job is not None
    assert again.job.message_id == "B"


def test_an_empty_queue(path: str) -> None:
    assert run_worker(path, compute=quick, until_idle=True) == 0


def test_keepout_error_is_the_message(store: Store, path: str, job: Job) -> None:
    assert run_worker(path, compute=raise_keepout, until_idle=True, **FAST) == 1
    failed = JobQueue(store).job(job.id)
    assert failed.status is JobStatus.FAILED
    assert failed.error == "no burn"
    assert failed.attempt == 1
    assert count_rows(store, models.Result) == 0


def test_bug_is_internal_error_with_traceback_logged(
    store: Store, path: str, job: Job, caplog: pytest.LogCaptureFixture
) -> None:
    assert run_worker(path, compute=raise_bug, until_idle=True, **FAST) == 1
    failed = JobQueue(store).job(job.id)
    assert failed.status is JobStatus.FAILED
    assert failed.error == "internal error"
    assert failed.attempt == 1
    assert "ZeroDivisionError" in caplog.text
    assert "Traceback" in caplog.text


def test_time_limit(store: Store, path: str, job: Job) -> None:
    started = time.monotonic()
    ran = run_worker(
        path, compute=sleep_forever, until_idle=True, time_limit=1.0, **FAST
    )
    assert ran == 1
    assert time.monotonic() - started < 5.0
    failed = JobQueue(store).job(job.id)
    assert failed.status is JobStatus.FAILED
    assert failed.error == "stopped at the time limit of 1 s"
    assert failed.attempt == 1
    assert multiprocessing.active_children() == []


def test_the_default_time_limit_in_words() -> None:
    assert duration_words(TIME_LIMIT_SECONDS) == "10 minutes"


@pytest.mark.parametrize(
    ("stopping", "computing", "since_beat", "step"),
    [
        (False, 1.0, 1.0, _Step.WAIT),
        (False, 1.0, 20.0, _Step.RENEW),
        (False, 600.0, 20.0, _Step.TIME_OUT),
        (True, 600.0, 20.0, _Step.GIVE_BACK),
        (True, 0.0, 0.0, _Step.GIVE_BACK),
    ],
)
def test_the_supervisor_s_next_step(
    *, stopping: bool, computing: float, since_beat: float, step: _Step
) -> None:
    # A stop comes first, then the time limit, then the heartbeat.
    assert (
        _next_step(
            stopping=stopping,
            computing=computing,
            since_beat=since_beat,
            time_limit=600.0,
            heartbeat=20.0,
        )
        is step
    )


def test_the_worker_restores_the_environment(
    monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    for name in ONE_THREAD:
        monkeypatch.delenv(name, raising=False)
    assert run_worker(path, until_idle=True, **FAST) == 0
    assert not set(ONE_THREAD) & set(os.environ)


def test_cancel_while_running(
    store: Store, path: str, job: Job, caplog: pytest.LogCaptureFixture
) -> None:
    with _in_thread(path, compute=sleep_forever, until_idle=True, **FAST) as ran:
        _until(_running(store, job))
        JobQueue(store).cancel(job.id)
        assert ran.result(timeout=5.0) == 1
    assert multiprocessing.active_children() == []
    cancelled = JobQueue(store).job(job.id)
    assert cancelled.status is JobStatus.CANCELLED
    assert count_rows(store, models.Result) == 0
    assert f"lost job {job.id}" in caplog.text


def test_crash_is_retried_once(store: Store, path: str, job: Job) -> None:
    ran = run_worker(path, compute=crash_once, until_idle=True, lease=1.0, **FAST)
    assert ran == 2
    done = JobQueue(store).job(job.id)
    assert done.status is JobStatus.SUCCEEDED
    assert done.attempt == 2
    assert count_rows(store, models.Result) == 1


def test_crash_twice_fails(store: Store, path: str, job: Job) -> None:
    ran = run_worker(path, compute=crash, until_idle=True, lease=1.0, **FAST)
    assert ran == 2
    failed = JobQueue(store).job(job.id)
    assert failed.status is JobStatus.FAILED
    assert failed.error == "the worker stopped twice without an answer"
    assert failed.attempt == 2
    assert count_rows(store, models.Result) == 0


def test_stop_releases_the_job(store: Store, path: str, job: Job) -> None:
    stop = threading.Event()
    with _in_thread(path, compute=sleep_forever, stop=stop, **FAST) as ran:
        _until(_running(store, job))
        stop.set()
        assert ran.result(timeout=5.0) == 0
    assert multiprocessing.active_children() == []
    released = JobQueue(store).job(job.id)
    assert released.status is JobStatus.QUEUED
    assert released.attempt == 0
    assert released.worker is None


def test_stop_while_idle(path: str) -> None:
    stop = threading.Event()
    with _in_thread(path, compute=quick, stop=stop, **FAST) as ran:
        time.sleep(0.3)
        assert not ran.done()
        stop.set()
        assert ran.result(timeout=5.0) == 0


def test_interrupt_releases_the_job(
    store: Store, path: str, job: Job, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(self: JobQueue, job: Job, *, lease: float) -> bool:
        raise KeyboardInterrupt

    monkeypatch.setattr(JobQueue, "heartbeat", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_worker(path, compute=sleep_forever, until_idle=True, **FAST)
    assert multiprocessing.active_children() == []
    released = JobQueue(store).job(job.id)
    assert released.status is JobStatus.QUEUED
    assert released.attempt == 0


def test_purged_event_while_running(
    store: Store, path: str, job: Job, caplog: pytest.LogCaptureFixture
) -> None:
    with _in_thread(path, compute=sleep_forever, until_idle=True, **FAST) as ran:
        _until(_running(store, job))
        assert store.purge(Time.now()) == 1
        assert ran.result(timeout=5.0) == 1
    assert multiprocessing.active_children() == []
    assert JobQueue(store).jobs() == []
    assert count_rows(store, models.Result) == 0
    assert f"lost job {job.id}" in caplog.text


def _with_catalog(store: Store, tmp_path: Path) -> str:
    """An event of one CDM made from a catalog (the store keeps a copy of
    the sample catalog); its id."""
    original = tmp_path / "active.csv"
    shutil.copyfile(SAMPLE_CSV, original)
    reference = CatalogReference.of_file(original)
    (event_id,) = store.ingest([message("A")], catalog=reference).events
    original.unlink()
    return event_id


def test_missing_catalog_copy_fails_the_job(
    store: Store, path: str, tmp_path: Path
) -> None:
    event_id = _with_catalog(store, tmp_path)
    submitted = JobQueue(store).submit(event_id, FOSTER).job
    assert submitted is not None
    stored = store.catalog(event_id)
    assert stored is not None
    (copy,) = (file.path for file in stored.files)
    copy.unlink()
    assert run_worker(path, until_idle=True, **FAST) == 1
    failed = JobQueue(store).job(submitted.id)
    assert failed.status is JobStatus.FAILED
    assert failed.error is not None
    assert copy.name in failed.error
    assert failed.attempt == 1


def test_run_maneuver_job_plans_the_pinned_cdm_with_its_catalog(
    store: Store, path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # In this process: what the job hands plan_event.
    event_id = _with_catalog(store, tmp_path)
    submitted = JobQueue(store).submit(event_id, FOSTER).job
    assert submitted is not None
    store.ingest([message("B", tca=1.0, created=-60.0)])  # after the pin
    given: dict[str, Any] = {}
    real = keepout.maneuver.outcome.plan_event

    def spy(event: Any, **options: Any) -> Any:
        given.update(options, event=event)
        radius = options["combined_hard_body_radius"]
        return real(event, grid=options["grid"], combined_hard_body_radius=radius)

    monkeypatch.setattr(keepout.maneuver.outcome, "plan_event", spy)
    run_maneuver_job(path, submitted, 3, grid=SMALL_GRID)
    assert [cdm.message_id for cdm in given["event"].cdms] == ["A"]
    assert given["catalog_label"] == f"stored copy of {tmp_path / 'active.csv'}"
    assert len(given["catalog"]) > 0
    assert given["workers"] == 3
    assert given["grid"] == SMALL_GRID
    assert given["compute"].method == "foster"
    assert given["combined_hard_body_radius"] == FOSTER.combined_hard_body_radius


def test_run_maneuver_job_without_a_catalog(store: Store, path: str, job: Job) -> None:
    payload = run_maneuver_job(path, job, 1, grid=SMALL_GRID)
    decoded = outcome_from_json(payload, store.event(job.event_id))
    assert decoded.catalog_label is None
    assert decoded.burns
    assert all(burn.rescreen is None for burn in decoded.burns)


def _not_a_dict(path: str, job: Job, workers: int | None) -> Any:
    return [1, 2]


def _traceback_apart(sent: object) -> tuple[object, str]:
    """``sent`` without its traceback, and the traceback's first word."""
    if isinstance(sent, _Failure):
        return dataclasses.replace(sent, traceback=""), sent.traceback[:9]
    return sent, ""


@pytest.mark.parametrize(
    ("compute", "answer", "traceback"),
    [
        (quick, _Answer({"ok": True}), ""),
        (
            raise_keepout,
            _Failure(message="no burn", from_keepout=True, traceback=""),
            "Traceback",
        ),
        (
            raise_bug,
            _Failure(message="division by zero", from_keepout=False, traceback=""),
            "Traceback",
        ),
        (
            _not_a_dict,
            _Failure(
                message="a job's result must be a dict, got list",
                from_keepout=False,
                traceback="",
            ),
            "",
        ),
    ],
    ids=["answer", "keepout-error", "bug", "not-a-dict"],
)
def test_the_child_sends_one_answer(
    compute: Callable[..., Any], answer: _Answer | _Failure, traceback: str, job: Job
) -> None:
    # In this process: what a child sends.
    receiver, sender = multiprocessing.Pipe(duplex=False)
    _send(compute, "keepout.db", job, None, sender)
    assert _traceback_apart(receiver.recv()) == (answer, traceback)
    with pytest.raises(EOFError):
        receiver.recv()
    receiver.close()


def test_the_child_sends_an_unpicklable_result_as_a_bug(job: Job) -> None:
    receiver, sender = multiprocessing.Pipe(duplex=False)
    _send(unpicklable, "keepout.db", job, None, sender)
    sent = receiver.recv()
    assert isinstance(sent, _Failure)
    assert sent.from_keepout is False
    assert "pickle" in sent.message
    assert sent.traceback.startswith("Traceback")
    with pytest.raises(EOFError):
        receiver.recv()
    receiver.close()


def test_an_unpicklable_result_is_internal_error(
    store: Store, path: str, job: Job
) -> None:
    assert run_worker(path, compute=unpicklable, until_idle=True, **FAST) == 1
    failed = JobQueue(store).job(job.id)
    assert failed.status is JobStatus.FAILED
    assert failed.error == "internal error"
    assert failed.attempt == 1


def test_heartbeat_must_be_less_than_lease(path: str) -> None:
    with pytest.raises(ValueError, match="heartbeat must be less than lease"):
        run_worker(path, compute=quick, until_idle=True, lease=1.0, heartbeat=1.0)


def _lease_until(store: Store, job: Job) -> Time:
    """When the lease of ``job`` runs out."""
    with store.read() as session:
        row = session.get(models.Job, job.id)
        assert row is not None
        assert row.lease_until is not None
        return row.lease_until


def test_heartbeats_renew_the_lease(store: Store, path: str, job: Job) -> None:
    options = {"lease": 2.0, "heartbeat": 0.1, "poll": 0.1}
    with _in_thread(path, compute=sleep_briefly, until_idle=True, **options) as ran:
        _until(_running(store, job))
        claimed = _lease_until(store, job)
        _until(lambda: _lease_until(store, job) > claimed)
        assert ran.result(timeout=5.0) == 1
    done = JobQueue(store).job(job.id)
    assert done.status is JobStatus.SUCCEEDED
    assert done.attempt == 1


def _gone(pid: int) -> bool:
    """Whether process ``pid`` is gone (or a zombie, dead but not reaped)."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return True
    return stat.rsplit(")", 1)[1].split()[0] in {"Z", "X"}


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs /proc")
def test_the_processes_of_a_child_go_with_it(store: Store, path: str, job: Job) -> None:
    marker = Path(path).with_name("grandchild")
    with _in_thread(path, compute=with_grandchild, until_idle=True, **FAST) as ran:
        _until(marker.exists, timeout=10.0)
        grandchild = int(marker.read_text())
        assert not _gone(grandchild)
        JobQueue(store).cancel(job.id)
        assert ran.result(timeout=5.0) == 1
    _until(lambda: _gone(grandchild))


def _the_child() -> BaseProcess:
    (child,) = multiprocessing.active_children()
    return child


def test_the_child_ignores_terminations(store: Store, path: str, job: Job) -> None:
    # A SIGTERM to the worker's process group does not reach the job.
    stop = threading.Event()
    ready = Path(path).with_name("ready")
    with _in_thread(path, compute=ready_then_sleep, stop=stop, **FAST) as ran:
        _until(ready.exists, timeout=10.0)
        child = _the_child()
        assert child.pid is not None
        os.kill(child.pid, signal.SIGTERM)
        time.sleep(0.5)
        assert child.is_alive()
        assert JobQueue(store).job(job.id).status is JobStatus.RUNNING
        stop.set()
        assert ran.result(timeout=5.0) == 0
    released = JobQueue(store).job(job.id)
    assert released.status is JobStatus.QUEUED
    assert released.attempt == 0


def test_a_child_gone_on_a_stop_is_given_back(
    store: Store, path: str, job: Job
) -> None:
    # The child exits right after stop is set, while the supervisor waits a
    # full second for it: its end is the stop's, not a crash.
    stop = threading.Event()
    ready = Path(path).with_name("ready")
    options = {"lease": 5.0, "heartbeat": 1.0, "poll": 0.1}
    with _in_thread(path, compute=exit_when_told, stop=stop, **options) as ran:
        _until(ready.exists, timeout=10.0)
        stop.set()
        Path(path).with_name("exit").touch()
        assert ran.result(timeout=5.0) == 0
    released = JobQueue(store).job(job.id)
    assert released.status is JobStatus.QUEUED
    assert released.attempt == 0


def _worker_until_killed(path: str) -> None:
    run_worker(path, compute=sleep_then_quick, lease=1.0, heartbeat=0.2)


def test_killed_worker_job_finishes_once(store: Store, path: str, job: Job) -> None:
    context = multiprocessing.get_context("spawn")
    worker = context.Process(target=_worker_until_killed, args=(path,))
    worker.start()
    try:
        _until(_running(store, job), timeout=30.0)
    finally:
        assert worker.pid is not None
        os.kill(worker.pid, signal.SIGKILL)
        worker.join()
    time.sleep(1.5)
    ran = run_worker(
        path, compute=sleep_then_quick, until_idle=True, lease=1.0, heartbeat=0.2
    )
    assert ran == 1
    # Taken over once the killed worker's lease ran out, and finished by the
    # second attempt. (One result is cached; the cache keeps one per CDM and
    # key anyway, so that alone would not show a second finish.)
    done = JobQueue(store).job(job.id)
    assert done.status is JobStatus.SUCCEEDED
    assert done.attempt == 2
    assert count_rows(store, models.Result) == 1


def _worker_of_a_long_job(path: str) -> None:
    run_worker(path, compute=pids_then_sleep, **FAST)


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs /proc")
def test_a_killed_worker_takes_its_child_with_it(
    store: Store, path: str, job: Job
) -> None:
    # A worker killed at once (SIGKILL) gives its child no word: the child
    # notices its parent is gone, and goes with the processes it started.
    marker = Path(path).with_name("pids")
    worker = multiprocessing.get_context("spawn").Process(
        target=_worker_of_a_long_job, args=(path,)
    )
    worker.start()
    pids: list[int] = []
    try:
        _until(marker.exists, timeout=30.0)
        pids = [int(pid) for pid in marker.read_text().split()]
        assert not any(_gone(pid) for pid in pids)
    finally:
        assert worker.pid is not None
        os.kill(worker.pid, signal.SIGKILL)
        worker.join()
    try:
        _until(lambda: all(_gone(pid) for pid in pids))
    finally:
        for pid in pids:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)


def test_real_maneuver_job(store: Store, path: str) -> None:
    crossing = keepout.cdm.loads(operator_kvn()[0])
    (event_id,) = store.ingest([CdmInput.of(crossing, "test")]).events
    settings = Settings(compute=computation("foster"))
    submitted = JobQueue(store).submit(event_id, settings).job
    assert submitted is not None
    assert run_worker(path, compute=small_grid_maneuver, until_idle=True) == 1
    assert JobQueue(store).job(submitted.id).status is JobStatus.SUCCEEDED
    with store.read() as session:
        payload = session.scalars(select(models.Result.payload)).one()
    event = store.event(event_id)
    decoded = outcome_from_json(payload, event)
    inline = plan_event(event, compute=settings.compute, grid=SMALL_GRID)
    assert outcome_to_json(decoded) == outcome_to_json(inline) == payload
