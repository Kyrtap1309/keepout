"""The worker of a store's job queue: each job computed in a child process.

:func:`run_worker` is a supervisor. It claims a job
(:meth:`~keepout.store.jobs.JobQueue.claim`) and runs its computation (by
default :func:`run_maneuver_job`) in a ``spawn`` child process, the only way
to stop a computation at once. While
the child computes, the supervisor renews the job's lease, ends it at the
time limit and notices a job that is no longer its own (cancelled, purged
with its event, or taken by another worker after a lapsed lease). The child
only reads the store; the supervisor alone writes the job's end and its
result.

How a job ends:

- an answer: the job succeeds, with its result cached on the job's CDM;
- an error of keepout (:class:`~keepout.errors.KeepoutError`): it fails
  with the error's message, and any other error with ``internal error``,
  the child's traceback in the log; neither is tried again;
- a child that exits without either (a crash, a killed process): the job
  goes back to the queue with the attempt counted, or fails after
  :data:`~keepout.store.jobs.MAX_ATTEMPTS`
  (:meth:`~keepout.store.jobs.JobQueue.abandon`);
- the time limit: the child is killed and the job fails;
- a stop request (``stop``, or ``KeyboardInterrupt``): the child is killed
  and the job goes back to the queue, the attempt not counted.

The worker logs on the ``keepout.worker`` logger.
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing
import os
import signal
import socket
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, TypeAlias

from keepout._workers import ONE_THREAD
from keepout.errors import KeepoutError
from keepout.maneuver.planner import DEFAULT_BURN_GRID, BurnGrid
from keepout.store.db import open_store
from keepout.store.jobs import LEASE_SECONDS, Job, JobQueue

if TYPE_CHECKING:
    from multiprocessing.context import SpawnProcess

    from keepout._types import JsonValue

__all__ = [
    "HEARTBEAT_SECONDS",
    "POLL_SECONDS",
    "TIME_LIMIT_SECONDS",
    "Compute",
    "run_maneuver_job",
    "run_worker",
]

HEARTBEAT_SECONDS: Final = 20.0
"""How often a worker renews the lease of its job, in seconds."""

TIME_LIMIT_SECONDS: Final = 600.0
"""How long a job may compute before it is stopped and failed, in seconds."""

POLL_SECONDS: Final = 2.0
"""How long an idle worker waits before it looks at the queue again, in
seconds."""

_TICK: Final = 1.0
"""Most seconds the supervisor waits for its child's answer before it looks
at the time, the lease and ``stop``."""

_PARENT_POLL: Final = 1.0
"""How often a child looks whether its supervisor is still there, in
seconds."""

_INTERNAL: Final = "internal error"
"""The error of a job whose computation raised anything but a keepout
error."""

Compute: TypeAlias = Callable[[str, Job, "int | None"], "dict[str, JsonValue]"]
"""The computation of a job: ``(store path, job, workers) -> result``. A
module-level function, which a spawned child process unpickles."""

logger = logging.getLogger("keepout.worker")


class _Receiver(Protocol):
    """The supervisor's end of the pipe from the child."""

    def poll(self, timeout: float | None = ..., /) -> bool: ...

    def recv(self) -> object: ...


class _Sender(Protocol):
    """The child's end of the pipe to the supervisor."""

    def send(self, obj: object, /) -> None: ...

    def close(self) -> None: ...


def run_maneuver_job(
    path: str, job: Job, workers: int | None, *, grid: BurnGrid = DEFAULT_BURN_GRID
) -> dict[str, JsonValue]:
    """The manoeuvre plan of a job, as JSON: its event as it was at the
    job's CDM, re-screened against that CDM's catalog (the store's copy),
    if it was made from one.

    Parameters
    ----------
    path
        The store's database file.
    job
        The job.
    workers
        Processes of each re-screen, as for
        :func:`~keepout.maneuver.plan_event`.
    grid
        The burns tried.

    Returns
    -------
    dict
        The outcome's :func:`~keepout.store.maneuver_codec.outcome_to_json`.

    Raises
    ------
    StoreError
        If the store cannot be opened or no longer holds the job's event.
    SourceUnavailable
        If the copy of the catalog is gone; the message names it.
    ChecksumMismatch
        If the copy of the catalog changed; the message names it.
    KeepoutError
        As :func:`~keepout.maneuver.plan_event`.
    ValueError
        As :func:`~keepout.maneuver.plan_event`.
    """
    from keepout.maneuver import outcome
    from keepout.sources.catalog import load_reference
    from keepout.store.maneuver_codec import outcome_to_json

    with open_store(path, create=False) as store:
        event = store.event(job.event_id, until=job.message_id)
        reference = store.catalog(job.event_id, until=job.message_id)
    if reference is None:
        catalog, label = None, None
    else:
        catalog, label = load_reference(reference), f"stored copy of {reference.label}"
    planned = outcome.plan_event(
        event,
        job.settings,
        catalog=catalog,
        catalog_label=label,
        grid=grid,
        workers=workers,
    )
    return outcome_to_json(planned)


def run_worker(
    path: str | Path,
    *,
    compute: Compute = run_maneuver_job,
    workers: int | None = None,
    until_idle: bool = False,
    stop: threading.Event | None = None,
    lease: float = LEASE_SECONDS,
    heartbeat: float = HEARTBEAT_SECONDS,
    time_limit: float = TIME_LIMIT_SECONDS,
    poll: float = POLL_SECONDS,
) -> int:
    """The number of jobs a worker ran from a store's queue, one at a time,
    each in a child process, until asked to stop.

    Parameters
    ----------
    path
        The store's database file.
    compute
        The computation of a job, a module-level function.
    workers
        Processes of each computation, passed to ``compute``.
    until_idle
        Return once the queue is empty, rather than wait for more jobs.
    stop
        Return once it is set; its job, if any, goes back to the queue with
        the attempt not counted.
    lease
        How long a claim or a heartbeat holds the job, in seconds.
    heartbeat
        How often the lease is renewed, in seconds: less than ``lease``.
    time_limit
        How long a job may compute before it is failed, in seconds.
    poll
        How long to wait for a job when the queue is empty, in seconds.

    Returns
    -------
    int
        How many jobs it ran: all it claimed but one given back on a stop.

    Raises
    ------
    ValueError
        If ``heartbeat`` is not less than ``lease``.
    StoreError
        If the store cannot be opened, or the database cannot be used.
    KeyboardInterrupt
        On an interrupt; its job goes back to the queue first, as on a stop.
    """
    if not heartbeat < lease:
        raise ValueError(
            f"heartbeat must be less than lease, got {heartbeat} and {lease}"
        )
    for name, value in ONE_THREAD.items():
        os.environ.setdefault(name, value)
    name = f"{socket.gethostname()}:{os.getpid()}"
    supervisor = _Supervisor(
        str(path),
        compute=compute,
        workers=workers,
        stop=stop,
        lease=lease,
        heartbeat=heartbeat,
        time_limit=time_limit,
    )
    ran = 0
    with open_store(path, create=False) as store:
        queue = JobQueue(store)
        while not _is_set(stop):
            job = queue.claim(name, lease=lease)
            if job is None:
                if until_idle:
                    break
                _wait(stop, poll)
                continue
            logger.info(
                "claimed job %d (event %s, attempt %d)",
                job.id,
                job.event_id,
                job.attempt,
            )
            if supervisor.run(queue, job):
                ran += 1
    return ran


def _is_set(stop: threading.Event | None) -> bool:
    """Whether ``stop`` is given and set."""
    return stop is not None and stop.is_set()


def _wait(stop: threading.Event | None, seconds: float) -> None:
    """A pause of ``seconds``, or until ``stop`` is set."""
    if stop is None:
        time.sleep(seconds)
    else:
        stop.wait(seconds)


def _duration(seconds: float) -> str:
    """``seconds`` for people: whole minutes in ``min``, else in ``s``."""
    if seconds >= 60.0 and seconds % 60.0 == 0.0:
        return f"{seconds / 60.0:g} min"
    return f"{seconds:g} s"


class _Supervisor:
    """What runs one job after another in a child process: the options of
    :func:`run_worker`."""

    def __init__(
        self,
        path: str,
        *,
        compute: Compute,
        workers: int | None,
        stop: threading.Event | None,
        lease: float,
        heartbeat: float,
        time_limit: float,
    ) -> None:
        self.path = path
        self.compute = compute
        self.workers = workers
        self.stop = stop
        self.lease = lease
        self.heartbeat = heartbeat
        self.time_limit = time_limit
        self.context = multiprocessing.get_context("spawn")

    def run(self, queue: JobQueue, job: Job) -> bool:
        """Whether a job the worker claimed ran to its end in a child
        process (not given back on a stop).

        Raises
        ------
        StoreError
            If the database cannot be used.
        KeyboardInterrupt
            On an interrupt, the job given back first.
        """
        receiver, sender = self.context.Pipe(duplex=False)
        child = self.context.Process(
            target=_child,
            args=(self.compute, self.path, job, self.workers, sender),
            name=f"keepout job {job.id}",
        )
        try:
            child.start()
            sender.close()
            return self._watch(queue, job, child, receiver)
        except KeyboardInterrupt:
            _kill(child)
            if queue.release(job):
                logger.info("job %d given back to the queue", job.id)
            raise
        finally:
            _kill(child)
            receiver.close()
            sender.close()

    def _watch(
        self, queue: JobQueue, job: Job, child: SpawnProcess, receiver: _Receiver
    ) -> bool:
        """Whether the job ran: the child's answer awaited, and the job
        ended by it, its time limit or the loss of its lease.

        Raises
        ------
        StoreError
            If the database cannot be used.
        """
        start = beat = time.monotonic()
        tick = min(self.heartbeat, _TICK)
        while True:
            if receiver.poll(tick):
                answer = _answer(receiver)
                # A child gone without an answer on a stop was stopped too.
                if answer is not None or not _is_set(self.stop):
                    _end(queue, job, answer)
                    return True
            if _is_set(self.stop):
                _kill(child)
                if queue.release(job):
                    logger.info("job %d given back to the queue", job.id)
                    return False
                _lost(job)
                return True
            now = time.monotonic()
            if now - start >= self.time_limit:
                _kill(child)
                message = f"stopped at the time limit of {_duration(self.time_limit)}"
                if queue.fail(job, message):
                    logger.warning("job %d failed: %s", job.id, message)
                else:
                    _lost(job)
                return True
            if now - beat >= self.heartbeat:
                beat = now
                if not queue.heartbeat(job, lease=self.lease):
                    _kill(child)
                    _lost(job)
                    return True


def _answer(receiver: _Receiver) -> object:
    """The child's message, ``None`` if it exited without one."""
    try:
        return receiver.recv()
    except EOFError:
        return None


def _end(queue: JobQueue, job: Job, answer: object) -> None:
    """The end of a job by its child's answer (``None``: none).

    Raises
    ------
    StoreError
        If the database cannot be used.
    """
    match answer:
        case ("ok", dict() as payload):
            held = queue.finish(job, payload)
            if held:
                logger.info("job %d succeeded", job.id)
        case ("error", str(message), bool(is_keepout), str(text)):
            error = message if is_keepout else _INTERNAL
            held = queue.fail(job, error)
            if held:
                if is_keepout:
                    logger.warning("job %d failed: %s", job.id, message)
                else:
                    logger.error("job %d failed: %s\n%s", job.id, error, text)
        case _:
            held = queue.abandon(job)
            if held:
                logger.warning(
                    "job %d stopped without an answer (attempt %d)",
                    job.id,
                    job.attempt,
                )
    if not held:
        _lost(job)


def _lost(job: Job) -> None:
    """The log line of a job that is no longer the worker's."""
    logger.warning("lost job %d: cancelled, purged or taken by another worker", job.id)


def _kill(child: SpawnProcess) -> None:
    """The end of a child process, if it started: its process group killed
    (on POSIX: the processes it started go with it), and it reaped."""
    if child.pid is None:
        return
    if hasattr(os, "killpg"):
        # Not yet a group of its own: the child alone, killed below.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(child.pid, signal.SIGKILL)
    if child.is_alive():
        child.kill()
    child.join()


def _child(
    compute: Compute,
    path: str,
    job: Job,
    workers: int | None,
    connection: _Sender,
) -> None:
    """The child process: :func:`_send` in a process group of its own (on
    POSIX), deaf to interrupts and terminations.

    The supervisor alone stops it, with its group: a pool it starts goes
    with it, and a signal to the worker's group (Ctrl+C, a service
    manager's SIGTERM) reaches the supervisor, which gives the job back,
    rather than kill the child, which would cost an attempt. A supervisor
    killed at once (SIGKILL) cannot stop it: the child goes by itself
    within a second or so (:func:`_orphaned`).
    """
    if hasattr(os, "setpgrp"):
        os.setpgrp()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    threading.Thread(
        target=_orphaned, args=(os.getppid(),), name="parent watch", daemon=True
    ).start()
    _send(compute, path, job, workers, connection)


def _orphaned(parent: int) -> None:
    """The end of the child once its supervisor ``parent`` is gone: polled
    every :data:`_PARENT_POLL` seconds, then the child's process group (on
    POSIX: the processes it started too) is killed, else the child alone
    exits. Its job's lease lapses and another worker takes it: the child
    must not compute on, unbounded by the time limit."""
    while os.getppid() == parent:
        time.sleep(_PARENT_POLL)
    if hasattr(os, "killpg"):
        os.killpg(os.getpgrp(), signal.SIGKILL)
    os._exit(1)


def _send(
    compute: Compute,
    path: str,
    job: Job,
    workers: int | None,
    connection: _Sender,
) -> None:
    """The child's one message: ``compute``'s answer, or its error; a
    result that cannot be sent is an error too."""
    try:
        try:
            connection.send(_message(compute, path, job, workers))
        except Exception as exc:  # noqa: BLE001 - not picklable: a bug
            connection.send(("error", str(exc), False, traceback.format_exc()))
    finally:
        connection.close()


def _message(
    compute: Compute, path: str, job: Job, workers: int | None
) -> tuple[object, ...]:
    """The message of ``compute``'s answer, or its error; a result that is
    not a dict is an error."""
    try:
        payload: object = compute(path, job, workers)
    except KeepoutError as exc:
        return ("error", str(exc), True, traceback.format_exc())
    except Exception as exc:  # noqa: BLE001 - any bug is the job's failure
        return ("error", str(exc), False, traceback.format_exc())
    if isinstance(payload, dict):
        return ("ok", payload)
    kind = type(payload).__name__
    return ("error", f"a job's result must be a dict, got {kind}", False, "")
