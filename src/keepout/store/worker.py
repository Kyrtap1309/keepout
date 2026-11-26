"""The worker of a store's job queue: each job computed in a child process.

:func:`run_worker` is a supervisor. It claims a job
(:meth:`~keepout.store.jobs.JobQueue.claim`) and runs its computation (by
default :func:`~keepout.store.planning.run_job`, a plan or the missing Pc
by the job's kind) in a ``spawn``
child process, the only way to stop a computation at once.

While the child computes, the supervisor renews the job's lease. It kills
the child at the time limit, on a stop, or once the job is no longer its own
(:mod:`keepout.store.jobs` says when). The child only reads the store. The
supervisor alone writes the job's end and its result. A computation that
yields its result in parts (a refresh, :func:`~keepout.store.planning.run_refresh_job`)
has each part written as it comes
(:meth:`~keepout.store.jobs.JobQueue.save_partial`), so a job stopped
halfway keeps them.

How a job ends:

- an answer: the job succeeds, with its result cached on the job's CDM;
- an answer the store cannot cache (a :class:`~keepout.errors.StoreError`
  of :meth:`~keepout.store.jobs.JobQueue.finish` or
  :meth:`~keepout.store.jobs.JobQueue.save_partial`): it fails with the
  reason;
- an error of keepout (:class:`~keepout.errors.KeepoutError`): it fails
  with the error's message, and any other error with ``internal error``,
  the child's traceback in the log; neither is tried again;
- a child that exits without either (a crash, a killed process): the job
  is abandoned (:meth:`~keepout.store.jobs.JobQueue.abandon`), which costs
  an attempt;
- the time limit: the child is killed and the job fails;
- a stop request (``stop``, or ``KeyboardInterrupt``): the child is killed
  and the job is released, the attempt not counted.

The worker logs on the :data:`LOGGER_NAME` logger. Given a heartbeat file,
it touches it each time it looks at the queue and renews a lease: a
container's healthcheck tells a live worker by its age.
"""

from __future__ import annotations

import contextlib
import enum
import logging
import multiprocessing
import os
import signal
import socket
import threading
import time
import traceback
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol, TypeAlias, cast

from keepout._text import duration_words
from keepout._workers import ONE_THREAD, environment
from keepout.errors import KeepoutError, StoreError
from keepout.store.db import open_store
from keepout.store.jobs import LEASE_SECONDS, Job, JobQueue
from keepout.store.planning import run_job

if TYPE_CHECKING:
    from multiprocessing.context import SpawnProcess
    from pathlib import Path

    from keepout._types import JsonValue

__all__ = [
    "HEARTBEAT_SECONDS",
    "LOGGER_NAME",
    "POLL_SECONDS",
    "TIME_LIMIT_SECONDS",
    "Compute",
    "run_worker",
]

HEARTBEAT_SECONDS: Final = 20.0
"""How often a worker renews the lease of its job, in seconds."""

TIME_LIMIT_SECONDS: Final = 600.0
"""How long a job may compute before it is stopped and failed, in seconds."""

POLL_SECONDS: Final = 2.0
"""How long an idle worker waits before it looks at the queue again, in
seconds."""

LOGGER_NAME: Final = "keepout.worker"
"""The name of the worker's logger."""

_TICK: Final = 1.0
"""Most seconds the supervisor waits for its child's answer before it looks
at the time, the lease and ``stop``."""

_PARENT_POLL: Final = 1.0
"""How often a child looks whether its supervisor is still there, in
seconds."""

_SPAWN: Final = multiprocessing.get_context("spawn")
"""The context of the children: ``spawn`` on every platform."""

_INTERNAL: Final = "internal error"
"""The error of a job whose computation raised anything but a keepout
error."""

Compute: TypeAlias = Callable[
    [str, Job, "int | None"],
    "dict[str, JsonValue] | Iterator[dict[str, JsonValue]]",
]
"""The computation of a job: ``(store path, job, workers) -> result``, or an
iterator of the parts of a refresh's result, each written as it comes. A
module-level function, which a spawned child process unpickles."""

logger = logging.getLogger(LOGGER_NAME)


class _Receiver(Protocol):
    """The supervisor's end of the pipe from the child."""

    def poll(self, timeout: float | None = ..., /) -> bool: ...

    def recv(self) -> object: ...


class _Sender(Protocol):
    """The child's end of the pipe to the supervisor."""

    def send(self, obj: object, /) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _Answer:
    """The child's message of a result (of its last part, if in parts)."""

    payload: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class _Part:
    """The child's message of a part of its result, more to come."""

    payload: dict[str, JsonValue]


@dataclass(frozen=True, slots=True, kw_only=True)
class _Failure:
    """The child's message of an error."""

    message: str
    from_keepout: bool
    """Whether it is a :class:`~keepout.errors.KeepoutError`, or a bug."""
    traceback: str


class _Run(enum.Enum):
    """How the supervisor's run of a job ended."""

    RAN = enum.auto()
    """Ended by the worker, or found no longer its own."""
    GIVEN_BACK = enum.auto()
    """Released to the queue on a stop."""


class _Step(enum.Enum):
    """What the supervisor does next while its child computes."""

    WAIT = enum.auto()
    RENEW = enum.auto()
    """Renew the lease."""
    GIVE_BACK = enum.auto()
    """Kill the child and release the job: a stop was asked for."""
    TIME_OUT = enum.auto()
    """Kill the child and fail the job at the time limit."""


def run_worker(
    path: str | Path,
    *,
    compute: Compute = run_job,
    workers: int | None = None,
    until_idle: bool = False,
    stop: threading.Event | None = None,
    lease: float = LEASE_SECONDS,
    heartbeat: float = HEARTBEAT_SECONDS,
    time_limit: float = TIME_LIMIT_SECONDS,
    poll: float = POLL_SECONDS,
    heartbeat_file: Path | None = None,
) -> int:
    """A worker of a store's queue: its jobs run one at a time, each in a
    child process, until it is asked to stop.

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
    heartbeat_file
        A file whose modification time the worker sets each time it looks
        at the queue (every ``poll`` seconds when idle) and renews a lease
        (every ``heartbeat`` seconds while a job computes): made if missing,
        its directory must exist. ``None``: none.

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
    OSError
        If ``heartbeat_file`` cannot be touched.
    KeyboardInterrupt
        On an interrupt; its job goes back to the queue first, as on a stop.
    """
    if not heartbeat < lease:
        raise ValueError(
            f"heartbeat must be less than lease, got {heartbeat} and {lease}"
        )
    stop = stop or threading.Event()
    name = f"{socket.gethostname()}:{os.getpid()}"
    supervisor = _Supervisor(
        path=str(path),
        compute=compute,
        workers=workers,
        stop=stop,
        lease=lease,
        heartbeat=heartbeat,
        time_limit=time_limit,
        heartbeat_file=heartbeat_file,
    )
    ran = 0
    # ONE_THREAD unless set already, while the worker runs: a child spawned
    # meanwhile inherits it, so its numerical libraries run on one thread
    # and do not compete with the processes of its re-screen.
    with (
        environment(ONE_THREAD, override=False),
        open_store(path, create=False) as store,
    ):
        queue = JobQueue(store)
        while not stop.is_set():
            supervisor.beat()
            job = queue.claim(name, lease=lease)
            if job is None:
                if until_idle:
                    break
                stop.wait(poll)
                continue
            logger.info(
                "claimed job %d (event %s, attempt %d)",
                job.id,
                job.event_id,
                job.attempt,
            )
            if supervisor.run(queue, job) is _Run.RAN:
                ran += 1
    return ran


def _next_step(
    *,
    stopping: bool,
    computing: float,
    since_beat: float,
    time_limit: float,
    heartbeat: float,
) -> _Step:
    """What the supervisor does next, its child ``computing`` for so many
    seconds and its last heartbeat ``since_beat`` seconds ago.

    A stop comes first, then the time limit, then the heartbeat.
    """
    if stopping:
        return _Step.GIVE_BACK
    if computing >= time_limit:
        return _Step.TIME_OUT
    if since_beat >= heartbeat:
        return _Step.RENEW
    return _Step.WAIT


@dataclass(frozen=True, slots=True, kw_only=True)
class _Supervisor:
    """What runs one job after another in a child process: the options of
    :func:`run_worker`."""

    path: str
    compute: Compute
    workers: int | None
    stop: threading.Event
    lease: float
    heartbeat: float
    time_limit: float
    heartbeat_file: Path | None

    def beat(self) -> None:
        """Touch the heartbeat file, if there is one."""
        if self.heartbeat_file is not None:
            self.heartbeat_file.touch()

    def run(self, queue: JobQueue, job: Job) -> _Run:
        """How a job the worker claimed ended, run in a child process.

        Raises
        ------
        StoreError
            If the database cannot be used.
        KeyboardInterrupt
            On an interrupt, the job given back first.
        """
        receiver, sender = _SPAWN.Pipe(duplex=False)
        child = _SPAWN.Process(
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
        self,
        queue: JobQueue,
        job: Job,
        child: SpawnProcess,
        receiver: _Receiver,
    ) -> _Run:
        """How the job ended: by the child's answer, the time limit, a stop
        or the loss of its lease; the parts of its result written meanwhile.

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
                if isinstance(answer, _Part):
                    if not _record_part(queue, job, answer.payload):
                        _kill(child)
                        return _Run.RAN
                # A child gone without an answer on a stop was stopped too.
                elif answer is not None or not self.stop.is_set():
                    _record_answer(queue, job, answer)
                    return _Run.RAN
            now = time.monotonic()
            step = _next_step(
                stopping=self.stop.is_set(),
                computing=now - start,
                since_beat=now - beat,
                time_limit=self.time_limit,
                heartbeat=self.heartbeat,
            )
            match step:
                case _Step.WAIT:
                    continue
                case _Step.RENEW:
                    beat = now
                    self.beat()
                    if queue.heartbeat(job, lease=self.lease):
                        continue
                    _kill(child)
                    _lost(job)
                    return _Run.RAN
                case _Step.GIVE_BACK:
                    _kill(child)
                    return self._give_back(queue, job)
                case _Step.TIME_OUT:
                    _kill(child)
                    self._time_out(queue, job)
                    return _Run.RAN

    @staticmethod
    def _give_back(queue: JobQueue, job: Job) -> _Run:
        """Release a job on a stop, its child killed.

        Raises
        ------
        StoreError
            If the database cannot be used.
        """
        if queue.release(job):
            logger.info("job %d given back to the queue", job.id)
            return _Run.GIVEN_BACK
        _lost(job)
        return _Run.RAN

    def _time_out(self, queue: JobQueue, job: Job) -> None:
        """Fail a job at the time limit, its child killed.

        Raises
        ------
        StoreError
            If the database cannot be used.
        """
        message = f"stopped at the time limit of {duration_words(self.time_limit)}"
        if queue.fail(job, message):
            logger.warning("job %d failed: %s", job.id, message)
        else:
            _lost(job)


def _answer(receiver: _Receiver) -> object:
    """The child's message, ``None`` if it exited without one."""
    try:
        return receiver.recv()
    except EOFError:
        return None


def _record_part(queue: JobQueue, job: Job, payload: dict[str, JsonValue]) -> bool:
    """Write a part of a job's result; whether its child computes on: not if
    the job is no longer the worker's, or failed for a part it cannot cache.

    Raises
    ------
    StoreError
        If the database cannot be used.
    """
    try:
        if queue.save_partial(job, payload):
            return True
    except StoreError as exc:
        # As a result finish cannot cache (_record_answer).
        if queue.fail(job, str(exc)):
            logger.error("job %d failed: %s", job.id, exc)
            return False
    _lost(job)
    return False


def _record_answer(queue: JobQueue, job: Job, answer: object) -> None:
    """The end of a job by its child's answer (``None``: none).

    Raises
    ------
    StoreError
        If the database cannot be used.
    """
    match answer:
        case _Answer(payload=payload):
            try:
                held = queue.finish(job, payload)
            except StoreError as exc:
                # A result that cannot be cached is a bug of the computation;
                # a store that cannot be used fails this write too, and the
                # error goes on.
                held = queue.fail(job, str(exc))
                if held:
                    logger.error("job %d failed: %s", job.id, exc)
            else:
                if held:
                    logger.info("job %d succeeded", job.id)
        case _Failure(message=message, from_keepout=from_keepout, traceback=text):
            error = message if from_keepout else _INTERNAL
            held = queue.fail(job, error)
            if held:
                if from_keepout:
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
        # If the child has not yet called setpgrp, killpg finds no group;
        # child.kill() below still kills it.
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
    """The child's messages: the parts of ``compute``'s answer, if any, then
    its answer, or its error; a message that cannot be sent is an error
    too, and the last."""
    try:
        for message in _messages(compute, path, job, workers):
            try:
                connection.send(message)
            except Exception as exc:  # noqa: BLE001 - not picklable: a bug
                connection.send(
                    _Failure(
                        message=str(exc),
                        from_keepout=False,
                        traceback=traceback.format_exc(),
                    )
                )
                return
    finally:
        connection.close()


def _failure(exc: Exception) -> _Failure:
    """The message of ``compute``'s error, with its traceback."""
    return _Failure(
        message=str(exc),
        from_keepout=isinstance(exc, KeepoutError),
        traceback=traceback.format_exc(),
    )


def _not_a_dict(result: object) -> _Failure | None:
    """The error of a result (or part) that is not a dict, which its type
    rules out; ``None`` if it is one."""
    if isinstance(result, dict):
        return None
    kind = type(result).__name__
    message = f"a job's result must be a dict, got {kind}"
    return _Failure(message=message, from_keepout=False, traceback="")


def _messages(
    compute: Compute, path: str, job: Job, workers: int | None
) -> Iterator[_Part | _Answer | _Failure]:
    """The messages of ``compute``'s answer, or of its error.

    An answer in parts is sent one part behind: its last part is the
    answer, which the supervisor writes with the job's end.
    """
    try:
        # Seen as an object: a job's code may break its declared type.
        result = cast("object", compute(path, job, workers))
        if isinstance(result, Iterator):
            parts = cast("Iterator[object]", result)
            result = next(parts, None)
            if result is None:
                yield _Failure(
                    message="a job's result in parts must have a part",
                    from_keepout=False,
                    traceback="",
                )
                return
            for part in parts:
                if (wrong := _not_a_dict(result)) is not None:
                    yield wrong
                    return
                yield _Part(cast("dict[str, JsonValue]", result))
                result = part
    except Exception as exc:  # noqa: BLE001 - any bug is the job's failure
        yield _failure(exc)
        return
    yield _not_a_dict(result) or _Answer(cast("dict[str, JsonValue]", result))
