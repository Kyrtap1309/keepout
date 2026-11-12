"""Computations of a job for the worker's tests: module-level, so a spawned
child unpickles them, and importable by the store and CLI tests."""

import multiprocessing
import os
import threading
import time
from pathlib import Path

from builders import SMALL_GRID

from keepout._types import JsonValue
from keepout.errors import ManeuverError
from keepout.store.jobs import Job
from keepout.store.planning import run_maneuver_job


def quick(path: str, job: Job, workers: int | None) -> dict[str, JsonValue]:
    """An answer at once."""
    return {"ok": True}


def workers_given(path: str, job: Job, workers: int | None) -> dict[str, JsonValue]:
    """An answer at once: the ``workers`` it was given."""
    return {"workers": workers}


def sleep_forever(path: str, job: Job, workers: int | None) -> dict[str, JsonValue]:
    """No answer: sleeps until killed."""
    while True:
        time.sleep(60.0)


def sleep_then_quick(path: str, job: Job, workers: int | None) -> dict[str, JsonValue]:
    """The answer of :func:`quick`, after 3 s."""
    time.sleep(3.0)
    return quick(path, job, workers)


def raise_keepout(path: str, job: Job, workers: int | None) -> dict[str, JsonValue]:
    """A keepout error: ``no burn``."""
    raise ManeuverError("no burn")


def raise_bug(path: str, job: Job, workers: int | None) -> dict[str, JsonValue]:
    """A bug: a ``ZeroDivisionError``."""
    return {"ratio": 1 / 0}


def crash(path: str, job: Job, workers: int | None) -> dict[str, JsonValue]:
    """No answer: the process exits at once."""
    os._exit(1)


def crash_once(path: str, job: Job, workers: int | None) -> dict[str, JsonValue]:
    """A crash the first time (no ``crashed`` file beside the store, which it
    makes), then the answer of :func:`quick`."""
    marker = Path(path).with_name("crashed")
    if not marker.exists():
        marker.touch()
        os._exit(1)
    return quick(path, job, workers)


def small_grid_maneuver(
    path: str, job: Job, workers: int | None
) -> dict[str, JsonValue]:
    """:func:`~keepout.store.planning.run_maneuver_job` on
    :data:`~builders.SMALL_GRID`."""
    return run_maneuver_job(path, job, workers, grid=SMALL_GRID)


def _beside(path: str, name: str) -> Path:
    """The file ``name`` beside the store."""
    return Path(path).with_name(name)


def ready_then_sleep(path: str, job: Job, workers: int | None) -> dict[str, JsonValue]:
    """No answer: makes a ``ready`` file beside the store, then sleeps until
    killed."""
    _beside(path, "ready").touch()
    return sleep_forever(path, job, workers)


def exit_when_told(path: str, job: Job, workers: int | None) -> dict[str, JsonValue]:
    """No answer: makes a ``ready`` file beside the store, then exits once
    an ``exit`` file is there."""
    _beside(path, "ready").touch()
    while not _beside(path, "exit").exists():
        time.sleep(0.01)
    os._exit(1)


def with_grandchild(path: str, job: Job, workers: int | None) -> dict[str, JsonValue]:
    """No answer: starts a process of its own, as a pool would, writes its
    pid to a ``grandchild`` file beside the store, then sleeps until
    killed."""
    grandchild = multiprocessing.get_context("spawn").Process(
        target=time.sleep, args=(3600.0,)
    )
    grandchild.start()
    written = _beside(path, "grandchild.part")
    written.write_text(str(grandchild.pid))
    written.replace(_beside(path, "grandchild"))
    return sleep_forever(path, job, workers)


def unpicklable(path: str, job: Job, workers: int | None) -> dict[str, JsonValue]:
    """A result that cannot be sent back: it holds a lock."""
    return {"lock": threading.Lock()}  # type: ignore[dict-item]


def sleep_briefly(path: str, job: Job, workers: int | None) -> dict[str, JsonValue]:
    """The answer of :func:`quick`, after 1 s."""
    time.sleep(1.0)
    return quick(path, job, workers)


def pids_then_sleep(path: str, job: Job, workers: int | None) -> dict[str, JsonValue]:
    """No answer: starts a process of its own, as a pool would, writes its
    own pid and that process's to a ``pids`` file beside the store, then
    sleeps until killed."""
    grandchild = multiprocessing.get_context("spawn").Process(
        target=time.sleep, args=(3600.0,)
    )
    grandchild.start()
    written = _beside(path, "pids.part")
    written.write_text(f"{os.getpid()} {grandchild.pid}")
    written.replace(_beside(path, "pids"))
    return sleep_forever(path, job, workers)
