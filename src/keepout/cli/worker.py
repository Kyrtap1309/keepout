"""``keepout worker``: the store's queued jobs (manoeuvre plans, missing Pc),
computed one after another until stopped.

``keepout.store`` is imported in the functions, never here: ``keepout
--help`` does not load it.
"""

from __future__ import annotations

import logging
import signal
import sys
import threading
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Annotated, Final, TypeAlias

import typer

from keepout._text import plural
from keepout.cli.options import Workers
from keepout.cli.store import StorePath, importing_store

if TYPE_CHECKING:
    from types import FrameType


__all__ = ["worker_command"]

_LOG_FORMAT: Final = "%(asctime)s %(levelname)s %(message)s"
"""The worker's log lines on standard error."""

_STOP_SIGNALS: Final = (signal.SIGTERM, signal.SIGINT)
"""The signals that stop the worker once its job is given back."""

UntilIdle: TypeAlias = Annotated[
    bool,
    typer.Option("--until-idle", help="Stop once the queue is empty."),
]
"""``--until-idle``: return when no job is queued instead of waiting."""


@contextmanager
def _stopped_by_signals(stop: threading.Event) -> Generator[None]:
    """SIGTERM and SIGINT (Ctrl+C) set ``stop`` while in the context; their
    handlers are restored on the way out."""

    def handler(_signum: int, _frame: FrameType | None) -> None:
        stop.set()

    previous = {signum: signal.signal(signum, handler) for signum in _STOP_SIGNALS}
    try:
        yield
    finally:
        for signum, old in previous.items():
            signal.signal(signum, old)


@contextmanager
def _logged(name: str) -> Generator[None]:
    """The log ``name`` at INFO on standard error while in the context."""
    logger = logging.getLogger(name)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)


def worker_command(
    *,
    store: StorePath,
    until_idle: UntilIdle = False,
    workers: Workers = None,
) -> None:
    """Compute the store's queued jobs, one at a time, until stopped.

    Needs the store extra. Each job is computed in a child process. A
    manoeuvre plan is cached on the CDM it was queued for, where keepout
    maneuver --event finds it; a refresh caches the Pc its event's CDMs
    lacked. SIGTERM or Ctrl+C stops the worker: its job goes back
    to the queue, not counted as an attempt, and the worker exits with
    status 0. Its log is on standard error; the number of jobs it ran is
    printed on exit.
    """
    with importing_store():
        from keepout.store import planning
        from keepout.store.worker import LOGGER_NAME, run_worker

    stop = threading.Event()
    with _logged(LOGGER_NAME), _stopped_by_signals(stop):
        ran = run_worker(
            store,
            # Looked up now, not bound at import: a test replaces it.
            compute=planning.run_job,
            workers=workers,
            until_idle=until_idle,
            stop=stop,
        )
    typer.echo(f"Ran {plural(ran, 'job')}.")
