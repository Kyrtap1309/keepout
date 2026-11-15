"""What the benchmarks' ``compare`` commands share: running a measurement
process, alternating the runs of two trees, and the job summary.

``tools/screening_benchmark.py`` and ``tools/dashboard_benchmark.py`` time
this tree and an older one (CI's merge base) in fresh processes, each with
one thread for the numerical libraries, alternately, and append their
Markdown summary to ``$GITHUB_STEP_SUMMARY``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from keepout._workers import ONE_THREAD

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = ["TimingFailed", "alternate", "run_process", "write_step_summary"]

_Run = TypeVar("_Run")


class TimingFailed(Exception):
    """A measurement process could not run; the message says why."""


def run_process(command: Sequence[str]) -> str:
    """The standard output of ``command``, run with one thread for the
    numerical libraries (:data:`keepout._workers.ONE_THREAD`).

    Raises
    ------
    TimingFailed
        If the process cannot start or fails; the message is the end of its
        standard error, or its exit status.
    """
    try:
        done = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env={**os.environ, **ONE_THREAD},
            check=False,
        )
    except OSError as error:
        raise TimingFailed(f"{command[0]}: {error}") from error
    if done.returncode != 0:
        lines = done.stderr.strip().splitlines()[-5:]
        raise TimingFailed("\n".join(lines) or f"exit status {done.returncode}")
    return done.stdout


def alternate(
    head: Sequence[str],
    base: Sequence[str],
    *,
    repeat: int,
    run: Callable[[Sequence[str]], _Run],
) -> tuple[tuple[_Run, ...], tuple[_Run, ...], str]:
    """The runs of ``head`` and ``base`` commands, ``repeat`` times each,
    alternately.

    The order alternates (head, base, base, head, ...) so drift of the
    machine's speed affects both alike. This tree runs first, and its
    failure is an error; if the base fails, the runs stop there.

    Returns
    -------
    tuple
        This tree's runs, the base's (empty if it failed) and why the base
        failed (empty if it did not).

    Raises
    ------
    TimingFailed
        If a run of this tree fails.
    """
    runs: dict[str, list[_Run]] = {"head": [], "base": []}
    commands = {"head": head, "base": base}
    for index in range(repeat):
        order = ("head", "base") if index % 2 == 0 else ("base", "head")
        for side in order:
            try:
                runs[side].append(run(commands[side]))
            except TimingFailed as error:
                if side == "head":
                    raise
                return tuple(runs["head"]), (), str(error)
    return tuple(runs["head"]), tuple(runs["base"]), ""


def write_step_summary(text: str) -> None:
    """Append ``text`` to the job summary, ``$GITHUB_STEP_SUMMARY``, if set
    (in GitHub Actions)."""
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as stream:
            stream.write(text + "\n")
