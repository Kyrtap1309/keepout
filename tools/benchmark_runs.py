"""What the benchmarks' ``compare`` commands share: running a measurement
process, alternating the runs of two trees, rendering the outcome, and the
job summary.

``tools/screening_benchmark.py`` and ``tools/dashboard_benchmark.py`` time
this tree and an older one (CI's merge base) in fresh processes, each with
one thread for the numerical libraries, alternately, and append their
Markdown summary to ``$GITHUB_STEP_SUMMARY``.
"""

from __future__ import annotations

import os
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Generic, Literal, TypeVar

from keepout._workers import ONE_THREAD

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

__all__ = [
    "SIDES",
    "Comparison",
    "Side",
    "TimingFailed",
    "alternate",
    "cpu_change",
    "run_process",
    "skipped_lines",
    "verdict_line",
    "write_step_summary",
]

_Run = TypeVar("_Run")

Side = Literal["head", "base"]
"""A tree of the comparison: this one (``head``) or the base."""

SIDES: Final[tuple[Side, Side]] = ("head", "base")
"""Both trees, this one first."""


class TimingFailed(Exception):
    """A measurement process could not run; the message says why."""


@dataclass(frozen=True, slots=True)
class Comparison(Generic[_Run]):
    """The runs of this tree and of the base.

    Parameters
    ----------
    head
        This tree's runs.
    base
        The base's runs; empty if it could not run.
    skipped
        Why the base could not run (empty if it did).
    """

    head: tuple[_Run, ...]
    base: tuple[_Run, ...]
    skipped: str = ""


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
) -> Comparison[_Run]:
    """The runs of ``head`` and ``base`` commands, ``repeat`` times each,
    alternately.

    The order alternates (head, base, base, head, ...) so drift of the
    machine's speed affects both alike. This tree runs first, and its
    failure is an error; if the base fails, the runs stop there.

    Raises
    ------
    TimingFailed
        If a run of this tree fails.
    """
    runs: dict[Side, list[_Run]] = {"head": [], "base": []}
    commands: dict[Side, Sequence[str]] = {"head": head, "base": base}
    for index in range(repeat):
        order = SIDES if index % 2 == 0 else SIDES[::-1]
        for side in order:
            try:
                runs[side].append(run(commands[side]))
            except TimingFailed as error:
                if side == "head":
                    raise
                return Comparison(tuple(runs["head"]), (), str(error))
    return Comparison(tuple(runs["head"]), tuple(runs["base"]))


def cpu_change(head: Iterable[float], base: Iterable[float]) -> float:
    """Relative change of the median CPU time from the base to this tree."""
    return statistics.median(head) / statistics.median(base) - 1.0


def skipped_lines(reason: str) -> list[str]:
    """The Markdown lines saying the base could not run, and why."""
    return [
        "The base cannot run the benchmark, so there is no comparison:",
        "",
        "```text",
        reason,
        "```",
    ]


def verdict_line(label: str, change: float, threshold: float) -> tuple[str, bool]:
    """The line judging a CPU time ``change`` against ``threshold``, which
    starts with ``label``, and whether it is a regression."""
    regression = change > threshold
    verdict = "slower than allowed" if regression else "within the limit"
    line = f"{label} {change:+.1%} (limit {threshold:+.0%}): {verdict}."
    return line, regression


def write_step_summary(text: str) -> None:
    """Append ``text`` to the job summary, ``$GITHUB_STEP_SUMMARY``, if set
    (in GitHub Actions)."""
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as stream:
            stream.write(text + "\n")
