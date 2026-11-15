"""A benchmark of the dashboard's reads on a store of 1 000 events and
20 000 CDMs, and of two trees.

Usage::

    uv run python tools/dashboard_benchmark.py run [--store PATH]
        [--events N] [--cdms N] [--seed N] [--repeat N]
    uv run python tools/dashboard_benchmark.py compare --base-python PYTHON
        [--events N] [--cdms N] [--seed N] [--repeat N] [--threshold FRACTION]

``run`` builds a synthetic store (:func:`build_store`; in a temporary
directory, or at ``--store``, where an existing store is timed as it is),
times the events list's first page and the event page's data path without
Streamlit (:func:`measure`, the median of ``--repeat`` reads after a warm
one) and prints both with their targets (spec §8): the list under 1 s and
the event page under 2 s. It fails (exit 1) when a median wall time misses
its target.

``compare`` is CI's regression check, as ``tools/screening_benchmark.py
compare`` (both use ``tools/benchmark_runs.py``): it builds the store once
with this tree's Python and once with ``--base-python``, the Python
environment of the merge base, each with ``tools/dashboard_timing.py``
(this tree's file, so both trees are measured alike), then times both reads
in fresh processes, alternately (this tree, base, base, this tree, ...),
``--repeat`` times (5) per tree. It fails (exit 1) when this tree's median
CPU time of either read exceeds the base's by more than ``--threshold``
(0.2). A base without the dashboard (no ``keepout.dashboard.listing``) is
skipped (exit 0); one that cannot be timed for another reason is skipped
too, with a GitHub Actions warning; one without
``keepout.dashboard.event_data`` is compared on the list only. ``compare``
also appends its Markdown summary to ``$GITHUB_STEP_SUMMARY`` when that is
set (in GitHub Actions).

Both commands report a failure of this tree's build or measurement as
``Error: ...`` and exit 1.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from benchmark_runs import TimingFailed, alternate, run_process, write_step_summary
from dashboard_timing import (
    CDMS,
    EVENTS,
    NO_DASHBOARD,
    REPEAT,
    TimingOutput,
    Timings,
    build_store,
    measure,
)
from generated import parse_json, positive, table

from keepout._text import plural
from keepout.errors import KeepoutError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = [
    "Comparison",
    "Run",
    "TimingFailed",
    "Timings",
    "build_store",
    "compare",
    "main",
    "measure",
    "render_comparison",
    "run_timing",
]

TIMING: Final = Path(__file__).resolve().with_name("dashboard_timing.py")
TOOL: Final = "tools/dashboard_benchmark.py"

LIST_TARGET: Final = 1.0
"""The events list's first page must take less than this [s] (spec §8)."""

EVENT_TARGET: Final = 2.0
"""The event page must take less than this [s] (spec §8)."""

COMPARE_REPEAT: Final = 5
"""Timing processes per tree of ``compare`` unless told otherwise (median)."""

THRESHOLD: Final = 0.2
"""``compare`` fails when this tree's median CPU time of a read exceeds the
base's by more than this fraction, unless told otherwise."""


@dataclass(frozen=True, slots=True)
class Run:
    """One timing process of ``compare`` (``tools/dashboard_timing.py time``).

    Parameters
    ----------
    list_wall, list_cpu
        Wall and CPU time [s] of the events list's first page.
    event_wall, event_cpu
        Wall and CPU time [s] of the event page's data path; ``None`` if
        the tree has no ``keepout.dashboard.event_data``.
    """

    list_wall: float
    list_cpu: float
    event_wall: float | None
    event_cpu: float | None


def run_timing(command: Sequence[str]) -> Run:
    """The measurement printed by a timing process run as ``command``.

    Raises
    ------
    TimingFailed
        If the process cannot start or fails.
    ValueError
        If it prints something other than a measurement.
    """
    output = parse_json(
        run_process(command), TimingOutput, f"the output of {command[0]}"
    )
    return Run(
        list_wall=output["list_wall"],
        list_cpu=output["list_cpu"],
        event_wall=output["event_wall"],
        event_cpu=output["event_cpu"],
    )


@dataclass(frozen=True, slots=True)
class Comparison:
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

    head: tuple[Run, ...]
    base: tuple[Run, ...]
    skipped: str = ""

    @property
    def list_change(self) -> float:
        """Relative change of the list's median CPU time from the base to
        this tree."""
        head = [run.list_cpu for run in self.head]
        base = [run.list_cpu for run in self.base]
        return _median(head) / _median(base) - 1.0

    @property
    def event_change(self) -> float | None:
        """Relative change of the event page's median CPU time from the base
        to this tree; ``None`` if a tree has no event page."""
        head = [run.event_cpu for run in self.head]
        base = [run.event_cpu for run in self.base]
        if None in head or None in base:
            return None
        return _median(head) / _median(base) - 1.0

    @property
    def quiet_skip(self) -> bool:
        """Whether the base was skipped for having no dashboard (no
        warning then)."""
        return NO_DASHBOARD in self.skipped


def _median(values: Sequence[float | None]) -> float:
    return statistics.median(value for value in values if value is not None)


def compare(
    head: Sequence[str],
    base: Sequence[str],
    *,
    repeat: int,
    run: Callable[[Sequence[str]], Run] = run_timing,
) -> Comparison:
    """Timings of ``head`` and ``base`` commands, ``repeat`` times each,
    alternately (:func:`benchmark_runs.alternate`).

    Raises
    ------
    TimingFailed
        If a run of this tree fails.
    """
    head_runs, base_runs, skipped = alternate(head, base, repeat=repeat, run=run)
    return Comparison(head_runs, base_runs, skipped)


def _seconds(values: Sequence[float | None]) -> str:
    """The median of ``values`` [s], ``-`` if there are none."""
    if not values or None in values:
        return "-"
    return f"{_median(values):.3f} s"


def _range(values: Sequence[float | None]) -> str:
    """The range of ``values`` [s], ``-`` if there are none."""
    known = [value for value in values if value is not None]
    if not values or len(known) < len(values):
        return "-"
    return f"{min(known):.3f} - {max(known):.3f} s"


def _verdict(name: str, change: float, threshold: float) -> tuple[str, bool]:
    """A read's line of the comparison, and whether it is a regression."""
    regression = change > threshold
    verdict = "slower than allowed" if regression else "within the limit"
    line = f"{name}: CPU time change {change:+.1%} (limit {threshold:+.0%}): {verdict}."
    return line, regression


def render_comparison(
    comparison: Comparison, setting: str, threshold: float
) -> tuple[str, bool]:
    """The comparison as Markdown, and whether it is a regression."""
    lines = [f"Dashboard benchmark: {setting}.", ""]
    if comparison.skipped:
        lines.append("The base cannot run the benchmark, so there is no comparison:")
        lines += ["", "```text", comparison.skipped, "```"]
        return "\n".join(lines), False
    header = (
        "Tree",
        "Events list: median CPU time",
        "Events list: CPU time range",
        "Event page: median CPU time",
        "Event page: CPU time range",
        "Events list: median wall time",
        "Event page: median wall time",
    )
    rows = []
    for name, runs in (("base", comparison.base), ("this", comparison.head)):
        list_cpu = [run.list_cpu for run in runs]
        event_cpu = [run.event_cpu for run in runs]
        rows.append(
            (
                name,
                _seconds(list_cpu),
                _range(list_cpu),
                _seconds(event_cpu),
                _range(event_cpu),
                _seconds([run.list_wall for run in runs]),
                _seconds([run.event_wall for run in runs]),
            )
        )
    lines += [*table(header, rows), ""]
    text, regression = _verdict("Events list", comparison.list_change, threshold)
    lines.append(text)
    event_change = comparison.event_change
    if event_change is None:
        lines.append(
            "Event page: the base has no event page data path "
            "(keepout.dashboard.event_data)."
        )
    else:
        text, slower = _verdict("Event page", event_change, threshold)
        lines.append(text)
        regression = regression or slower
    return "\n".join(lines), regression


def _command(python: str, action: str, store: Path, *options: str) -> list[str]:
    """A run of ``dashboard_timing.py`` with ``python`` on ``store``."""
    return [python, str(TIMING), action, str(store), *options]


def _compared(args: argparse.Namespace, directory: Path) -> Comparison:
    """The comparison of this tree with the base, each timing its own store
    built in ``directory`` (the base's schema may differ).

    Raises
    ------
    TimingFailed
        If this tree cannot build its store or time it.
    """
    pythons = {"head": sys.executable, "base": args.base_python}
    stores = {side: directory / f"{side}.db" for side in pythons}
    size = ["--events", str(args.events), "--cdms", str(args.cdms)]
    size += ["--seed", str(args.seed)]
    run_process(_command(pythons["head"], "build", stores["head"], *size))
    try:
        run_process(_command(pythons["base"], "build", stores["base"], *size))
    except TimingFailed as error:
        return Comparison((), (), str(error))
    head, base = (
        _command(pythons[side], "time", stores[side], "--repeat", str(REPEAT))
        for side in ("head", "base")
    )
    return compare(head, base, repeat=args.repeat)


def _warning(text: str) -> str:
    """A GitHub Actions warning of ``text``: its last non-empty line, the
    exception of a traceback."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    last = lines[-1] if lines else "no reason given"
    return f"::warning::The merge base cannot be timed: {last}"


def compare_main(args: argparse.Namespace) -> int:
    """The comparison of this tree's dashboard reads with another tree's.

    Returns
    -------
    int
        1 if this tree is slower than the threshold allows, or cannot be
        timed; else 0.
    """
    try:
        with tempfile.TemporaryDirectory() as directory:
            comparison = _compared(args, Path(directory))
    except (TimingFailed, ValueError) as error:
        print(f"Error: this tree cannot be timed: {error}", file=sys.stderr)
        return 1
    setting = (
        f"{plural(args.events, 'event')}, {plural(args.cdms, 'CDM')}, "
        f"{plural(args.repeat, 'run')} per tree"
    )
    text, regression = render_comparison(comparison, setting, args.threshold)
    print(text)
    if comparison.skipped and not comparison.quiet_skip:
        print(_warning(comparison.skipped))
    write_step_summary(text)
    return 1 if regression else 0


def _target(name: str, seconds: float, target: float) -> tuple[str, bool]:
    """A read's line of ``run``, and whether it missed its target."""
    missed = seconds >= target
    verdict = "missed" if missed else "met"
    return f"{name}: {seconds:.3f} s (target < {target:g} s): {verdict}.", missed


def _timed_store(args: argparse.Namespace, directory: Path) -> Timings:
    """The timings of ``--store``, built first unless it exists, or of a
    store built in ``directory``.

    Raises
    ------
    KeepoutError
        If the store cannot be built or read.
    ValueError
        If it cannot be built as asked, or has no upcoming event.
    """
    path: Path = args.store or directory / "dashboard.db"
    if not path.exists():
        start = time.perf_counter()
        build_store(path, events=args.events, cdms=args.cdms, seed=args.seed)
        print(
            f"Built a store of {plural(args.events, 'event')} and "
            f"{plural(args.cdms, 'CDM')} in {time.perf_counter() - start:.1f} s."
        )
    return measure(path, repeat=args.repeat)


def run_main(args: argparse.Namespace) -> int:
    """The dashboard's reads timed against their targets.

    Returns
    -------
    int
        1 if a read misses its target, or the store cannot be built or
        read; else 0.
    """
    try:
        with tempfile.TemporaryDirectory() as directory:
            timings = _timed_store(args, Path(directory))
    except (KeepoutError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    reads = plural(args.repeat, "read")
    print(f"Median of {reads} after a warm one, without Streamlit:")
    lines = [_target("Events list (first page)", timings.list_s, LIST_TARGET)]
    if timings.event_s is not None:
        lines.append(_target("Event page", timings.event_s, EVENT_TARGET))
    for text, _ in lines:
        print(text)
    return 1 if any(missed for _, missed in lines) else 0


def main(argv: Sequence[str] | None = None) -> int:
    """The entry point: ``run`` or ``compare``.

    Returns
    -------
    int
        0 on success; 1 if ``run`` misses a target, ``compare`` finds a
        regression, or this tree cannot be built or timed.
    """
    store = argparse.ArgumentParser(add_help=False)
    store.add_argument("--events", type=positive, default=EVENTS)
    store.add_argument("--cdms", type=positive, default=CDMS)
    store.add_argument("--seed", type=int, default=0)
    parser = argparse.ArgumentParser(prog=TOOL)
    commands = parser.add_subparsers(dest="command", required=True)
    running = commands.add_parser(
        "run", parents=[store], help="time the dashboard's reads against the targets"
    )
    running.add_argument("--store", type=Path, help="default: a temporary store")
    running.add_argument("--repeat", type=positive, default=REPEAT)
    comparing = commands.add_parser(
        "compare",
        parents=[store],
        help="compare this tree's dashboard reads with another tree's",
    )
    comparing.add_argument(
        "--base-python",
        required=True,
        help="Python of the base tree's environment (with its keepout)",
    )
    comparing.add_argument("--repeat", type=positive, default=COMPARE_REPEAT)
    comparing.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args(argv)
    if args.command == "compare":
        return compare_main(args)
    return run_main(args)


if __name__ == "__main__":
    sys.exit(main())
