"""A benchmark of the dashboard's reads on a store of 1 000 events and
20 000 CDMs, and of two trees.

``run`` times the events list's first page and the event page's data path on
a synthetic store, without Streamlit, against their targets (under 1 s and
under 2 s; ``docs/dashboard.md``, Speed).

``compare`` is CI's regression check: it times both reads with this tree's
Python and with ``--base-python`` (the merge base's environment), alternately
in fresh processes (``tools/benchmark_runs.py``), and appends its summary to
``$GITHUB_STEP_SUMMARY`` when that is set. A base without the dashboard is
skipped; one that cannot be timed otherwise is skipped with a warning.

Exit status: 1 if ``run`` misses a target, ``compare`` finds this tree's
median CPU time of a read more than ``--threshold`` above the base's, or this
tree cannot be built or timed (``Error: ...``); else 0.
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

from benchmark_runs import (
    SIDES,
    Comparison,
    Side,
    TimingFailed,
    alternate,
    cpu_cells,
    cpu_change,
    run_process,
    skipped_lines,
    verdict_line,
    write_step_summary,
)
from dashboard_timing import (
    CDMS,
    EVENTS,
    NO_DASHBOARD,
    REPEAT,
    Timings,
    build_store,
    measure,
)
from generated import parse_json, positive, table

from keepout._text import plural
from keepout.errors import KeepoutError

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "main",
    "render_comparison",
    "run_timing",
]

TIMING: Final = Path(__file__).resolve().with_name("dashboard_timing.py")
TOOL: Final = "tools/dashboard_benchmark.py"

LIST_TARGET: Final = 1.0
"""The events list's first page must take less than this [s]."""

EVENT_TARGET: Final = 2.0
"""The event page must take less than this [s]."""

COMPARE_REPEAT: Final = 5
"""Timing processes per tree of ``compare`` unless told otherwise (median)."""

THRESHOLD: Final = 0.2
"""``compare`` fails when this tree's median CPU time of a read exceeds the
base's by more than this fraction, unless told otherwise."""

DIGITS: Final = 3
"""Decimals of the times [s] the tool prints."""


@dataclass(frozen=True, slots=True)
class StoreSize:
    """The synthetic store timed: ``events`` events of ``cdms`` CDMs in all,
    drawn with the random ``seed`` (:func:`dashboard_timing.build_store`)."""

    events: int
    cdms: int
    seed: int


@dataclass(frozen=True, slots=True)
class RunOptions:
    """What ``run`` is asked: the store to build (unless ``store`` exists),
    the store's file (``None``: a temporary one) and the timed reads."""

    size: StoreSize
    store: Path | None
    repeat: int


@dataclass(frozen=True, slots=True)
class CompareOptions:
    """What ``compare`` is asked: the store each tree builds, the base's
    Python, the timing processes per tree and the largest CPU time change
    that is not a regression."""

    size: StoreSize
    base_python: str
    repeat: int
    threshold: float


def run_timing(command: Sequence[str]) -> Timings:
    """The measurement printed by a timing process run as ``command``.

    Raises
    ------
    TimingFailed
        If the process cannot start or fails.
    ValueError
        If it prints something other than a measurement.
    """
    return parse_json(run_process(command), Timings, f"the output of {command[0]}")


def _seconds(values: Sequence[float]) -> str:
    """The median of ``values`` [s]."""
    return f"{statistics.median(values):.{DIGITS}f} s"


def render_comparison(
    comparison: Comparison[Timings], setting: str, threshold: float
) -> tuple[str, bool]:
    """The comparison as Markdown, and whether it is a regression."""
    lines = [f"Dashboard benchmark: {setting}.", ""]
    if comparison.skipped:
        lines += skipped_lines(comparison.skipped)
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
    runs: dict[Side, tuple[Timings, ...]] = {
        "head": comparison.head,
        "base": comparison.base,
    }
    list_cpu = {side: [run["list_cpu"] for run in runs[side]] for side in runs}
    event_cpu = {side: [run["event_cpu"] for run in runs[side]] for side in runs}
    trees: tuple[tuple[str, Side], ...] = (("base", "base"), ("this", "head"))
    rows: list[tuple[str, ...]] = [
        (
            name,
            *cpu_cells(list_cpu[side], DIGITS),
            *cpu_cells(event_cpu[side], DIGITS),
            _seconds([run["list_wall"] for run in runs[side]]),
            _seconds([run["event_wall"] for run in runs[side]]),
        )
        for name, side in trees
    ]
    lines += [*table(header, rows), ""]
    verdicts = [
        verdict_line(
            f"{name}: CPU time change",
            cpu_change(cpu["head"], cpu["base"]),
            threshold,
        )
        for name, cpu in (("Events list", list_cpu), ("Event page", event_cpu))
    ]
    lines += [text for text, _ in verdicts]
    return "\n".join(lines), any(slower for _, slower in verdicts)


def _command(python: str, action: str, store: Path, *options: str) -> list[str]:
    """A run of ``dashboard_timing.py`` with ``python`` on ``store``."""
    return [python, str(TIMING), action, str(store), *options]


def _compared(options: CompareOptions, directory: Path) -> Comparison[Timings]:
    """The comparison of this tree with the base, each timing its own store
    built in ``directory`` (the base's schema may differ).

    Raises
    ------
    TimingFailed
        If this tree cannot build its store or time it.
    """
    pythons: dict[Side, str] = {"head": sys.executable, "base": options.base_python}
    stores = {side: directory / f"{side}.db" for side in pythons}
    store = options.size
    size = ["--events", str(store.events), "--cdms", str(store.cdms)]
    size += ["--seed", str(store.seed)]
    run_process(_command(pythons["head"], "build", stores["head"], *size))
    try:
        run_process(_command(pythons["base"], "build", stores["base"], *size))
    except TimingFailed as error:
        return Comparison((), (), str(error))
    head, base = (
        _command(pythons[side], "time", stores[side], "--repeat", str(REPEAT))
        for side in SIDES
    )
    return alternate(head, base, repeat=options.repeat, run=run_timing)


def _warning(text: str) -> str:
    """A GitHub Actions warning of ``text``: its last non-empty line, the
    exception of a traceback."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    last = lines[-1] if lines else "no reason given"
    return f"::warning::The merge base cannot be timed: {last}"


def compare_main(options: CompareOptions) -> int:
    """The comparison of this tree's dashboard reads with another tree's.

    Returns
    -------
    int
        1 if this tree is slower than the threshold allows, or cannot be
        timed; else 0.
    """
    try:
        with tempfile.TemporaryDirectory() as directory:
            comparison = _compared(options, Path(directory))
    except (TimingFailed, ValueError) as error:
        print(f"Error: this tree cannot be timed: {error}", file=sys.stderr)
        return 1
    size = options.size
    setting = (
        f"{plural(size.events, 'event')}, {plural(size.cdms, 'CDM')}, "
        f"{plural(options.repeat, 'run')} per tree"
    )
    text, regression = render_comparison(comparison, setting, options.threshold)
    print(text)
    # A base without the dashboard is skipped quietly; any other failure warns.
    if comparison.skipped and NO_DASHBOARD not in comparison.skipped:
        print(_warning(comparison.skipped))
    write_step_summary(text)
    return 1 if regression else 0


def _target(name: str, seconds: float, target: float) -> tuple[str, bool]:
    """A read's line of ``run``, and whether it missed its target."""
    missed = seconds >= target
    verdict = "missed" if missed else "met"
    return f"{name}: {seconds:.{DIGITS}f} s (target < {target:g} s): {verdict}.", missed


def _timed_store(options: RunOptions, directory: Path) -> Timings:
    """The timings of ``--store``, built first unless it exists, or of a
    store built in ``directory``.

    Raises
    ------
    KeepoutError
        If the store cannot be built or read.
    ValueError
        If it cannot be built as asked, or has no upcoming event.
    """
    path = options.store or directory / "dashboard.db"
    size = options.size
    if not path.exists():
        start = time.perf_counter()
        build_store(path, events=size.events, cdms=size.cdms, seed=size.seed)
        print(
            f"Built a store of {plural(size.events, 'event')} and "
            f"{plural(size.cdms, 'CDM')} in {time.perf_counter() - start:.1f} s."
        )
    return measure(path, repeat=options.repeat)


def run_main(options: RunOptions) -> int:
    """The dashboard's reads timed against their targets.

    Returns
    -------
    int
        1 if a read misses its target, or the store cannot be built or
        read; else 0.
    """
    try:
        with tempfile.TemporaryDirectory() as directory:
            timings = _timed_store(options, Path(directory))
    except (KeepoutError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    reads = plural(options.repeat, "read")
    print(f"Median of {reads} after a warm one, without Streamlit:")
    lines = [
        _target("Events list (first page)", timings["list_wall"], LIST_TARGET),
        _target("Event page", timings["event_wall"], EVENT_TARGET),
    ]
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
    size = StoreSize(events=args.events, cdms=args.cdms, seed=args.seed)
    if args.command == "compare":
        return compare_main(
            CompareOptions(
                size=size,
                base_python=args.base_python,
                repeat=args.repeat,
                threshold=args.threshold,
            )
        )
    return run_main(RunOptions(size=size, store=args.store, repeat=args.repeat))


if __name__ == "__main__":
    sys.exit(main())
