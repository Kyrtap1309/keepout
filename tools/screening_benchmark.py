"""Benchmark screening on the frozen public catalog, and compare two trees.

Usage::

    uv run python tools/screening_benchmark.py [--check] [--workers N] [--repeat N]
    uv run python tools/screening_benchmark.py compare --base-python PYTHON
        [--every K] [--repeat N] [--threshold FRACTION]

The first form screens :data:`PRIMARIES` (20 LEO satellites, 390 to 1340 km
up) against the whole frozen public catalog (``tests/screening/data/``, see
``tools/screening_snapshot.py``) for 7 days from 0 h UTC of the day it was
retrieved, with each volume of :data:`VOLUMES` and ``--workers`` processes
(4), ``--repeat`` times (3), and writes ``docs/screening-benchmark.json`` and
the README summary block: the settings and what each volume found (the
data), and the median wall and CPU time of the runs with the CPU model
(the timings). ``--check`` screens once and fails when the data differs from
the committed files; it takes the timings from the committed JSON, so they
are shown but never checked (they depend on the machine and the run). The
runs of a volume must find the same, or the tool fails.

``compare`` is CI's regression check. It screens :data:`PRIMARIES` against
every ``--every``-th object of the catalog (:func:`reduced_catalog`; 8th:
about 2 500 objects) for 7 days with the 5 km sphere in one process, timed
by ``tools/screening_timing.py`` in fresh processes, alternately with this
tree's Python and with ``--base-python``, the environment of another tree
(the merge base), ``--repeat`` times each (5). Both sides run the same
measurement file, with one thread for numerical libraries, so only keepout
differs. It fails if the median CPU time of this tree exceeds the base's by
more than ``--threshold`` (0.2). Alternate runs share whatever else the
machine does, and the CPU time of one process is less disturbed by it than
the wall time. If the base cannot run
the measurement (a tree from before ``keepout.screening``, or one whose
``screen`` takes other arguments), the comparison is skipped (exit 0).
``compare`` also appends its Markdown summary to ``$GITHUB_STEP_SUMMARY``
when that is set (in GitHub Actions).

Worker processes are spawned, so this module is only run as a script (or
imported), never from standard input.
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypedDict

import screening_snapshot as snapshot
from astropy.time import Time
from generated import replace_block, write_or_check
from screening_timing import TimingOutput, select, timed

from keepout._units import DAY
from keepout.omm import Omm
from keepout.omm.celestrak import dumps_csv
from keepout.screening import (
    DEFAULT_VOLUME,
    ScreeningResult,
    ScreeningVolume,
    SkipReason,
    VolumeRule,
)
from keepout.screening.pipeline import DEFAULT_DURATION
from keepout.screening.volume import SOCRATES_RADIUS

REPO: Final = Path(__file__).resolve().parents[1]
RESULTS: Final = REPO / "docs/screening-benchmark.json"
TIMING: Final = Path(__file__).resolve().with_name("screening_timing.py")
START: Final = "<!-- screening-benchmark-summary:start -->"
END: Final = "<!-- screening-benchmark-summary:end -->"
TOOL: Final = "tools/screening_benchmark.py"
CPUINFO: Final = Path("/proc/cpuinfo")


PRIMARIES: Final = (
    25544,  # ISS (ZARYA)
    48274,  # CSS (TIANHE)
    20580,  # HST
    45212,  # STARLINK-1231
    43613,  # ICESAT-2
    43476,  # GRACE-FO 1
    39634,  # SENTINEL-1A
    39084,  # LANDSAT 8
    25994,  # TERRA
    27424,  # AQUA
    36508,  # CRYOSAT 2
    40697,  # SENTINEL-2A
    41335,  # SENTINEL-3A
    43013,  # NOAA 20 (JPSS-1)
    38771,  # METOP-B
    54754,  # SWOT
    43070,  # IRIDIUM 135
    47284,  # ONEWEB-0138
    41240,  # JASON-3
    46984,  # SENTINEL-6A
)
"""The benchmark's primaries: 20 LEO payloads from 390 km (CSS) to 1340 km
(SENTINEL-6A), the busy shells included, all in the frozen catalog."""

VOLUMES: Final[Mapping[str, ScreeningVolume | VolumeRule]] = {
    "5 km sphere (default)": DEFAULT_VOLUME,
    "10 km sphere (NASA, GP data)": ScreeningVolume.nasa_gp_data(),
    "SFS Handbook Table 3": ScreeningVolume.sfs_handbook,
}
"""The volumes benchmarked, by the name the README gives them."""

WORKERS: Final = 4
"""Processes the benchmark screens with unless told otherwise, as in the
performance target: 20 primaries x the public catalog x 7 days in under 60 s
with 4 workers."""

REPEAT: Final = 3
"""Runs per volume of the benchmark unless told otherwise (median timing)."""

EVERY: Final = 8
"""``compare`` screens every this-th object of the catalog unless told
otherwise: about 2 500 objects, some 4 s of CPU per run."""

COMPARE_REPEAT: Final = 5
"""Runs per tree of ``compare`` unless told otherwise (median)."""

THRESHOLD: Final = 0.2
"""``compare`` fails when this tree's median CPU time exceeds the base's by
more than this fraction, unless told otherwise."""

ONE_THREAD: Final[Mapping[str, str]] = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}
"""Environment of the timed processes: numerical libraries use one thread
each, so their CPU time does not depend on idle cores."""


class Settings(TypedDict):
    """What the benchmark screens."""

    primaries: list[int]
    objects: int
    retrieved: str
    start: str
    days: float
    workers: int


class Found(TypedDict):
    """What one volume found: the checked data."""

    volume: str
    screened: int
    skipped: dict[str, int]
    approaches: int


class Measured(TypedDict):
    """The median timings of one volume [s]."""

    volume: str
    wall: float
    cpu: float


class Timings(TypedDict):
    """The timings and where they were measured: shown, never checked."""

    cpu_model: str
    logical_cpus: int
    python: str
    runs: int
    date: str
    volumes: list[Measured]


class Summary(TypedDict):
    """The benchmark's results, as written to JSON."""

    settings: Settings
    found: list[Found]
    timings: Timings | None


def cpu_model(cpuinfo: Path = CPUINFO) -> str:
    """The CPU's model name, as the operating system gives it."""
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "model name":
                return value.strip()
    return platform.processor() or platform.machine() or "unknown CPU"


def _days(count: float) -> str:
    return "1 day" if count == 1.0 else f"{count:g} days"


def _processes(workers: int) -> str:
    return "in one process" if workers == 1 else f"with {workers} worker processes"


def read_catalog(data: Path) -> tuple[tuple[Omm, ...], str]:
    """The snapshot's element sets and the day it was retrieved (UTC, ISO)."""
    manifest = snapshot.read_manifest(data)
    return snapshot.read_catalog(data), snapshot.retrieval_day(manifest)


def window_start(retrieved: str) -> Time:
    """0 h UTC of the retrieval day: the benchmark's window starts there."""
    return Time(retrieved, scale="utc")


def found(volume: str, result: ScreeningResult) -> Found:
    """What a screening with ``volume`` found."""
    reasons = [skipped.reason for skipped in result.skipped]
    return {
        "volume": volume,
        "screened": result.catalog_count,
        "skipped": {
            reason: reasons.count(reason) for reason in SkipReason if reason in reasons
        },
        "approaches": len(result.approaches),
    }


def benchmark(
    primaries: Sequence[Omm],
    catalog: Sequence[Omm],
    start: Time,
    *,
    workers: int,
    repeat: int,
    volumes: Mapping[str, ScreeningVolume | VolumeRule] = VOLUMES,
) -> tuple[list[Found], list[Measured]]:
    """Screen with each volume ``repeat`` times; what each found, median times.

    Raises
    ------
    RuntimeError
        If the runs of a volume do not find the same.
    """
    data: list[Found] = []
    timings: list[Measured] = []
    for name, volume in volumes.items():
        runs = [
            timed(primaries, catalog, start, DEFAULT_DURATION, volume, workers=workers)
            for _ in range(repeat)
        ]
        results = [found(name, result) for result, _ in runs]
        if any(result != results[0] for result in results):
            raise RuntimeError(f"{name}: the runs found different results: {results}")
        data.append(results[0])
        timings.append(
            {
                "volume": name,
                "wall": statistics.median(timing.wall for _, timing in runs),
                "cpu": statistics.median(timing.cpu for _, timing in runs),
            }
        )
    return data, timings


def read_timings(path: Path) -> Timings | None:
    """The timings of the results file at ``path``, if it exists."""
    if not path.exists():
        return None
    summary: Summary = json.loads(path.read_text(encoding="utf-8"))
    return summary["timings"]


def positive(text: str) -> int:
    """A command-line count: an integer of at least 1."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _seconds(value: float) -> str:
    return f"{value:.1f} s"


def render_summary(summary: Summary) -> str:
    """The README block."""
    settings = summary["settings"]
    timings = summary["timings"]
    measured = {} if timings is None else {m["volume"]: m for m in timings["volumes"]}
    rows = []
    for numbers in summary["found"]:
        skipped = ", ".join(f"{n} {reason}" for reason, n in numbers["skipped"].items())
        times = measured.get(numbers["volume"])
        rows.append(
            f"| {numbers['volume']} | {numbers['screened']} | {skipped or '-'} "
            f"| {numbers['approaches']} "
            f"| {'-' if times is None else _seconds(times['wall'])} "
            f"| {'-' if times is None else _seconds(times['cpu'])} |"
        )
    if timings is None:
        machine = "Not measured yet."
    else:
        runs = "1 run" if timings["runs"] == 1 else f"median of {timings['runs']} runs"
        machine = (
            f"Times: {runs} on {timings['date']}, {timings['cpu_model']} "
            f"({timings['logical_cpus']} logical CPUs), Python {timings['python']}; "
            "`--check` checks the counts, not the times."
        )
    return "\n".join(
        [
            f"Screening {len(settings['primaries'])} LEO primaries against the "
            f"{settings['objects']} element sets of the frozen public catalog "
            f"(CelesTrak, retrieved {settings['retrieved']}) for "
            f"{_days(settings['days'])} from {settings['start']} UTC, "
            f"{_processes(settings['workers'])} (`{TOOL}`):",
            "",
            "| Volume | Screened | Left out | Approaches | Wall time | CPU time |",
            "|---|---|---|---|---|---|",
            *rows,
            "",
            machine,
        ]
    )


def benchmark_main(args: argparse.Namespace) -> int:
    """Run the benchmark (or with ``--check`` verify its data).

    Returns
    -------
    int
        1 if, with ``--check``, a file is out of date, else 0.
    """
    catalog, retrieved = read_catalog(args.data)
    start = window_start(retrieved)
    primaries = select(catalog, args.primaries)
    data, measured = benchmark(
        primaries,
        catalog,
        start,
        workers=args.workers,
        repeat=1 if args.check else args.repeat,
    )
    if args.check:
        timings = read_timings(args.results)
    else:
        timings = {
            "cpu_model": cpu_model(),
            "logical_cpus": os.cpu_count() or 1,
            "python": platform.python_version(),
            "runs": args.repeat,
            "date": datetime.datetime.now(datetime.UTC).date().isoformat(),
            "volumes": [
                {
                    "volume": m["volume"],
                    "wall": round(m["wall"], 3),
                    "cpu": round(m["cpu"], 3),
                }
                for m in measured
            ],
        }
    summary: Summary = {
        "settings": {
            "primaries": list(args.primaries),
            "objects": len(catalog),
            "retrieved": retrieved,
            "start": start.isot[:16].replace("T", " "),
            "days": DEFAULT_DURATION / DAY,
            "workers": args.workers,
        },
        "found": data,
        "timings": timings,
    }
    files = {
        args.results: json.dumps(summary, indent=2) + "\n",
        args.readme: replace_block(
            args.readme.read_text(encoding="utf-8"), START, END, render_summary(summary)
        ),
    }
    return write_or_check(files, check=args.check, tool=TOOL)


def reduced_catalog(
    catalog: Sequence[Omm], primaries: Sequence[int], every: int
) -> list[Omm]:
    """Every ``every``-th element set of ``catalog`` (the first included) and
    those of the primaries, in catalog order."""
    wanted = set(primaries)
    return [
        omm
        for index, omm in enumerate(catalog)
        if index % every == 0 or omm.catalog_number in wanted
    ]


@dataclass(frozen=True)
class Run:
    """One timed screening of ``compare`` (``tools/screening_timing.py``).

    Parameters
    ----------
    wall, cpu
        Wall and CPU time [s].
    approaches
        Approaches found.
    """

    wall: float
    cpu: float
    approaches: int


class TimingFailed(Exception):
    """A timed screening could not run; the message says why."""


def run_timing(command: Sequence[str]) -> Run:
    """Run a measurement process and read its result.

    Raises
    ------
    TimingFailed
        If the process cannot start or fails.
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
    output: TimingOutput = json.loads(done.stdout)
    return Run(wall=output["wall"], cpu=output["cpu"], approaches=output["approaches"])


@dataclass(frozen=True)
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
    def change(self) -> float:
        """Relative change of the median CPU time from the base to this tree."""
        return _median_cpu(self.head) / _median_cpu(self.base) - 1.0


def _median_cpu(runs: Sequence[Run]) -> float:
    return statistics.median(run.cpu for run in runs)


def compare(
    head: Sequence[str],
    base: Sequence[str],
    *,
    repeat: int,
    run: Callable[[Sequence[str]], Run] = run_timing,
) -> Comparison:
    """Time ``head`` and ``base`` commands ``repeat`` times each, alternately.

    The order alternates (head, base, base, head, ...) so drift of the
    machine's speed affects both alike. This tree runs first, and its
    failure is an error; if the base fails, the comparison stops there.

    Raises
    ------
    TimingFailed
        If a run of this tree fails.
    """
    runs: dict[str, list[Run]] = {"head": [], "base": []}
    commands = {"head": head, "base": base}
    for index in range(repeat):
        order = ("head", "base") if index % 2 == 0 else ("base", "head")
        for side in order:
            try:
                runs[side].append(run(commands[side]))
            except TimingFailed as error:
                if side == "head":
                    raise
                return Comparison(tuple(runs["head"]), (), str(error))
    return Comparison(tuple(runs["head"]), tuple(runs["base"]))


def render_comparison(
    comparison: Comparison, setting: str, threshold: float
) -> tuple[str, bool]:
    """The comparison as Markdown, and whether it is a regression."""
    lines = [f"Screening benchmark: {setting}.", ""]
    if comparison.skipped:
        lines.append("The base cannot run the benchmark, so there is no comparison:")
        lines += ["", "```text", comparison.skipped, "```"]
        return "\n".join(lines), False
    lines += [
        "| Tree | Median CPU time | CPU time range | Median wall time | Approaches |",
        "|---|---|---|---|---|",
    ]
    for name, runs in (("base", comparison.base), ("this", comparison.head)):
        cpu = [run.cpu for run in runs]
        approaches = sorted({run.approaches for run in runs})
        lines.append(
            f"| {name} | {statistics.median(cpu):.2f} s "
            f"| {min(cpu):.2f} - {max(cpu):.2f} s "
            f"| {statistics.median(run.wall for run in runs):.2f} s "
            f"| {', '.join(map(str, approaches))} |"
        )
    regression = comparison.change > threshold
    verdict = "slower than allowed" if regression else "within the limit"
    lines += [
        "",
        f"CPU time change: {comparison.change:+.1%} (limit {threshold:+.0%}): "
        f"{verdict}.",
    ]
    return "\n".join(lines), regression


def compare_main(args: argparse.Namespace) -> int:
    """Compare this tree's screening speed with another tree's.

    Returns
    -------
    int
        1 if this tree is slower than the threshold allows, else 0.
    """
    catalog, retrieved = read_catalog(args.data)
    select(catalog, args.primaries)  # all there, or ValueError
    reduced = reduced_catalog(catalog, args.primaries, args.every)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "catalog.csv.gz"
        path.write_bytes(gzip.compress(dumps_csv(reduced).encode("utf-8")))
        arguments = [
            str(TIMING),
            str(path),
            "--primaries",
            *map(str, args.primaries),
            "--start",
            window_start(retrieved).isot,
            "--days",
            f"{args.days:g}",
            "--radius",
            f"{SOCRATES_RADIUS:g}",
            "--workers",
            "1",
        ]
        comparison = compare(
            [sys.executable, *arguments],
            [args.base_python, *arguments],
            repeat=args.repeat,
        )
    setting = (
        f"{len(args.primaries)} primaries x {len(reduced)} objects x "
        f"{_days(args.days)}, {SOCRATES_RADIUS / 1e3:g} km sphere, 1 process, "
        f"{args.repeat} {'run' if args.repeat == 1 else 'runs'} per tree"
    )
    text, regression = render_comparison(comparison, setting, args.threshold)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as stream:
            stream.write(text + "\n")
    return 1 if regression else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark, or ``compare`` if that is the command.

    Returns
    -------
    int
        0 on success; 1 if, with ``--check``, a file is out of date, or if
        ``compare`` finds a regression.
    """
    data = argparse.ArgumentParser(add_help=False)
    data.add_argument("--data", type=Path, default=snapshot.DATA)
    data.add_argument(
        "--primaries",
        type=int,
        nargs="+",
        default=list(PRIMARIES),
        help=argparse.SUPPRESS,
    )
    parser = argparse.ArgumentParser(prog=TOOL, parents=[data])
    parser.add_argument("--check", action="store_true", help="fail if out of date")
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--readme", type=Path, default=REPO / "README.md")
    parser.add_argument("--workers", type=positive, default=WORKERS)
    parser.add_argument(
        "--repeat", type=positive, default=REPEAT, help="runs per volume (not --check)"
    )
    commands = parser.add_subparsers(dest="command")
    comparing = commands.add_parser(
        "compare",
        parents=[data],
        help="compare this tree's screening speed with another tree's",
    )
    comparing.add_argument(
        "--base-python",
        required=True,
        help="Python of the base tree's environment (with its keepout)",
    )
    comparing.add_argument(
        "--every", type=positive, default=EVERY, help="catalog sample"
    )
    comparing.add_argument("--days", type=float, default=DEFAULT_DURATION / DAY)
    comparing.add_argument("--repeat", type=positive, default=COMPARE_REPEAT)
    comparing.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args(argv)
    if args.command == "compare":
        return compare_main(args)
    return benchmark_main(args)


if __name__ == "__main__":
    sys.exit(main())
