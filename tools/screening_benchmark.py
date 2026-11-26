"""A benchmark of screening on the frozen public catalog, and of two trees.

The benchmark (no command) screens :data:`PRIMARIES` (20 LEO satellites)
against a frozen catalog (``--source``: :data:`SOURCES`) for 7 days from 0 h
UTC of its retrieval day with each volume of :data:`VOLUMES`, and writes the
source's results file and README block: what each volume found (the data)
and the median wall and CPU time of the runs (the timings). ``--check``
screens once and fails when the data differs from the committed files; it
shows the committed timings, never checks them. The runs of a volume must
find the same, or the tool fails.

``compare`` is CI's regression check: this tree and ``--base-python`` (the
merge base's environment) each time ``tools/screening_timing.py`` on a
sample of the catalog (:func:`reduced_catalog`) in one process, alternately
in fresh processes with one thread for numerical libraries
(``tools/benchmark_runs.py``). It fails if this tree's median CPU time
exceeds the base's by more than ``--threshold``; a base that cannot run the
measurement is skipped (exit 0). Its Markdown summary goes to
``$GITHUB_STEP_SUMMARY`` when that is set.

Worker processes are spawned, so this module is only run as a script (or
imported), never from standard input. ``--help`` lists the options.
"""

from __future__ import annotations

import argparse
import gzip
import statistics
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Generic, Protocol, TypedDict, TypeVar

import screening_snapshot as snapshot
import spacetrack_snapshot
from astropy.time import Time
from benchmark_runs import (
    Comparison,
    alternate,
    cpu_cells,
    cpu_change,
    run_process,
    skipped_lines,
    verdict_line,
    write_step_summary,
)
from generated import (
    README,
    REPO,
    Machine,
    json_text,
    machine,
    machine_text,
    parse_json,
    positive,
    read_timings,
    readme_with,
    table,
    write_or_check,
)
from screening_timing import TimingOutput, select, timed

from keepout._text import isot, minute, plural
from keepout.omm.celestrak import dumps_csv
from keepout.screening import (
    DEFAULT_DURATION,
    DEFAULT_VOLUME,
    ScreeningResult,
    ScreeningVolume,
    SkipReason,
    VolumeRule,
)
from keepout.screening.volume import SOCRATES_RADIUS
from keepout.units import DAY, KM

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from keepout.omm import Omm

__all__ = [
    "Run",
    "main",
    "render_comparison",
    "run_timing",
]

TIMING: Final = Path(__file__).resolve().with_name("screening_timing.py")
TOOL: Final = "tools/screening_benchmark.py"


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

COMPARE_DIGITS: Final = 2
"""Decimals of the CPU times [s] of ``compare``'s table."""

_Manifest = TypeVar("_Manifest")


class Settings(TypedDict):
    """What the benchmark screens."""

    source: str
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


class Timings(Machine):
    """The timings and where they were measured: shown, never checked."""

    runs: int
    volumes: list[Measured]


class Summary(TypedDict):
    """The benchmark's results, as written to JSON."""

    settings: Settings
    found: list[Found]
    timings: Timings | None


def _days(count: float) -> str:
    return "1 day" if count == 1.0 else f"{count:g} days"


def _processes(workers: int) -> str:
    return "in one process" if workers == 1 else f"with {workers} worker processes"


class Snapshot(Protocol[_Manifest]):
    """A snapshot tool, :mod:`screening_snapshot` or
    :mod:`spacetrack_snapshot`: how the benchmark reads its frozen catalog."""

    def read_manifest(self, data: Path) -> _Manifest:
        """The manifest in ``data``."""
        ...

    def read_catalog(self, data: Path) -> tuple[Omm, ...]:
        """The catalog in ``data``."""
        ...

    def retrieval_day(self, manifest: _Manifest) -> str:
        """The day the catalog was retrieved, UTC, ISO 8601."""
        ...


@dataclass(frozen=True, slots=True)
class Source(Generic[_Manifest]):
    """A frozen catalog the benchmark screens.

    Parameters
    ----------
    name
        Its ``--source`` name.
    catalog, provider
        What the README block calls it: the frozen ``catalog`` catalog
        (``provider``, retrieved ...).
    data
        Its directory (the default of ``--data``).
    snapshot
        The tool that froze it, which reads it.
    """

    name: str
    catalog: str
    provider: str
    data: Path
    snapshot: Snapshot[_Manifest]

    def read(self, data: Path) -> tuple[tuple[Omm, ...], str]:
        """The element sets in the directory ``data`` and the day they were
        retrieved (UTC, ISO).

        Raises
        ------
        ValueError
            If its manifest is not one the snapshot tool writes.
        """
        manifest = self.snapshot.read_manifest(data)
        return self.snapshot.read_catalog(data), self.snapshot.retrieval_day(manifest)

    @property
    def stem(self) -> str:
        """The name of its results file and README block:
        ``screening-benchmark``, and the source's name unless it is the
        default."""
        if self.name == DEFAULT_SOURCE:
            return "screening-benchmark"
        return f"screening-benchmark-{self.name}"

    @property
    def results(self) -> Path:
        """Its results file (the default of ``--results``)."""
        return REPO / "docs" / f"{self.stem}.json"

    @property
    def start(self) -> str:
        """The start marker of its README block."""
        return f"<!-- {self.stem}-summary:start -->"

    @property
    def end(self) -> str:
        """The end marker of its README block."""
        return f"<!-- {self.stem}-summary:end -->"

    @property
    def command(self) -> str:
        """The command that writes this source's files."""
        return TOOL if self.name == DEFAULT_SOURCE else f"{TOOL} --source {self.name}"


DEFAULT_SOURCE: Final = "celestrak"
"""The source benchmarked unless told otherwise."""

SOURCES: Final[
    Mapping[str, Source[snapshot.Manifest] | Source[spacetrack_snapshot.Manifest]]
] = {
    source.name: source
    for source in (
        Source(
            name="celestrak",
            catalog="public",
            provider="CelesTrak",
            data=snapshot.DATA,
            snapshot=snapshot,
        ),
        Source(
            name="spacetrack",
            catalog="full",
            provider="Space-Track",
            data=spacetrack_snapshot.DATA,
            snapshot=spacetrack_snapshot,
        ),
    )
}
"""The frozen catalogs the benchmark can screen, by ``--source`` name."""


def retrieval_start(retrieved: str) -> Time:
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
    """What each volume found, and its median times to the millisecond over
    ``repeat`` screenings.

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
                "wall": round(statistics.median(timing.wall for _, timing in runs), 3),
                "cpu": round(statistics.median(timing.cpu for _, timing in runs), 3),
            }
        )
    return data, timings


def _seconds_text(value: float) -> str:
    return f"{value:.1f} s"


def render_summary(summary: Summary) -> str:
    """The README block."""
    settings = summary["settings"]
    source = SOURCES[settings["source"]]
    timings = summary["timings"]
    measured = {} if timings is None else {m["volume"]: m for m in timings["volumes"]}
    rows = []
    for numbers in summary["found"]:
        skipped = ", ".join(f"{n} {reason}" for reason, n in numbers["skipped"].items())
        times = measured.get(numbers["volume"])
        rows.append(
            (
                numbers["volume"],
                numbers["screened"],
                skipped or "-",
                numbers["approaches"],
                "-" if times is None else _seconds_text(times["wall"]),
                "-" if times is None else _seconds_text(times["cpu"]),
            )
        )
    if timings is None:
        machine = "Not measured yet."
    else:
        runs = "1 run" if timings["runs"] == 1 else f"median of {timings['runs']} runs"
        machine = (
            f"Times: {runs} on {machine_text(timings)}; "
            "`--check` checks the counts, not the times."
        )
    return "\n".join(
        [
            (
                f"Screening {len(settings['primaries'])} LEO primaries against the "
                f"{settings['objects']} element sets of the frozen {source.catalog} "
                f"catalog ({source.provider}, retrieved {settings['retrieved']}) for "
                f"{_days(settings['days'])} from {settings['start']} UTC, "
                f"{_processes(settings['workers'])} (`{source.command}`):"
            ),
            "",
            *table(
                (
                    "Volume",
                    "Screened",
                    "Left out",
                    "Approaches",
                    "Wall time",
                    "CPU time",
                ),
                rows,
            ),
            "",
            machine,
        ]
    )


@dataclass(frozen=True, slots=True)
class BenchmarkOptions:
    """What the benchmark command is asked: its command-line options.

    Parameters
    ----------
    source
        The frozen catalog screened.
    data
        Its directory.
    primaries
        The primaries' catalog numbers.
    results
        The results file.
    readme
        The README holding the source's block.
    workers
        Processes to screen with.
    repeat
        Runs per volume (1 with ``check``).
    check
        Fail if a file is out of date instead of writing.
    """

    source: Source[snapshot.Manifest] | Source[spacetrack_snapshot.Manifest]
    data: Path
    primaries: tuple[int, ...]
    results: Path
    readme: Path
    workers: int
    repeat: int
    check: bool


def benchmark_main(options: BenchmarkOptions) -> int:
    """The benchmark command (or with ``--check`` the verification of its data).

    Returns
    -------
    int
        1 if, with ``--check``, a file is out of date, else 0.
    """
    source = options.source
    catalog, retrieved = source.read(options.data)
    start = retrieval_start(retrieved)
    primaries = select(catalog, options.primaries)
    data, measured = benchmark(
        primaries,
        catalog,
        start,
        workers=options.workers,
        repeat=1 if options.check else options.repeat,
    )
    timings: Timings | None
    if options.check:
        timings = read_timings(options.results, Timings)
    else:
        timings = {**machine(), "runs": options.repeat, "volumes": measured}
    summary: Summary = {
        "settings": {
            "source": source.name,
            "primaries": list(options.primaries),
            "objects": len(catalog),
            "retrieved": retrieved,
            "start": minute(start),
            "days": DEFAULT_DURATION / DAY,
            "workers": options.workers,
        },
        "found": data,
        "timings": timings,
    }
    files = {
        options.results: json_text(summary),
        options.readme: readme_with(
            options.readme, source.start, source.end, render_summary(summary)
        ),
    }
    return write_or_check(files, check=options.check, tool=source.command)


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


@dataclass(frozen=True, slots=True)
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


def run_timing(command: Sequence[str]) -> Run:
    """The result read from a measurement process run as ``command``.

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
    return Run(wall=output["wall"], cpu=output["cpu"], approaches=output["approaches"])


def render_comparison(
    comparison: Comparison[Run], setting: str, threshold: float
) -> tuple[str, bool]:
    """The comparison as Markdown, and whether it is a regression."""
    lines = [f"Screening benchmark: {setting}.", ""]
    if comparison.skipped:
        lines += skipped_lines(comparison.skipped)
        return "\n".join(lines), False
    rows: list[tuple[str, ...]] = []
    for name, runs in (("base", comparison.base), ("this", comparison.head)):
        approaches = sorted({run.approaches for run in runs})
        rows.append(
            (
                name,
                *cpu_cells([run.cpu for run in runs], COMPARE_DIGITS),
                f"{statistics.median(run.wall for run in runs):.2f} s",
                ", ".join(map(str, approaches)),
            )
        )
    header = (
        "Tree",
        "Median CPU time",
        "CPU time range",
        "Median wall time",
        "Approaches",
    )
    lines += table(header, rows)
    change = cpu_change(
        (run.cpu for run in comparison.head), (run.cpu for run in comparison.base)
    )
    text, regression = verdict_line("CPU time change:", change, threshold)
    lines += ["", text]
    return "\n".join(lines), regression


@dataclass(frozen=True, slots=True)
class CompareOptions:
    """What ``compare`` is asked: its command-line options.

    Parameters
    ----------
    source
        The frozen catalog sampled.
    data
        Its directory.
    primaries
        The primaries' catalog numbers.
    base_python
        The Python of the base tree's environment.
    every
        The catalog sample: every this-th object.
    days
        The screening's length.
    repeat
        Runs per tree.
    threshold
        The largest CPU time change that is not a regression.
    """

    source: Source[snapshot.Manifest] | Source[spacetrack_snapshot.Manifest]
    data: Path
    primaries: tuple[int, ...]
    base_python: str
    every: int
    days: float
    repeat: int
    threshold: float


def compare_main(options: CompareOptions) -> int:
    """The comparison of this tree's screening speed with another tree's.

    Returns
    -------
    int
        1 if this tree is slower than the threshold allows, else 0.
    """
    catalog, retrieved = options.source.read(options.data)
    select(catalog, options.primaries)  # all there, or ValueError
    reduced = reduced_catalog(catalog, options.primaries, options.every)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "catalog.csv.gz"
        path.write_bytes(gzip.compress(dumps_csv(reduced).encode("utf-8")))
        arguments = [
            str(TIMING),
            str(path),
            "--primaries",
            *map(str, options.primaries),
            "--start",
            isot(retrieval_start(retrieved)),
            "--days",
            f"{options.days:g}",
            "--radius",
            f"{SOCRATES_RADIUS:g}",
            "--workers",
            "1",
        ]
        comparison = alternate(
            [sys.executable, *arguments],
            [options.base_python, *arguments],
            repeat=options.repeat,
            run=run_timing,
        )
    setting = (
        f"{len(options.primaries)} primaries x {len(reduced)} objects x "
        f"{_days(options.days)}, {SOCRATES_RADIUS / KM:g} km sphere, 1 process, "
        f"{plural(options.repeat, 'run')} per tree"
    )
    text, regression = render_comparison(comparison, setting, options.threshold)
    print(text)
    write_step_summary(text)
    return 1 if regression else 0


def main(argv: Sequence[str] | None = None) -> int:
    """The entry point: the benchmark, or ``compare`` if that is the command.

    Returns
    -------
    int
        0 on success; 1 if, with ``--check``, a file is out of date, or if
        ``compare`` finds a regression.
    """
    data = argparse.ArgumentParser(add_help=False)
    data.add_argument("--data", type=Path, help="default: the source's")
    data.add_argument(
        "--primaries",
        type=int,
        nargs="+",
        default=list(PRIMARIES),
        help=argparse.SUPPRESS,
    )
    parser = argparse.ArgumentParser(prog=TOOL, parents=[data])
    parser.add_argument(
        "--source",
        choices=list(SOURCES),
        default=DEFAULT_SOURCE,
        help="the frozen catalog screened",
    )
    parser.add_argument("--check", action="store_true", help="fail if out of date")
    parser.add_argument("--results", type=Path, help="default: the source's")
    parser.add_argument("--readme", type=Path, default=README)
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
    source = SOURCES[args.source]
    data_dir: Path = args.data or source.data
    primaries: tuple[int, ...] = tuple(args.primaries)
    if args.command == "compare":
        return compare_main(
            CompareOptions(
                source=source,
                data=data_dir,
                primaries=primaries,
                base_python=args.base_python,
                every=args.every,
                days=args.days,
                repeat=args.repeat,
                threshold=args.threshold,
            )
        )
    return benchmark_main(
        BenchmarkOptions(
            source=source,
            data=data_dir,
            primaries=primaries,
            results=args.results or source.results,
            readme=args.readme,
            workers=args.workers,
            repeat=args.repeat,
            check=args.check,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
