"""Time one screening: the measurement behind the screening benchmark.

Usage::

    uv run python tools/screening_timing.py CATALOG --primaries N [N ...]
        --start TIME [--days DAYS] [--radius METRES] [--workers N]

Screens the primaries (catalog numbers of objects in ``CATALOG``, any file
:func:`keepout.omm.load` reads) against every object of ``CATALOG`` with
:func:`keepout.screening.screen` in a sphere of ``--radius`` (5 km by
default) over ``--days`` from ``--start``, and prints one JSON object: the
wall and CPU time of the screening [s] (reading the catalog excluded), the
number of approaches, of catalog objects screened and of objects read.

``tools/screening_benchmark.py`` measures with :func:`timed`, and its
``compare`` command runs this file of the tree being tested with the Python
environment of an older tree, to time that tree's keepout under the same
measurement. So this file imports nothing else from ``tools/`` and only
names keepout has had since it could screen: ``keepout.omm.load``,
``keepout.screening`` (``screen``, ``ScreeningResult``, ``ScreeningVolume``,
``VolumeRule``) and ``keepout.screening.volume.SOCRATES_RADIUS``. CPU time
comes from ``resource`` (Linux and macOS).
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from astropy.time import Time

from keepout._units import DAY
from keepout.omm import Omm, load
from keepout.screening import ScreeningResult, ScreeningVolume, VolumeRule, screen
from keepout.screening.volume import SOCRATES_RADIUS


@dataclass(frozen=True)
class Timing:
    """How long a screening took.

    Parameters
    ----------
    wall
        Elapsed time [s].
    cpu
        CPU time [s], user and system, of this process and of the worker
        processes it waited for.
    """

    wall: float
    cpu: float


class TimingOutput(TypedDict):
    """What :func:`main` prints, as JSON: the measurement."""

    wall: float
    """Wall time of the screening [s]."""
    cpu: float
    """CPU time of the screening [s] (:class:`Timing`)."""
    approaches: int
    """Approaches found."""
    screened: int
    """Catalog objects screened."""
    objects: int
    """Element sets read."""


def select(catalog: Sequence[Omm], numbers: Sequence[int]) -> list[Omm]:
    """The element sets of ``numbers``, in that order.

    Raises
    ------
    ValueError
        If a number is not in the catalog.
    """
    by_number = {omm.catalog_number: omm for omm in catalog}
    missing = [number for number in numbers if number not in by_number]
    if missing:
        raise ValueError(f"primaries not in the catalog: {missing}")
    return [by_number[number] for number in numbers]


def cpu_time() -> float:
    """CPU time used so far [s] by this process and its finished children."""
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def timed(
    primaries: Sequence[Omm],
    catalog: Sequence[Omm],
    start: Time,
    duration: float,
    volume: ScreeningVolume | VolumeRule,
    *,
    workers: int,
) -> tuple[ScreeningResult, Timing]:
    """Screen with :func:`keepout.screening.screen` and time it.

    The CPU time counts the worker processes, which ``screen`` has waited
    for when it returns.

    Parameters
    ----------
    primaries, catalog, start, duration, volume, workers
        As for :func:`keepout.screening.screen` (``duration`` in s).

    Returns
    -------
    tuple of ScreeningResult and Timing
        The result and how long it took.
    """
    wall, cpu = time.perf_counter(), cpu_time()
    result = screen(primaries, catalog, start, duration, volume, workers=workers)
    timing = Timing(wall=time.perf_counter() - wall, cpu=cpu_time() - cpu)
    return result, timing


def main(argv: Sequence[str] | None = None) -> int:
    """Time one screening and print the measurement as JSON.

    Returns
    -------
    int
        0.
    """
    parser = argparse.ArgumentParser(prog="tools/screening_timing.py")
    parser.add_argument("catalog", type=Path, help="element sets to screen against")
    parser.add_argument(
        "--primaries", type=int, nargs="+", required=True, help="catalog numbers"
    )
    parser.add_argument("--start", required=True, help="start of the window (UTC)")
    parser.add_argument("--days", type=float, default=7.0, help="window length")
    parser.add_argument(
        "--radius", type=float, default=SOCRATES_RADIUS, help="sphere [m]"
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)

    catalog = load(args.catalog)
    try:
        primaries = select(catalog, args.primaries)
    except ValueError as error:
        parser.error(f"{args.catalog}: {error}")
    result, timing = timed(
        primaries,
        catalog,
        Time(args.start, scale="utc"),
        args.days * DAY,
        ScreeningVolume.sphere(args.radius),
        workers=args.workers,
    )
    output: TimingOutput = {
        "wall": timing.wall,
        "cpu": timing.cpu,
        "approaches": len(result.approaches),
        "screened": result.catalog_count,
        "objects": len(catalog),
    }
    print(json.dumps(output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
