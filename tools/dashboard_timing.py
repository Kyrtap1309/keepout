"""The measurement behind the dashboard benchmark: a synthetic store, and how
long the events list and the event page take to read it.

Usage::

    uv run python tools/dashboard_timing.py build STORE [--events N]
        [--cdms N] [--seed N]
    uv run python tools/dashboard_timing.py time STORE [--repeat N]

``build`` makes ``STORE`` a store of ``--events`` events (1 000) and
``--cdms`` CDMs (20 000) with their Pc cached (:func:`build_store`); it
prints how long that took. ``time`` reads it as the dashboard does, without
Streamlit (:func:`measure`), and prints one JSON object: the median wall
and CPU time [s] of the events list's first page and of the event page's
data path (``null`` when this keepout has no
``keepout.dashboard.event_data``).

``tools/dashboard_benchmark.py`` runs both, and its ``compare`` command
runs this file of the tree being tested with the Python environment of an
older tree, to time that tree's keepout under the same measurement. So this
file imports nothing from ``tools/`` and only names keepout has had since
its dashboard: ``keepout.cdm`` (``make_cdm``, ``ObjectMetadata``),
``keepout.conjunction.Conjunction``, ``keepout.objects.ObjectState``,
``keepout.state.StateVector``, ``keepout.covariance.Covariance``,
``keepout.orbits.EARTH_MU``, ``keepout.propagation.forces.EARTH_RADIUS``,
``keepout.risk`` (``assess``, ``PcObservation``,
``DEFAULT_DECISION_CONFIG``), ``keepout.settings.Settings``,
``keepout.store`` (``open_store``, ``CdmInput``, ``Store.ingest``,
``Store.cache``), ``keepout.dashboard.listing`` (``read_page``: the
events page's read) and ``keepout.dashboard.event_data`` (the event page's
data path, without Streamlit). A keepout without ``listing`` cannot be
timed; one without ``event_data`` is timed on the list only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypedDict

import numpy as np
from astropy import units as u
from astropy.time import Time

from keepout.cdm import ObjectMetadata, make_cdm
from keepout.conjunction import Conjunction
from keepout.covariance import Covariance
from keepout.objects import ObjectState
from keepout.orbits import EARTH_MU
from keepout.propagation.forces import EARTH_RADIUS
from keepout.risk import PcObservation
from keepout.settings import Settings
from keepout.state import StateVector

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from keepout.cdm import Cdm
    from keepout.store import CdmInput, Store

EVENTS: Final = 1000
"""Events of the benchmark's store unless told otherwise (spec §8)."""

CDMS: Final = 20_000
"""CDMs of the benchmark's store unless told otherwise (spec §8)."""

REPEAT: Final = 5
"""Timed reads of each path unless told otherwise (median)."""

START: Final = Time("2026-01-01T00:00:00", scale="utc")
"""The start of the year the events' TCAs fall in."""

AS_OF: Final = Time("2026-07-01T00:00:00", scale="utc")
"""The time the list is read at: half of the events are upcoming."""

PRIMARIES: Final = 20
"""Satellites of the fleet: each event is one of theirs."""

ISSUE_SPAN: Final = 5 * 86400.0
"""How long before TCA an event's first CDM is issued [s]."""

LAST_ISSUE: Final = 2 * 3600.0
"""How long before TCA an event's last CDM is issued [s]."""

_BATCH: Final = 2000
"""CDMs ingested per write transaction."""

_SETTINGS: Final = Settings()
"""The settings the Pc are cached and read with: keepout's defaults."""


@dataclass(frozen=True, slots=True)
class Timings:
    """The median times of the dashboard's reads [s].

    Parameters
    ----------
    list_s, list_cpu
        Wall and CPU time of the events list's first page.
    event_s, event_cpu
        Wall and CPU time of the event page's data path; ``None`` if this
        keepout has no ``keepout.dashboard.event_data``.
    """

    list_s: float
    list_cpu: float
    event_s: float | None
    event_cpu: float | None


class TimingOutput(TypedDict):
    """What ``time`` prints, as JSON: :class:`Timings`."""

    list_wall: float
    """Wall time of the events list's first page [s]."""
    list_cpu: float
    """Its CPU time [s]."""
    event_wall: float | None
    """Wall time of the event page's data path [s]; ``null`` without it."""
    event_cpu: float | None
    """Its CPU time [s]."""


def _object(
    object_id: str, epoch: Time, r: float, angle: float, offset: float
) -> ObjectState:
    """An object on a circular orbit of radius ``r`` inclined by ``angle``,
    ``offset`` [m] out along x at ``epoch``, its covariance 100 m and
    1 cm/s (GCRF) and its hard-body radius 5 m."""
    speed = math.sqrt(EARTH_MU / r)
    position = np.array([r + offset, 0.0, 0.0])
    velocity = speed * np.array([0.0, math.cos(angle), math.sin(angle)])
    covariance = np.diag([100.0**2] * 3 + [0.01**2] * 3)
    return ObjectState(
        object_id,
        StateVector(epoch, position, velocity, "GCRF"),
        Covariance(covariance, "GCRF", epoch),
        5.0,
    )


def _shares(events: int, cdms: int) -> list[int]:
    """The CDMs of each event: as even as can be, the first ones one more."""
    each, extra = divmod(cdms, events)
    return [each + (1 if index < extra else 0) for index in range(events)]


def _event_cdms(
    index: int, count: int, rng: np.random.Generator
) -> Iterator[tuple[Cdm, PcObservation]]:
    """The CDMs of event ``index`` and their Pc: a crossing of a fleet
    satellite and a secondary of its own, a TCA in the year from
    :data:`START`, ``count`` CDMs issued from :data:`ISSUE_SPAN` to
    :data:`LAST_ISSUE` before it, each with a miss and a Pc of its own
    (drawn, not computed: the reads timed never compute one)."""
    primary = str(40001 + index % PRIMARIES)
    secondary = str(60001 + index)
    tca = START + round(rng.uniform(0.0, 365.0 * 86400.0), 3) * u.s
    radius = EARTH_RADIUS + rng.uniform(400e3, 1200e3)
    angle = rng.uniform(0.2, 1.4)
    step = (ISSUE_SPAN - LAST_ISSUE) / max(count - 1, 1)
    metadata = ObjectMetadata(f"SAT-{primary}", f"2020-{index % PRIMARIES + 1:03d}A")
    other = ObjectMetadata(f"OBJ-{secondary}", f"2019-{index + 1:03d}B")
    for k in range(count):
        created = tca - (ISSUE_SPAN - k * step) * u.s
        # Successive CDMs move the TCA by a second or so (well within the
        # grouping tolerance) and the miss by hundreds of metres.
        epoch = tca + round(rng.normal(0.0, 1.0), 3) * u.s
        miss = rng.uniform(50.0, 2000.0)
        conjunction = Conjunction(
            _object(primary, epoch, radius, angle, 0.0),
            _object(secondary, epoch, radius, -angle, miss),
            10.0,
        )
        message_id = f"BENCH-{index:05d}-{k:02d}"
        cdm = make_cdm(
            conjunction, metadata, other, created=created, message_id=message_id
        )
        observation = PcObservation(
            time_to_tca=(epoch - created).to_value(u.s),
            source="keepout (foster)",
            pc=float(10.0 ** rng.uniform(-10.0, -3.0)),
            message_id=message_id,
            method="foster",
        )
        yield cdm, observation


def build_store(
    path: Path, *, events: int = EVENTS, cdms: int = CDMS, seed: int = 0
) -> Path:
    """A store of synthetic events, each CDM's Pc cached for keepout's
    default settings without computing it.

    Event ``i`` is a crossing of fleet satellite ``i mod 20`` with an
    object of its own, its TCA drawn from the year 2026; its CDMs (an even
    share of ``cdms``) are issued from 5 days to 2 hours before TCA. The
    store is the same for the same arguments (but for the times rows were
    written).

    Parameters
    ----------
    path
        The store's file: made, or added to if it is a store without these
        CDMs.
    events
        How many events.
    cdms
        How many CDMs: at least one per event.
    seed
        The seed of the random generator drawing TCAs, misses and Pc.

    Returns
    -------
    pathlib.Path
        ``path``.

    Raises
    ------
    ValueError
        If there are fewer CDMs than events.
    StoreError
        If ``path`` cannot be made a store or holds other CDMs of the same
        ``MESSAGE_ID``.
    """
    from keepout.store import CdmInput, open_store

    if events < 1 or cdms < events:
        raise ValueError(f"at least one CDM per event: {events} events, {cdms} CDMs")
    rng = np.random.default_rng(seed)
    with open_store(path) as store:
        inputs: list[CdmInput] = []
        observations: list[PcObservation] = []
        for index, count in enumerate(_shares(events, cdms)):
            for cdm, observation in _event_cdms(index, count, rng):
                inputs.append(CdmInput.of(cdm, "benchmark"))
                observations.append(observation)
            if len(inputs) >= _BATCH:
                _write(store, inputs, observations)
                inputs, observations = [], []
        _write(store, inputs, observations)
    return path


def _write(
    store: Store, inputs: Sequence[CdmInput], observations: Sequence[PcObservation]
) -> None:
    """Ingest the CDMs, then cache their Pc."""
    store.ingest(inputs)
    store.cache(_SETTINGS, observations)


def _list_page(path: Path) -> str | None:
    """The events list's first page at :data:`AS_OF`, read as the events
    page reads it; the id of its first event (``None`` if it is empty)."""
    from keepout.dashboard.listing import read_page

    page = read_page(path, _SETTINGS, as_of=AS_OF)
    return page.items[0].event_id if page.items else None


def _event_page(path: Path, event_id: str) -> None:
    """The event page's data path, as ``keepout/dashboard/views/event.py``
    takes it (:mod:`keepout.dashboard.event_data`), without Streamlit: the
    event read in one transaction and assessed, its Pc trend, and the
    encounter plane and geometry at TCA of its latest CDM, drawn with
    Plotly.

    Raises
    ------
    ValueError
        If the store has no such event, or a Pc is not cached.
    """
    from keepout.dashboard import event_data as data

    view = data.read_event(path, event_id, _SETTINGS)
    trend = None if view is None else data.trend_of(view, _SETTINGS)
    if view is None or trend is None:
        raise ValueError(f"event {event_id} is not stored with every Pc")
    data.trend_plot(trend)
    conjunction = data.latest_conjunction(view, _SETTINGS)
    max_pc = data.is_max_pc(view, conjunction)
    data.encounter_plot(conjunction, max_pc=max_pc)
    data.geometry_plot(data.geometry_of(conjunction), max_pc=max_pc)


def _timed(read: Callable[[], object], repeat: int) -> tuple[float, float]:
    """The median wall and CPU time [s] of ``repeat`` calls of ``read``,
    after one untimed call (as in the dashboard's long-running server, the
    imports and caches are warm)."""
    read()
    walls: list[float] = []
    cpus: list[float] = []
    for _ in range(repeat):
        wall, cpu = time.perf_counter(), time.process_time()
        read()
        cpus.append(time.process_time() - cpu)
        walls.append(time.perf_counter() - wall)
    return statistics.median(walls), statistics.median(cpus)


NO_DASHBOARD: Final = "no dashboard to time"
"""How :func:`main` starts its message when this keepout has no dashboard
(:func:`missing_dashboard`): ``compare`` skips such a base quietly."""


def missing_dashboard() -> str:
    """Why this keepout cannot be timed (it has no
    ``keepout.dashboard.listing``); empty if it can."""
    if importlib.util.find_spec("keepout.dashboard") is None:
        return f"{NO_DASHBOARD}: this keepout has no keepout.dashboard"
    if importlib.util.find_spec("keepout.dashboard.listing") is None:
        return f"{NO_DASHBOARD}: this keepout has no keepout.dashboard.listing"
    return ""


def has_event_page() -> bool:
    """Whether this keepout has the event page's data path without
    Streamlit (``keepout.dashboard.event_data``)."""
    return importlib.util.find_spec("keepout.dashboard.event_data") is not None


def measure(path: Path, *, repeat: int = REPEAT) -> Timings:
    """How long the dashboard's reads of the store at ``path`` take,
    without Streamlit: the events list's first page at :data:`AS_OF`
    (``keepout.dashboard.listing.read_page``) and the event page's data
    path of the page's first event (:func:`_event_page`), each store opened
    as the pages open it.

    Parameters
    ----------
    path
        A store, e.g. of :func:`build_store`.
    repeat
        Timed reads of each path, after an untimed one.

    Returns
    -------
    Timings
        The median times; the event page's ``None`` if this keepout has no
        ``keepout.dashboard.event_data``.

    Raises
    ------
    ValueError
        If the list's first page is empty.
    StoreError
        If the store cannot be read.
    """
    first = _list_page(path)
    if first is None:
        raise ValueError(f"{path}: no event upcoming at {AS_OF.isot}")
    list_s, list_cpu = _timed(lambda: _list_page(path), repeat)
    if not has_event_page():
        return Timings(list_s, list_cpu, None, None)
    event_s, event_cpu = _timed(lambda: _event_page(path, first), repeat)
    return Timings(list_s, list_cpu, event_s, event_cpu)


def output_of(timings: Timings) -> TimingOutput:
    """What ``time`` prints of ``timings``."""
    return {
        "list_wall": timings.list_s,
        "list_cpu": timings.list_cpu,
        "event_wall": timings.event_s,
        "event_cpu": timings.event_cpu,
    }


def _positive(text: str) -> int:
    """A positive integer argument.

    Raises
    ------
    argparse.ArgumentTypeError
        If ``text`` is not one.
    """
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        raise argparse.ArgumentTypeError(f"not a positive integer: {text!r}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """The entry point building a store or timing the reads of one.

    Returns
    -------
    int
        0; it exits with status 2 if this keepout has no dashboard to time
        (:func:`missing_dashboard`).
    """
    parser = argparse.ArgumentParser(prog="tools/dashboard_timing.py")
    commands = parser.add_subparsers(dest="command", required=True)
    building = commands.add_parser("build", help="make a benchmark store")
    building.add_argument("store", type=Path)
    building.add_argument("--events", type=_positive, default=EVENTS)
    building.add_argument("--cdms", type=_positive, default=CDMS)
    building.add_argument("--seed", type=int, default=0)
    timing = commands.add_parser("time", help="time the dashboard's reads")
    timing.add_argument("store", type=Path)
    timing.add_argument("--repeat", type=_positive, default=REPEAT)
    args = parser.parse_args(argv)
    missing = missing_dashboard()
    if missing:
        parser.exit(2, f"{parser.prog}: {missing}\n")
    if args.command == "build":
        start = time.perf_counter()
        build_store(args.store, events=args.events, cdms=args.cdms, seed=args.seed)
        elapsed = time.perf_counter() - start
        print(
            f"{args.store}: built, {args.events} events and {args.cdms} CDMs, "
            f"in {elapsed:.1f} s"
        )
        return 0
    print(json.dumps(output_of(measure(args.store, repeat=args.repeat))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
