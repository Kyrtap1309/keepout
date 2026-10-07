"""Close approaches of primaries with a catalog: a sieve on a time grid, then TCA.

:class:`ApproachSearch` finds every local minimum of the distance (range)
between a primary and a catalog object that lies inside the primary's
screening volume over a window. Both objects are sampled on a uniform time
grid (:func:`time_grid`, step ``h``); two sieves rule out the grid intervals
in which the pair cannot come inside the volume; in the others, the sign
changes of ``f = rho . drho/dt`` (``rho`` the position of the catalog object
relative to the primary, ``f`` half the derivative of ``|rho|^2``) bracket
the minima, and :func:`scipy.optimize.brentq` finds each time of closest
approach (TCA) as the root of ``f``.

**Sieves.** For an interval ``[ta, tb]`` of length ``h`` with relative
positions ``rho_a`` and ``rho_b`` at its ends, and ``R`` the bounding radius
of the volume (:attr:`~keepout.screening.volume.ScreeningVolume.bounding_radius`;
no point of the volume is farther):

1. *Speed sieve.* Each object's speed is below :func:`speed_bounds` (the
   speed at perigee of the most eccentric orbit inside the object's radius
   bounds, plus :data:`SPEED_MARGIN`), so the relative speed is below ``V =
   v_p + v_s`` and every point of the interval is within ``V (t - ta)`` of
   ``rho_a`` and ``V (tb - t)`` of ``rho_b``. A point inside ``R`` needs
   ``|rho_a| + |rho_b| <= 2 R + V h``, and the range never exceeds
   ``rho_up = (|rho_a| + |rho_b| + V h) / 2``.
2. *Chord sieve.* With ``A`` a bound on the relative acceleration, the
   relative position is within ``A (t - ta) (tb - t) / 2 <= A h^2 / 8`` of
   the chord from ``rho_a`` to ``rho_b``; a point inside ``R`` needs the
   chord to pass within ``R + A h^2 / 8`` of the primary. ``A`` is the
   difference of the two objects' gravity, at most the gravity gradient
   (``2 mu / r^3`` for a point mass, times :data:`GRAVITY_MARGIN` for the
   zonal terms) times the largest range over the interval, ``r`` being the
   lowest radius on the segment joining the objects (from their radius
   bounds and ``rho_up``); or at most the sum of both gravities; plus twice
   :data:`PERTURBATION`, a bound on every other acceleration (drag, the Sun
   and Moon, SGP4's own terms). The largest range is the smaller of
   ``rho_up`` and the chord's own bound, ``max(|rho_a|, |rho_b|) + A h^2 /
   8``, solved for the range.

Both sieves hold for any step, so the step only trades propagation (one
state per object per step) against the work on the intervals that pass.

**Minima.** ``f`` is computed from positions only, by a central difference
over :data:`DERIVATIVE_STEP`: SGP4's velocity is not exactly the derivative
of its position (by up to 12 m/s), which would move the TCA of a slow pair
by seconds. In an interval that passes the sieves, ``f`` is increasing where
``f' = |drho/dt|^2 + rho . d2rho/dt2 > 0``. The relative velocity stays
within ``A h / 2`` of the chord velocity ``c = (rho_b - rho_a) / h``, so
this holds over the whole interval when ``(|c| - A h / 2)^2 > rho_max A``
(``rho_max`` the largest range over the interval); then the interval holds
a minimum exactly when ``f`` goes from negative at ``ta`` to non-negative at
``tb``. Over a week of the public catalog (20 LEO primaries, 5 km sphere,
default step) every interval in which the pair moves faster than 0.5 km/s
is decided this way. Any other interval (a slow, co-orbital pair) is
resampled every :data:`SLOW_STEP` and each sign change of ``f`` between the
samples is a minimum: there, a minimum and a maximum of the range less than
:data:`SLOW_STEP` apart are not told apart (on the frozen fixture the
closest are 75 s apart).

**Inside the volume.** A minimum counts when the catalog object's position
relative to the primary at TCA, in the primary's RTN frame, is inside the
volume (:meth:`~keepout.screening.volume.ScreeningVolume.contains_position`).
For an ellipsoid, a pass that crosses the volume away from its point of
closest approach does not count.

**Window.** A minimum counts when its TCA lies in the window ``(start,
start + duration]``: a range still falling at the end of the window or
already rising at its start has its minimum outside, so consecutive windows
together find the minima of their union, each once.

**Same object.** A pair at exactly the same position at both ends of an
interval is taken as one object over it, with no minimum: CelesTrak gives
docked spacecraft the element set of their station.

**Determinism.** Each interval is decided from the states of its own pair
only, so the result does not depend on how the grid is cut into chunks or
into runs (:meth:`ApproachSearch.run` over parts of the grid, e.g. one per
worker, merged with :meth:`SearchResult.merge`).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import numpy as np
import numpy.typing as npt
from astropy.time import Time
from scipy.optimize import brentq

from keepout._types import (
    BoolArray,
    FloatArray,
    IndexArray,
    read_only,
    require_non_negative,
    require_positive,
)
from keepout.orbits import EARTH_MU
from keepout.propagation.base import CHUNK_STATES, NO_ERROR, Propagator, RadiusBounds
from keepout.screening.volume import ScreeningVolume

__all__ = [
    "DEFAULT_STEP",
    "DERIVATIVE_STEP",
    "GRAVITY_MARGIN",
    "MIN_RADIUS",
    "PERTURBATION",
    "SLOW_STEP",
    "SPEED_MARGIN",
    "TCA_TOLERANCE",
    "ApproachSearch",
    "ObjectFailure",
    "RangeMinimum",
    "SearchResult",
    "speed_bounds",
    "time_grid",
]

DEFAULT_STEP: Final = 120.0
"""Longest time between grid samples [s] unless told otherwise."""

SLOW_STEP: Final = 10.0
"""Longest time between the samples of an interval whose minima the grid
cannot decide (a slow pair) [s]."""

DERIVATIVE_STEP: Final = 1.0
"""Step of the central difference that gives ``rho . drho/dt`` [s]: four
positions at -2, -1, +1 and +2 steps. On SGP4 the error is below 1e-4
m^2/s, mostly SGP4's own rounding."""

TCA_TOLERANCE: Final = 1e-6
"""Absolute tolerance of the TCA [s] given to Brent's method."""

SPEED_MARGIN: Final = 0.05
"""Relative margin on :func:`speed_bounds`. SGP4's speed exceeds the
two-body bound by 0.034 % at most on the public catalog (its short-period
terms move the osculating semi-major axis)."""

GRAVITY_MARGIN: Final = 1.05
"""Factor on the point-mass gravity and its gradient for the zonal terms
(J2 adds less than 1 %) and the gravitational parameters of other models."""

PERTURBATION: Final = 1.0
"""Bound on each object's acceleration beyond the point mass and J2 [m/s^2].
On the public catalog SGP4 stays below 0.032 m/s^2 over the week after its
download and 0.2 m/s^2 up to 30 days after (drag of objects with a large or
negative B*); it exceeds 1 m/s^2 only long before an element set's epoch or
just before SGP4 fails."""

MIN_RADIUS: Final = 6_356_752.0
"""Lowest distance from the Earth's centre assumed for a state without an
error [m]: the Earth's polar radius (WGS 84). SGP4 reports error 6 below
6378.135 km."""

_STENCIL: Final = np.array([-2.0, -1.0, 1.0, 2.0]) * DERIVATIVE_STEP
_WEIGHTS: Final = np.array([1.0, -8.0, 8.0, -1.0]) / (24.0 * DERIVATIVE_STEP)
"""Fourth-order central difference of ``|rho|^2``, halved: ``f``."""


def time_grid(duration: float, step: float = DEFAULT_STEP) -> FloatArray:
    """The uniform grid of a window: its start, its end, at most ``step`` apart.

    Parameters
    ----------
    duration
        Length of the window [s], finite and >= 0.
    step
        Longest time between samples [s], finite and > 0.

    Returns
    -------
    FloatArray
        Seconds since the start of the window, ``ceil(duration / step) + 1``
        samples evenly spaced from 0 to ``duration`` (one sample, 0, for an
        empty window); read-only.

    Raises
    ------
    ValueError
        If ``duration`` or ``step`` is out of range.
    """
    require_non_negative(duration, "duration")
    require_positive(step, "step")
    count = math.ceil(duration / step) + 1 if duration > 0.0 else 1
    grid = np.linspace(0.0, duration, count)
    read_only(grid)
    return grid


def speed_bounds(bounds: RadiusBounds) -> FloatArray:
    """Bounds on each object's speed from bounds on its distance to the Earth.

    The fastest an object goes, with its distance from the Earth's centre in
    ``[lowest, highest]``, is at the perigee of the orbit from ``lowest`` to
    ``highest``: ``sqrt(2 mu highest / (lowest (lowest + highest)))`` (vis
    viva), the escape speed at ``lowest`` when ``highest`` is infinite. The
    lowest radius is taken at least :data:`MIN_RADIUS`, and the result is
    widened by :data:`SPEED_MARGIN` and :data:`GRAVITY_MARGIN`.

    Parameters
    ----------
    bounds
        Radius bounds of N objects over a span.

    Returns
    -------
    FloatArray
        Speed bounds [m/s], shape ``(N,)``.
    """
    lowest = np.maximum(bounds.lowest, MIN_RADIUS)
    highest = np.maximum(bounds.highest, lowest)
    # highest / (lowest + highest), written to give 1 for an infinite highest.
    share = 1.0 / (1.0 + lowest / highest)
    speeds: FloatArray = np.sqrt(2.0 * EARTH_MU * GRAVITY_MARGIN * share / lowest)
    return speeds * (1.0 + SPEED_MARGIN)


@dataclass(frozen=True)
class ObjectFailure:
    """The earliest failed state of an object seen by a search.

    Parameters
    ----------
    offset
        Time of the state [s since the start of the window].
    code
        Its error code (:data:`keepout.propagation.ERROR_MESSAGES`).
    """

    offset: float
    code: int


def _vector(value: npt.ArrayLike, name: str) -> FloatArray:
    array = np.array(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}")
    read_only(array)
    return array


@dataclass(frozen=True, eq=False)
class RangeMinimum:
    """A local minimum of a pair's range, inside the primary's volume.

    Parameters
    ----------
    primary
        Position of the primary among the search's primaries.
    secondary
        Position of the other object in the search's catalog.
    offset
        Time of closest approach [s since the start of the window].
    distance
        Range at TCA [m].
    r_primary, v_primary
        The primary's position [m] and velocity [m/s] at TCA, in the
        propagators' frame; read-only copies, shape ``(3,)``.
    r_secondary, v_secondary
        The other object's state at TCA, likewise.

    Raises
    ------
    ValueError
        If a state vector does not have shape ``(3,)``.
    """

    primary: int
    secondary: int
    offset: float
    distance: float
    r_primary: FloatArray
    v_primary: FloatArray
    r_secondary: FloatArray
    v_secondary: FloatArray

    def __post_init__(self) -> None:
        for name in ("r_primary", "v_primary", "r_secondary", "v_secondary"):
            object.__setattr__(self, name, _vector(getattr(self, name), name))


def _order(minimum: RangeMinimum) -> tuple[int, int, float]:
    return minimum.primary, minimum.secondary, minimum.offset


def _earliest(
    failures: dict[int, ObjectFailure], index: int, failure: ObjectFailure
) -> None:
    """Keep the earlier of an object's recorded failure and a new one."""
    known = failures.get(index)
    if known is None or (failure.offset, failure.code) < (known.offset, known.code):
        failures[index] = failure


@dataclass(frozen=True, eq=False)
class SearchResult:
    """What a search finds over its part of the grid.

    Parameters
    ----------
    minima
        The minima inside the volumes; kept sorted by primary, secondary and
        offset.
    primary_failures, catalog_failures
        The earliest failed state seen of each primary and catalog object,
        by position (read-only mappings, sorted by position). Screening
        skips these objects: minima of a failed object are still listed
        here, from the intervals in which both objects were propagated.
    """

    minima: tuple[RangeMinimum, ...]
    primary_failures: Mapping[int, ObjectFailure]
    catalog_failures: Mapping[int, ObjectFailure]

    def __post_init__(self) -> None:
        object.__setattr__(self, "minima", tuple(sorted(self.minima, key=_order)))
        for name in ("primary_failures", "catalog_failures"):
            failures = dict(sorted(getattr(self, name).items()))
            object.__setattr__(self, name, MappingProxyType(failures))

    @classmethod
    def merge(cls, results: Iterable[SearchResult]) -> SearchResult:
        """One result from the results of runs over parts of a grid.

        Merging the runs over consecutive parts gives exactly the result of
        one run over the whole.
        """
        minima: list[RangeMinimum] = []
        primary: dict[int, ObjectFailure] = {}
        catalog: dict[int, ObjectFailure] = {}
        for result in results:
            minima.extend(result.minima)
            for index, failure in result.primary_failures.items():
                _earliest(primary, index, failure)
            for index, failure in result.catalog_failures.items():
                _earliest(catalog, index, failure)
        return cls(tuple(minima), primary, catalog)


def _dot(a: FloatArray, b: FloatArray) -> FloatArray:
    """Dot products along the last axis, element by element: the same bits
    whatever the shape of the arrays (a reduction might not)."""
    product: FloatArray = (
        a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]
    )
    return product


def _acceleration(
    largest: FloatArray, farthest: FloatArray, lowest: FloatArray, length: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Bounds on the relative acceleration and the range over intervals.

    ``largest`` is the larger range at the two ends of each interval,
    ``farthest`` the speed sieve's bound on the range over it, ``lowest``
    the lowest radius of either object and ``length`` the interval's
    length. Returns the bound ``A`` on the relative acceleration and a bound
    on the range: the range stays within ``A h^2 / 8`` of the chord, so it
    is at most ``largest + A h^2 / 8``, with ``A`` itself growing with it.
    """
    mu = EARTH_MU * GRAVITY_MARGIN
    both = 2.0 * mu / (lowest * lowest)
    inner = lowest * lowest - 0.25 * farthest * farthest
    gradient = np.full_like(farthest, np.inf)
    segment = inner > 0.0
    gradient[segment] = 2.0 * mu / inner[segment] ** 1.5
    widest = farthest.copy()
    shrink = 1.0 - gradient * length * length / 8.0
    solvable = shrink > 0.0
    widest[solvable] = np.minimum(
        farthest[solvable],
        (largest[solvable] + PERTURBATION * length[solvable] ** 2 / 4.0)
        / shrink[solvable],
    )
    acceleration: FloatArray = np.minimum(both, gradient * widest) + 2.0 * PERTURBATION
    return acceleration, widest


class _Failed(Exception):
    """Propagation failed while refining a pair."""

    def __init__(
        self, primary: ObjectFailure | None, secondary: ObjectFailure | None
    ) -> None:
        super().__init__()
        self.primary = primary
        self.secondary = secondary


def _first_failure(
    errors: npt.NDArray[np.uint8], stamps: FloatArray
) -> ObjectFailure | None:
    bad = np.flatnonzero(errors != NO_ERROR)
    if bad.size == 0:
        return None
    first = bad[np.argmin(stamps[bad])]
    return ObjectFailure(float(stamps[first]), int(errors[first]))


class _Pair:
    """One primary and one catalog object, evaluated at any times."""

    def __init__(self, search: ApproachSearch, primary: int, secondary: int) -> None:
        self.start = search.start
        self.primary = search.primaries.subset(np.array([primary]))
        self.secondary = search.catalog.subset(np.array([secondary]))
        self.known: dict[float, float] = {}

    def states(
        self, stamps: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Positions and velocities of both at ``stamps``, ``(n, 3)`` each."""
        own = self.primary.propagate(self.start, stamps)
        other = self.secondary.propagate(self.start, stamps)
        failed = (
            _first_failure(own.errors[0], stamps),
            _first_failure(other.errors[0], stamps),
        )
        if failed != (None, None):
            raise _Failed(*failed)
        return own.r[0], own.v[0], other.r[0], other.v[0]

    def derivative(self, times: FloatArray) -> FloatArray:
        """``rho . drho/dt`` at ``times`` from positions only."""
        stamps = (times[:, None] + _STENCIL).reshape(-1)
        r_own, _, r_other, _ = self.states(stamps)
        rho = r_other - r_own
        squared = _dot(rho, rho).reshape(-1, len(_STENCIL))
        derivative: FloatArray = (
            squared[:, 0] * _WEIGHTS[0]
            + squared[:, 1] * _WEIGHTS[1]
            + squared[:, 2] * _WEIGHTS[2]
            + squared[:, 3] * _WEIGHTS[3]
        )
        return derivative

    def at(self, time: float) -> float:
        """``rho . drho/dt`` at one time, for Brent's method (which starts
        at the ends of its bracket, already known)."""
        known = self.known.get(time)
        if known is not None:
            return known
        return float(self.derivative(np.array([time]))[0])


def _samples(first: float, last: float, *, quick: bool) -> FloatArray:
    """Where ``f`` is evaluated in an interval: its ends, or every SLOW_STEP."""
    if quick:
        return np.array([first, last])
    count = math.ceil((last - first) / SLOW_STEP)
    samples = first + (last - first) * (np.arange(count + 1) / count)
    samples[-1] = last
    return samples


def _record(
    failures: dict[int, ObjectFailure],
    positions: IndexArray,
    errors: npt.NDArray[np.uint8],
    times: FloatArray,
) -> None:
    """Record the first failed state of each object of a chunk (in time
    order)."""
    failed = errors != NO_ERROR
    for row in np.flatnonzero(np.any(failed, axis=1)):
        first = int(np.argmax(failed[row]))
        failure = ObjectFailure(float(times[first]), int(errors[row, first]))
        _earliest(failures, int(positions[row]), failure)


class ApproachSearch:
    """The minima of the range of primary-catalog pairs inside the volumes.

    See the module's description for the method and what it guarantees.
    A search pickles (to be run in worker processes) and can run over any
    part of its grid; the runs over consecutive parts, merged
    (:meth:`SearchResult.merge`), give exactly the result of one run.

    Parameters
    ----------
    primaries
        The P primaries.
    catalog
        The S catalog objects, in the frame of ``primaries``.
    start
        Start of the window; one time.
    duration
        Length of the window [s], finite and >= 0.
    volumes
        The volume of every primary, or one volume per primary (in order).
    pairs
        Which pairs to search, boolean ``(P, S)``: for example the
        apogee/perigee filter's result, without a primary paired with
        itself.
    step
        Longest time between grid samples [s] (:func:`time_grid`).
    primary_bounds, catalog_bounds
        The objects' radius bounds over the window, if already known;
        computed from the propagators otherwise.

    Raises
    ------
    ValueError
        If the propagators' frames differ, ``start`` is not one time, the
        window, step, volumes, pairs or bounds do not fit.
    """

    def __init__(
        self,
        primaries: Propagator,
        catalog: Propagator,
        start: Time,
        duration: float,
        volumes: ScreeningVolume | Sequence[ScreeningVolume],
        pairs: npt.ArrayLike,
        *,
        step: float = DEFAULT_STEP,
        primary_bounds: RadiusBounds | None = None,
        catalog_bounds: RadiusBounds | None = None,
    ) -> None:
        if primaries.frame != catalog.frame:
            raise ValueError(
                f"primaries and catalog must share a frame, got {primaries.frame} "
                f"and {catalog.frame}"
            )
        if not start.isscalar:
            raise ValueError(f"start must be a single time, got shape {start.shape}")
        self._offsets = time_grid(duration, step)
        mask = np.array(pairs)
        if mask.dtype != np.bool_ or mask.shape != (len(primaries), len(catalog)):
            raise ValueError(
                f"pairs must be booleans of shape ({len(primaries)}, {len(catalog)}), "
                f"got {mask.dtype} {mask.shape}"
            )
        read_only(mask)
        if isinstance(volumes, ScreeningVolume):
            volumes = (volumes,) * len(primaries)
        if len(volumes) != len(primaries):
            raise ValueError(
                f"{len(volumes)} volumes given for {len(primaries)} primaries"
            )
        if primary_bounds is None:
            primary_bounds = primaries.radius_bounds(start, duration)
        if catalog_bounds is None:
            catalog_bounds = catalog.radius_bounds(start, duration)
        if len(primary_bounds) != len(primaries) or len(catalog_bounds) != len(catalog):
            raise ValueError(
                f"radius bounds of {len(primary_bounds)} primaries and "
                f"{len(catalog_bounds)} catalog objects given for {len(primaries)} "
                f"and {len(catalog)}"
            )
        self._primaries = primaries
        self._catalog = catalog
        self._start = start
        self._duration = float(duration)
        self._step = float(step)
        self._volumes = tuple(volumes)
        self._pairs = mask
        self._speed_primary = speed_bounds(primary_bounds)
        self._speed_catalog = speed_bounds(catalog_bounds)
        self._lowest_primary = np.maximum(primary_bounds.lowest, MIN_RADIUS)
        self._lowest_catalog = np.maximum(catalog_bounds.lowest, MIN_RADIUS)
        self._reach = np.array([volume.bounding_radius for volume in self._volumes])

    @property
    def primaries(self) -> Propagator:
        """The primaries."""
        return self._primaries

    @property
    def catalog(self) -> Propagator:
        """The catalog."""
        return self._catalog

    @property
    def start(self) -> Time:
        """Start of the window."""
        return self._start

    @property
    def duration(self) -> float:
        """Length of the window [s]."""
        return self._duration

    @property
    def step(self) -> float:
        """Longest time between grid samples [s]."""
        return self._step

    @property
    def volumes(self) -> tuple[ScreeningVolume, ...]:
        """The volume of each primary."""
        return self._volumes

    @property
    def pairs(self) -> BoolArray:
        """The pairs searched, ``(P, S)``, read-only."""
        return self._pairs

    @property
    def offsets(self) -> FloatArray:
        """The grid [s since the start], read-only."""
        return self._offsets

    @property
    def intervals(self) -> int:
        """Number of grid intervals (one less than the samples)."""
        return len(self._offsets) - 1

    def run(
        self, first: int = 0, last: int | None = None, *, max_states: int = CHUNK_STATES
    ) -> SearchResult:
        """Search the grid intervals ``first`` to ``last - 1``.

        Parameters
        ----------
        first, last
            The intervals searched, ``[first, last)``; all by default.
            Interval ``k`` runs from ``offsets[k]`` to ``offsets[k + 1]``.
        max_states
            Most states (objects x times) propagated at once.

        Returns
        -------
        SearchResult
            The minima inside the volumes whose TCA lies in the intervals
            (each interval includes its end, not its start), and the failed
            objects.

        Raises
        ------
        ValueError
            If ``0 <= first <= last <= intervals`` does not hold or
            ``max_states`` is below 1.
        """
        stop = self.intervals if last is None else last
        if not 0 <= first <= stop <= self.intervals:
            raise ValueError(
                f"intervals must satisfy 0 <= first <= last <= {self.intervals}, "
                f"got {first} and {last}"
            )
        if max_states < 1:
            raise ValueError(f"max_states must be at least 1, got {max_states}")
        failures: tuple[dict[int, ObjectFailure], dict[int, ObjectFailure]] = ({}, {})
        rows = np.flatnonzero(self._pairs.any(axis=1))
        if first == stop or rows.size == 0:
            return SearchResult((), {}, {})
        candidates = self._candidates(rows, first, stop, max_states, failures)
        minima: list[RangeMinimum] = []
        for (primary, secondary), found in sorted(candidates.items()):
            try:
                minima.extend(self._refine(primary, secondary, found))
            except _Failed as failed:
                if failed.primary is not None:
                    _earliest(failures[0], primary, failed.primary)
                if failed.secondary is not None:
                    _earliest(failures[1], secondary, failed.secondary)
        return SearchResult(tuple(minima), *failures)

    def _candidates(
        self,
        rows: IndexArray,
        first: int,
        stop: int,
        max_states: int,
        failures: tuple[dict[int, ObjectFailure], dict[int, ObjectFailure]],
    ) -> dict[tuple[int, int], list[tuple[int, bool]]]:
        """Propagate the grid in chunks and sieve every pair's intervals."""
        columns = np.flatnonzero(self._pairs.any(axis=0))
        primaries = self._primaries.subset(rows)
        catalog = self._catalog.subset(columns)
        own = [
            np.searchsorted(columns, np.flatnonzero(self._pairs[row])) for row in rows
        ]
        samples = self._offsets[first : stop + 1]
        per_chunk = max(1, max_states // (len(rows) + len(columns)))
        found: dict[tuple[int, int], list[tuple[int, bool]]] = {}
        carried: tuple[FloatArray, FloatArray, FloatArray] | None = None
        for begin in range(0, len(samples), per_chunk):
            times = samples[begin : begin + per_chunk]
            ephemeris_p = primaries.propagate(self._start, times)
            ephemeris_c = catalog.propagate(self._start, times)
            _record(failures[0], rows, ephemeris_p.errors, times)
            _record(failures[1], columns, ephemeris_c.errors, times)
            r_p, r_c, offset = ephemeris_p.r, ephemeris_c.r, first + begin
            if carried is not None:
                r_p = np.concatenate([carried[0], r_p], axis=1)
                r_c = np.concatenate([carried[1], r_c], axis=1)
                times = np.concatenate([carried[2], times])
                offset -= 1
            carried = (r_p[:, -1:], r_c[:, -1:], times[-1:])
            if len(times) < 2:
                continue
            for i, row in enumerate(rows):
                secondaries, intervals, quick = self._sieve(
                    int(row), r_p[i], r_c[own[i]], columns[own[i]], times
                )
                for secondary, interval, fast in zip(
                    secondaries.tolist(),
                    intervals.tolist(),
                    quick.tolist(),
                    strict=True,
                ):
                    found.setdefault((int(row), secondary), []).append(
                        (offset + interval, fast)
                    )
        return found

    def _sieve(
        self,
        row: int,
        r_primary: FloatArray,
        r_others: FloatArray,
        others: IndexArray,
        times: FloatArray,
    ) -> tuple[IndexArray, IndexArray, BoolArray]:
        """The intervals of one primary's pairs that pass both sieves.

        Returns the catalog positions and interval indices (within
        ``times``) that pass, and whether ``f`` is monotone over each.
        """
        rho = r_others - r_primary
        distance = np.sqrt(_dot(rho, rho))
        length = np.diff(times)
        speed = self._speed_primary[row] + self._speed_catalog[others]
        reach = self._reach[row]
        with np.errstate(invalid="ignore"):
            near = distance[:, :-1] + distance[:, 1:] <= 2.0 * reach + np.outer(
                speed, length
            )
        pair, interval = np.nonzero(near)
        a, b = rho[pair, interval], rho[pair, interval + 1]
        h = length[interval]
        chord = b - a
        squared = _dot(chord, chord)
        along = np.zeros_like(squared)
        moving = squared > 0.0
        along[moving] = -_dot(a[moving], chord[moving]) / squared[moving]
        closest = a + np.clip(along, 0.0, 1.0)[:, None] * chord
        nearest = np.sqrt(_dot(closest, closest))
        ends = distance[pair, interval], distance[pair, interval + 1]
        farthest = 0.5 * (ends[0] + ends[1] + speed[pair] * h)
        lowest = np.minimum(
            self._lowest_primary[row], self._lowest_catalog[others[pair]]
        )
        acceleration, widest = _acceleration(
            np.maximum(ends[0], ends[1]), farthest, lowest, h
        )
        pad = acceleration * h * h / 8.0
        same = np.all(a == 0.0, axis=1) & np.all(b == 0.0, axis=1)
        keep = (nearest - pad <= reach) & ~same
        sweep = np.sqrt(squared) / h - 0.5 * acceleration * h
        quick = (sweep > 0.0) & (sweep * sweep > widest * acceleration)
        return others[pair[keep]], interval[keep], quick[keep]

    def _refine(
        self, primary: int, secondary: int, found: list[tuple[int, bool]]
    ) -> list[RangeMinimum]:
        """The minima of one pair in its intervals that passed the sieves."""
        pair = _Pair(self, primary, secondary)
        grid = self._offsets
        brackets = [_samples(grid[k], grid[k + 1], quick=quick) for k, quick in found]
        points = np.unique(np.concatenate(brackets))
        derivative = pair.derivative(points)
        pair.known = dict(zip(points.tolist(), derivative.tolist(), strict=True))
        minima: list[RangeMinimum] = []
        for samples in brackets:
            values = derivative[np.searchsorted(points, samples)]
            for j in np.flatnonzero((values[:-1] < 0.0) & (values[1:] >= 0.0)):
                tca = brentq(pair.at, samples[j], samples[j + 1], xtol=TCA_TOLERANCE)
                minimum = self._inside(pair, primary, secondary, float(tca))
                if minimum is not None:
                    minima.append(minimum)
        return minima

    def _inside(
        self, pair: _Pair, primary: int, secondary: int, tca: float
    ) -> RangeMinimum | None:
        """The minimum at ``tca`` if the other object is inside the volume."""
        r_p, v_p, r_s, v_s = (state[0] for state in pair.states(np.array([tca])))
        if not self._volumes[primary].contains_position(r_p, v_p, r_s):
            return None
        rho = r_s - r_p
        return RangeMinimum(
            primary, secondary, tca, float(np.sqrt(_dot(rho, rho))), r_p, v_p, r_s, v_s
        )
