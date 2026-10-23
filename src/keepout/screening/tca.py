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
   bounds, plus :data:`SPEED_BOUND_MARGIN`), so the relative speed is below ``V =
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

Within ``2 DERIVATIVE_STEP`` of the window's ends the central difference
takes positions outside the window. An error SGP4 reports there (an object
decaying a second after the window) is not the object's: ``f`` at such a
time comes from a one-sided fourth-order difference over five positions
inside the window instead.

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

**Failures.** An object fails when SGP4 reports an error for it at a
time in the window: on the grid, or while an interval is refined. A failure
while refining an interval loses that interval's minima only; the pair's
other intervals are refined as if it had none.

**Determinism.** Each interval is decided from the states of its own pair
only, and so is each failure found while refining it, so the result does
not depend on how the grid is cut into chunks (``max_states``), nor on how
the catalog is cut into parts searched apart (e.g. one per worker) and
merged with :meth:`SearchResult.merge`.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, NamedTuple

import numpy as np
import numpy.typing as npt
from astropy.time import Time

from keepout._linalg import as_vector
from keepout._types import BoolArray, FloatArray, IndexArray, read_only
from keepout._validation import (
    require_at_least,
    require_non_negative,
    require_positive,
    require_single_time,
)
from keepout.orbits import EARTH_MU
from keepout.propagation import (
    CHUNK_STATES,
    NO_ERROR,
    ErrorArray,
    PairStates,
    Propagator,
    RadiusBounds,
    States,
    pair_states,
    propagate_in_chunks,
)
from keepout.screening.volume import ScreeningVolume, per_primary

__all__ = [
    "DEFAULT_STEP",
    "DERIVATIVE_STEP",
    "GRAVITY_MARGIN",
    "MIN_RADIUS",
    "PERTURBATION",
    "SLOW_STEP",
    "SPEED_BOUND_MARGIN",
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

SPEED_BOUND_MARGIN: Final = 0.05
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

_CENTRAL_OFFSETS: Final = np.array([-2.0, -1.0, 1.0, 2.0]) * DERIVATIVE_STEP
"""Offsets [s] of the four positions of the central difference that gives
``f``, from the time it is computed at."""

_CENTRAL_WEIGHTS: Final = np.array([1.0, -8.0, 8.0, -1.0]) / (24.0 * DERIVATIVE_STEP)
"""Weights of ``|rho|^2`` at :data:`_CENTRAL_OFFSETS`: its fourth-order
central difference, halved, ``f`` [m^2/s]."""

_ONE_SIDED_WEIGHTS: Final = np.array([-25.0, 48.0, -36.0, 16.0, -3.0]) / 24.0
"""Fourth-order one-sided difference of ``|rho|^2`` over positions 0 to 4
steps away, halved, per unit step: ``f`` at a window's ends when the
central difference reaches a failed state outside the window."""


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
    widened by :data:`SPEED_BOUND_MARGIN` and :data:`GRAVITY_MARGIN`.

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
    return speeds * (1.0 + SPEED_BOUND_MARGIN)


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, eq=False, slots=True)
class RangeMinimum:
    """A local minimum of a pair's range, inside the primary's volume.

    Parameters
    ----------
    primary_index
        Position of the primary among the search's primaries.
    secondary_index
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
        If a state vector does not have shape ``(3,)`` or is not finite.
    """

    primary_index: int
    secondary_index: int
    offset: float
    distance: float
    r_primary: FloatArray
    v_primary: FloatArray
    r_secondary: FloatArray
    v_secondary: FloatArray

    def __post_init__(self) -> None:
        for name in ("r_primary", "v_primary", "r_secondary", "v_secondary"):
            object.__setattr__(self, name, as_vector(getattr(self, name), name))


def _minimum_sort_key(minimum: RangeMinimum) -> tuple[int, int, float]:
    """The order of minima in a result: by primary, secondary, then offset."""
    return minimum.primary_index, minimum.secondary_index, minimum.offset


def _keep_earliest(
    failures: dict[int, ObjectFailure], index: int, failure: ObjectFailure
) -> None:
    """An object's new failure, stored in ``failures`` in place of the one
    recorded unless that one is earlier (by offset, then code)."""
    known = failures.get(index)
    if known is None or (failure.offset, failure.code) < (known.offset, known.code):
        failures[index] = failure


@dataclass(frozen=True, eq=False, slots=True)
class SearchResult:
    """What a search finds.

    Parameters
    ----------
    minima
        The minima inside the volumes; kept sorted by primary, secondary and
        offset.
    primary_failures, catalog_failures
        The earliest failed state seen of each primary and catalog object,
        by position (copies, sorted by position). Screening skips these
        objects: minima of a failed object are still listed here, from the
        intervals in which both objects were propagated.
    """

    minima: tuple[RangeMinimum, ...]
    primary_failures: Mapping[int, ObjectFailure]
    catalog_failures: Mapping[int, ObjectFailure]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "minima", tuple(sorted(self.minima, key=_minimum_sort_key))
        )
        primary = dict(sorted(self.primary_failures.items()))
        catalog = dict(sorted(self.catalog_failures.items()))
        object.__setattr__(self, "primary_failures", primary)
        object.__setattr__(self, "catalog_failures", catalog)

    @classmethod
    def merge(cls, results: Iterable[SearchResult]) -> SearchResult:
        """One result from the results of searches over parts of a catalog.

        The minima are put together and each object keeps its earliest
        failure; the results must number the objects alike.
        """
        minima: list[RangeMinimum] = []
        primary: dict[int, ObjectFailure] = {}
        catalog: dict[int, ObjectFailure] = {}
        for result in results:
            minima.extend(result.minima)
            for index, failure in result.primary_failures.items():
                _keep_earliest(primary, index, failure)
            for index, failure in result.catalog_failures.items():
                _keep_earliest(catalog, index, failure)
        return cls(tuple(minima), primary, catalog)


def _dot(a: FloatArray, b: FloatArray) -> FloatArray:
    """Dot products along the last axis, element by element: the same bits
    whatever the shape of the arrays (a reduction might not)."""
    product: FloatArray = a[..., 0] * b[..., 0]
    product += a[..., 1] * b[..., 1]
    product += a[..., 2] * b[..., 2]
    return product


def _acceleration_and_range_bounds(
    end_range: FloatArray,
    speed_range_bound: FloatArray,
    lowest_radius: FloatArray,
    interval: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Bounds on the relative acceleration and the range over intervals.

    The range stays within ``A h^2 / 8`` of the chord, so it is at most
    ``end_range + A h^2 / 8``, with ``A`` itself growing with the range: the
    bound on the range solves that inequality, capped by the speed sieve's.

    Parameters
    ----------
    end_range
        The larger range at the two ends of each interval [m].
    speed_range_bound
        The speed sieve's bound on the range over each interval (``rho_up``)
        [m].
    lowest_radius
        The lowest distance of either object from the Earth's centre [m].
    interval
        The length of each interval (``h``) [s].

    Returns
    -------
    tuple[FloatArray, FloatArray]
        The bound ``A`` on the relative acceleration [m/s^2] and the bound
        on the range over each interval [m].
    """
    mu = EARTH_MU * GRAVITY_MARGIN
    # Both objects' gravity added: a bound whatever their separation [m/s^2].
    gravity_sum = 2.0 * mu / (lowest_radius * lowest_radius)
    # The squared lowest radius on the segment joining the objects [m^2].
    segment_radius_squared = (
        lowest_radius * lowest_radius - 0.25 * speed_range_bound * speed_range_bound
    )
    # The gravity gradient there [1/s^2], infinite when the segment may pass
    # through the centre.
    gradient = np.full_like(speed_range_bound, np.inf)
    segment = segment_radius_squared > 0.0
    gradient[segment] = 2.0 * mu / segment_radius_squared[segment] ** 1.5
    range_bound = speed_range_bound.copy()
    # range <= end_range + (gradient range + 2 PERTURBATION) h^2 / 8 gives
    # range (1 - gradient h^2 / 8) <= end_range + PERTURBATION h^2 / 4.
    range_factor = 1.0 - gradient * interval * interval / 8.0
    solvable = range_factor > 0.0
    range_bound[solvable] = np.minimum(
        speed_range_bound[solvable],
        (end_range[solvable] + PERTURBATION * interval[solvable] ** 2 / 4.0)
        / range_factor[solvable],
    )
    acceleration: FloatArray = (
        np.minimum(gravity_sum, gradient * range_bound) + 2.0 * PERTURBATION
    )
    return acceleration, range_bound


class _Failed(Exception):
    """Propagation failed, in the window, while refining a pair."""

    def __init__(
        self, primary: ObjectFailure | None, secondary: ObjectFailure | None
    ) -> None:
        super().__init__()
        self.primary = primary
        self.secondary = secondary


def _first_failure(errors: ErrorArray, stamps: FloatArray) -> ObjectFailure | None:
    """The earliest failed state among ``errors`` at ``stamps``, or None when
    none failed."""
    bad = np.flatnonzero(errors != NO_ERROR)
    if bad.size == 0:
        return None
    first = bad[np.argmin(stamps[bad])]
    return ObjectFailure(float(stamps[first]), int(errors[first]))


def _require_no_failure(
    primary_errors: ErrorArray, secondary_errors: ErrorArray, stamps: FloatArray
) -> None:
    """A check that neither object of a pair failed at any of ``stamps``.

    Raises
    ------
    _Failed
        If ``primary_errors`` or ``secondary_errors`` holds an SGP4 error
        code; it carries the earliest failure of each.
    """
    failed = (
        _first_failure(primary_errors, stamps),
        _first_failure(secondary_errors, stamps),
    )
    if failed != (None, None):
        raise _Failed(*failed)


@dataclass(frozen=True, slots=True)
class _Failures:
    """The earliest failure seen of each primary and catalog object, by
    position."""

    primary: dict[int, ObjectFailure] = field(default_factory=dict)
    catalog: dict[int, ObjectFailure] = field(default_factory=dict)

    def note(self, primary: int, secondary: int, failed: _Failed) -> None:
        """A pair's failure found while refining it, kept as each failed
        object's earliest."""
        if failed.primary is not None:
            _keep_earliest(self.primary, primary, failed.primary)
        if failed.secondary is not None:
            _keep_earliest(self.catalog, secondary, failed.secondary)


def _stencil_sum(rho: FloatArray, weights: FloatArray) -> FloatArray:
    """``sum_k weights[k] |rho_k|^2`` over each run of ``len(weights)``
    consecutive relative positions, added term by term in order: the same
    bits whatever the number of runs (a matrix product might not)."""
    squared = _dot(rho, rho).reshape(-1, len(weights))
    total: FloatArray = squared[:, 0] * weights[0]
    for k in range(1, len(weights)):
        total = total + squared[:, k] * weights[k]
    return total


class _Pair:
    """One primary and one catalog object, evaluated at any times.

    ``sample`` gives the states of both (rows 0 and 1) at offsets from the
    start of the window, which is ``duration`` long.
    """

    def __init__(self, sample: States, duration: float) -> None:
        self.sample = sample
        self.duration = duration
        self._known_f: dict[float, float] = {}

    def states(
        self, stamps: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Positions and velocities of both at ``stamps``, ``(n, 3)`` each."""
        errors, r, v = self.sample(stamps)
        if errors.any():
            _require_no_failure(errors[0], errors[1], stamps)
        return r[0], v[0], r[1], v[1]

    def _computed_f(self, times: FloatArray) -> FloatArray:
        """``f = rho . drho/dt`` at ``times`` (in the window) from positions
        only, whether known or not.

        Raises :class:`_Failed` for a failure in the window only: a time
        whose central difference reaches a failed state outside the window
        takes the one-sided difference inside it.
        """
        stamps = (times[:, None] + _CENTRAL_OFFSETS).reshape(-1)
        errors, r, _ = self.sample(stamps)
        if not errors.any():
            return _stencil_sum(r[1] - r[0], _CENTRAL_WEIGHTS)
        outside = (stamps < 0.0) | (stamps > self.duration)
        inside = ~outside
        _require_no_failure(errors[0][inside], errors[1][inside], stamps[inside])
        derivative = _stencil_sum(r[1] - r[0], _CENTRAL_WEIGHTS)
        failed = (errors[0] != NO_ERROR) | (errors[1] != NO_ERROR)
        reaching = (failed & outside).reshape(-1, len(_CENTRAL_OFFSETS))
        edge = np.flatnonzero(np.any(reaching, axis=1))
        if edge.size > 0:
            derivative[edge] = self._one_sided_f(times[edge])
        return derivative

    def _one_sided_f(self, times: FloatArray) -> FloatArray:
        """``f`` at ``times`` from five positions inside the window, towards
        its far end (closer together than DERIVATIVE_STEP in a window
        shorter than four of them)."""
        after = self.duration - times
        forward = after >= times
        room = np.where(forward, after, times)
        spacing = np.minimum(DERIVATIVE_STEP, room / 4.0) * np.where(forward, 1.0, -1.0)
        nodes = times[:, None] + spacing[:, None] * np.arange(len(_ONE_SIDED_WEIGHTS))
        stamps = np.clip(nodes, 0.0, self.duration).reshape(-1)
        r_primary, _, r_secondary, _ = self.states(stamps)
        derivative: FloatArray = (
            _stencil_sum(r_secondary - r_primary, _ONE_SIDED_WEIGHTS) / spacing
        )
        return derivative

    def f_values(self, times: FloatArray) -> FloatArray:
        """``f`` at ``times``, from the values already known if all are,
        else computed (and kept)."""
        listed = times.tolist()
        if all(time in self._known_f for time in listed):
            return np.array([self._known_f[time] for time in listed])
        values = self._computed_f(times)
        self._known_f.update(zip(listed, values.tolist(), strict=True))
        return values

    def f_at(self, time: float) -> float:
        """``f`` at one time, for Brent's method (which starts at the ends
        of its bracket, already known)."""
        known = self._known_f.get(time)
        if known is not None:
            return known
        return float(self._computed_f(np.array([time]))[0])


class _Interval(NamedTuple):
    """A grid interval of a pair that passed the sieves."""

    first: float
    last: float
    monotone: bool
    """Whether ``f`` is monotone over it: its sign at the ends decides."""


def _samples(interval: _Interval) -> FloatArray:
    """Where ``f`` is evaluated in an interval: its ends, or every SLOW_STEP."""
    first, last = interval.first, interval.last
    if interval.monotone:
        return np.array([first, last])
    count = math.ceil((last - first) / SLOW_STEP)
    samples = first + (last - first) * (np.arange(count + 1) / count)
    samples[-1] = last
    return samples


def _roots(pair: _Pair, samples: FloatArray) -> tuple[list[float], _Failed | None]:
    """The TCAs of a pair in one interval, sampled at ``samples``, in order,
    and the failure that stopped the search, if one did."""
    # Imported here: scipy.optimize takes a noticeable time to import.
    from scipy.optimize import brentq

    try:
        values = pair.f_values(samples)
    except _Failed as exc:
        return [], exc
    tcas: list[float] = []
    for j in np.flatnonzero((values[:-1] < 0.0) & (values[1:] >= 0.0)):
        try:
            tca = brentq(pair.f_at, samples[j], samples[j + 1], xtol=TCA_TOLERANCE)
        except _Failed as exc:
            return tcas, exc
        tcas.append(float(tca))
    return tcas, None


def _record_first_failures(
    failures: dict[int, ObjectFailure],
    positions: IndexArray,
    errors: ErrorArray,
    times: FloatArray,
) -> None:
    """The first failed state of each object of a chunk (in time order),
    kept as its earliest failure."""
    failed = errors != NO_ERROR
    for row in np.flatnonzero(np.any(failed, axis=1)):
        first = int(np.argmax(failed[row]))
        failure = ObjectFailure(float(times[first]), int(errors[row, first]))
        _keep_earliest(failures, int(positions[row]), failure)


class ApproachSearch:
    """The minima of the range of primary-catalog pairs inside the volumes.

    See the module's description for the method and what it guarantees.

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
        The objects' radius bounds over the window
        (:meth:`~keepout.propagation.base.Propagator.radius_bounds`).

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
        primary_bounds: RadiusBounds,
        catalog_bounds: RadiusBounds,
    ) -> None:
        if primaries.frame != catalog.frame:
            raise ValueError(
                f"primaries and catalog must share a frame, got {primaries.frame} "
                f"and {catalog.frame}"
            )
        require_single_time(start, "start")
        self._offsets = time_grid(duration, step)
        mask = np.array(pairs)
        if mask.dtype != np.bool_ or mask.shape != (len(primaries), len(catalog)):
            raise ValueError(
                f"pairs must be booleans of shape ({len(primaries)}, {len(catalog)}), "
                f"got {mask.dtype} {mask.shape}"
            )
        volumes = per_primary(volumes, len(primaries))
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
        self._volumes = volumes
        self._pairs = mask
        self._speed_primary = speed_bounds(primary_bounds)
        self._speed_catalog = speed_bounds(catalog_bounds)
        self._lowest_primary = np.maximum(primary_bounds.lowest, MIN_RADIUS)
        self._lowest_catalog = np.maximum(catalog_bounds.lowest, MIN_RADIUS)
        self._reach = np.array([volume.bounding_radius for volume in self._volumes])

    def run(self, *, max_states: int = CHUNK_STATES) -> SearchResult:
        """The search over the whole grid.

        Parameters
        ----------
        max_states
            Most states (objects x times) propagated at once; the result
            does not depend on it.

        Returns
        -------
        SearchResult
            The minima inside the volumes whose TCA lies in the window, and
            the failed objects.

        Raises
        ------
        ValueError
            If ``max_states`` is below 1.
        """
        require_at_least(max_states, 1, "max_states")
        rows = np.flatnonzero(self._pairs.any(axis=1))
        if len(self._offsets) < 2 or rows.size == 0:
            return SearchResult((), {}, {})
        failures = _Failures()
        candidates = self._candidates(rows, max_states, failures)
        states_of = pair_states(self._start, self._primaries, self._catalog)
        minima: list[RangeMinimum] = []
        for (primary, secondary), intervals in sorted(candidates.items()):
            minima.extend(
                self._refine(states_of, primary, secondary, intervals, failures)
            )
        return SearchResult(tuple(minima), failures.primary, failures.catalog)

    def _candidates(
        self, rows: IndexArray, max_states: int, failures: _Failures
    ) -> dict[tuple[int, int], list[_Interval]]:
        """The intervals of each pair that pass the sieves, in time order.

        The grid is propagated in chunks; the failures seen are noted in
        ``failures``.
        """
        columns = np.flatnonzero(self._pairs.any(axis=0))
        own = [
            np.searchsorted(columns, np.flatnonzero(self._pairs[row])) for row in rows
        ]
        chunks = propagate_in_chunks(
            (self._primaries.subset(rows), self._catalog.subset(columns)),
            self._start,
            self._offsets,
            max_states=max_states,
            shared_times=1,
        )
        intervals_by_pair: dict[tuple[int, int], list[_Interval]] = {}
        for primaries, catalog in chunks:
            _record_first_failures(
                failures.primary, rows, primaries.errors, primaries.offsets
            )
            _record_first_failures(
                failures.catalog, columns, catalog.errors, catalog.offsets
            )
            for i, row in enumerate(rows):
                self._collect(
                    intervals_by_pair,
                    int(row),
                    primaries.r[i],
                    catalog.r[own[i]],
                    columns[own[i]],
                    primaries.offsets,
                )
        return intervals_by_pair

    def _collect(
        self,
        intervals_by_pair: dict[tuple[int, int], list[_Interval]],
        row: int,
        r_primary: FloatArray,
        r_others: FloatArray,
        others: IndexArray,
        times: FloatArray,
    ) -> None:
        """The intervals of one primary's pairs over a chunk that pass the
        sieves, added to ``intervals_by_pair``."""
        secondaries, starts, monotone = self._sieve(
            row, r_primary, r_others, others, times
        )
        for secondary, first, last, is_monotone in zip(
            secondaries.tolist(),
            times[starts].tolist(),
            times[starts + 1].tolist(),
            monotone.tolist(),
            strict=True,
        ):
            intervals_by_pair.setdefault((row, secondary), []).append(
                _Interval(first, last, is_monotone)
            )

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
        pair, interval = self._speed_sieve(row, distance, length, speed)
        keep, monotone = self._chord_sieve(
            row,
            others[pair],
            rho[pair, interval],
            rho[pair, interval + 1],
            distance[pair, interval],
            distance[pair, interval + 1],
            length[interval],
            speed[pair],
        )
        return others[pair[keep]], interval[keep], monotone[keep]

    def _speed_sieve(
        self,
        row: int,
        distance: FloatArray,
        length: FloatArray,
        speed: FloatArray,
    ) -> tuple[IndexArray, IndexArray]:
        """The pairs (rows of ``distance``) and intervals that pass the speed
        sieve, from each pair's range on the grid, the intervals' lengths and
        each pair's relative speed bound."""
        reach = self._reach[row]
        with np.errstate(invalid="ignore"):
            near = distance[:, :-1] + distance[:, 1:] <= 2.0 * reach + np.outer(
                speed, length
            )
        pair, interval = np.nonzero(near)
        return pair, interval

    def _chord_sieve(
        self,
        row: int,
        secondaries: IndexArray,
        rho_start: FloatArray,
        rho_end: FloatArray,
        range_start: FloatArray,
        range_end: FloatArray,
        interval_length: FloatArray,
        speed: FloatArray,
    ) -> tuple[BoolArray, BoolArray]:
        """Which intervals pass the chord sieve, and whether ``f`` is
        monotone over each.

        Each entry is one interval of one pair that passed the speed sieve:
        its catalog object, the relative positions and ranges at its ends,
        its length and the pair's relative speed bound. A pair at exactly
        the same position at both ends is taken as one object and fails.
        """
        chord = rho_end - rho_start
        squared = _dot(chord, chord)
        along = np.zeros_like(squared)
        moving = squared > 0.0
        along[moving] = -_dot(rho_start[moving], chord[moving]) / squared[moving]
        closest = rho_start + np.clip(along, 0.0, 1.0)[:, None] * chord
        nearest = np.sqrt(_dot(closest, closest))
        farthest = 0.5 * (range_start + range_end + speed * interval_length)
        lowest = np.minimum(
            self._lowest_primary[row], self._lowest_catalog[secondaries]
        )
        acceleration, range_bound = _acceleration_and_range_bounds(
            np.maximum(range_start, range_end), farthest, lowest, interval_length
        )
        pad = acceleration * interval_length * interval_length / 8.0
        same = np.all(rho_start == 0.0, axis=1) & np.all(rho_end == 0.0, axis=1)
        keep = (nearest - pad <= self._reach[row]) & ~same
        sweep = (
            np.sqrt(squared) / interval_length - 0.5 * acceleration * interval_length
        )
        monotone = (sweep > 0.0) & (sweep * sweep > range_bound * acceleration)
        return keep, monotone

    def _refine(
        self,
        states_of: PairStates,
        primary: int,
        secondary: int,
        intervals: list[_Interval],
        failures: _Failures,
    ) -> list[RangeMinimum]:
        """The minima of one pair in its intervals that passed the sieves.

        A failure while refining an interval is recorded in ``failures`` and
        loses that interval's minima only.
        """
        pair = _Pair(states_of(primary, secondary), self._duration)
        brackets = [_samples(interval) for interval in intervals]
        # Every interval's samples in one evaluation, kept by the pair; after a
        # failure, each interval is evaluated, and its failure found, on its own.
        with contextlib.suppress(_Failed):
            pair.f_values(np.unique(np.concatenate(brackets)))
        roots = [_roots(pair, samples) for samples in brackets]
        tcas = np.array([tca for interval_tcas, _ in roots for tca in interval_tcas])
        if tcas.size == 0:
            states = (np.empty((0, 3)),) * 4
            errors = np.zeros((2, 0), dtype=np.uint8)
        else:
            errors, r, v = pair.sample(tcas)
            states = r[0], v[0], r[1], v[1]
        minima: list[RangeMinimum] = []
        first = 0
        for interval_tcas, failed in roots:
            span = slice(first, first + len(interval_tcas))
            first = span.stop
            try:
                minima.extend(
                    self._minima_inside_volume(
                        primary,
                        secondary,
                        tcas[span],
                        errors[:, span],
                        tuple(state[span] for state in states),
                        failed,
                    )
                )
            except _Failed as exc:
                failures.note(primary, secondary, exc)
        return minima

    def _minima_inside_volume(
        self,
        primary: int,
        secondary: int,
        tcas: FloatArray,
        errors: ErrorArray,
        states: tuple[FloatArray, ...],
        failed: _Failed | None,
    ) -> list[RangeMinimum]:
        """The minima of a pair in one interval: those of its TCAs (states at
        each given) at which the other object is inside the volume.

        Raises, as if each TCA's state were propagated after its root was
        found, :class:`_Failed` for the first TCA whose state failed, else
        the failure that stopped the search for roots (``failed``).
        """
        bad = np.flatnonzero(np.any(errors != NO_ERROR, axis=0))
        count = int(bad[0]) if bad.size > 0 else len(tcas)
        r_primary, v_primary, r_secondary, v_secondary = (
            state[:count] for state in states
        )
        inside = self._volumes[primary].contains_position(
            r_primary, v_primary, r_secondary
        )
        if count < len(tcas):
            rows = slice(count, count + 1)
            _require_no_failure(errors[0, rows], errors[1, rows], tcas[rows])
        if failed is not None:
            raise failed
        rho = r_secondary - r_primary
        distance = np.sqrt(_dot(rho, rho))
        return [
            RangeMinimum(
                primary,
                secondary,
                float(tcas[k]),
                float(distance[k]),
                r_primary[k],
                v_primary[k],
                r_secondary[k],
                v_secondary[k],
            )
            for k in np.flatnonzero(inside)
        ]
