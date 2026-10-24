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
   ``rho_up = (|rho_a| + |rho_b| + V h) / 2`` (``speed_range_bound`` in the
   code).
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
merged with
:meth:`~keepout.screening.search_result.SearchResult.merge`.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Sequence
from typing import Final, NamedTuple

import numpy as np
import numpy.typing as npt
from astropy.time import Time

from keepout._types import BoolArray, FloatArray, IndexArray, read_only
from keepout._validation import (
    require_at_least,
    require_non_negative,
    require_positive,
    require_single_time,
)
from keepout.propagation import (
    CHUNK_STATES,
    PairStates,
    Propagator,
    RadiusBounds,
    pair_states,
    propagate_in_chunks,
)
from keepout.screening._refine import (
    DERIVATIVE_STEP,
    SLOW_STEP,
    TCA_TOLERANCE,
    Failed,
    Failures,
    Interval,
    Pair,
    interval_samples,
    minima_inside_volume,
    record_first_failures,
    roots,
)
from keepout.screening._sieve import (
    GRAVITY_MARGIN,
    MIN_RADIUS,
    PERTURBATION,
    SPEED_BOUND_MARGIN,
    chord_sieve,
    dot,
    floored_lowest,
    speed_bounds,
    speed_sieve,
)
from keepout.screening.search_result import RangeMinimum, SearchResult
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
    "speed_bounds",
    "time_grid",
]

DEFAULT_STEP: Final = 120.0
"""Longest time between grid samples [s] unless told otherwise."""


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


class _PrimaryChunk(NamedTuple):
    """One primary and the catalog objects it is paired with, over a chunk
    of the grid."""

    row: int
    """Position of the primary."""
    r_primary: FloatArray
    """The primary's positions at ``offsets`` [m], shape ``(T, 3)``."""
    r_secondaries: FloatArray
    """Its catalog objects' positions at ``offsets`` [m], shape ``(S, T, 3)``."""
    secondaries: IndexArray
    """Their positions in the catalog."""
    offsets: FloatArray
    """Offsets of the chunk's grid points [s]."""


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
        self._lowest_primary = floored_lowest(primary_bounds)
        self._lowest_catalog = floored_lowest(catalog_bounds)
        self._bounding_radii = np.array(
            [volume.bounding_radius for volume in self._volumes]
        )

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
            return SearchResult.empty()
        failures = Failures()
        candidates = self._candidates(rows, max_states, failures)
        states_of = pair_states(self._start, self._primaries, self._catalog)
        minima: list[RangeMinimum] = []
        for (primary, secondary), intervals in sorted(candidates.items()):
            minima.extend(
                self._refine(states_of, primary, secondary, intervals, failures)
            )
        return SearchResult(tuple(minima), failures.primary, failures.catalog)

    def _candidates(
        self, rows: IndexArray, max_states: int, failures: Failures
    ) -> dict[tuple[int, int], list[Interval]]:
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
        intervals_by_pair: dict[tuple[int, int], list[Interval]] = {}
        for primaries, catalog in chunks:
            record_first_failures(
                failures.primary, rows, primaries.errors, primaries.offsets
            )
            record_first_failures(
                failures.catalog, columns, catalog.errors, catalog.offsets
            )
            for i, row in enumerate(rows):
                chunk = _PrimaryChunk(
                    int(row),
                    primaries.r[i],
                    catalog.r[own[i]],
                    columns[own[i]],
                    primaries.offsets,
                )
                self._add_passing_intervals(intervals_by_pair, chunk)
        return intervals_by_pair

    def _add_passing_intervals(
        self,
        intervals_by_pair: dict[tuple[int, int], list[Interval]],
        chunk: _PrimaryChunk,
    ) -> None:
        """The intervals of one primary's pairs over a chunk that pass the
        sieves, added to ``intervals_by_pair``."""
        secondaries, starts, monotone = self._sieve(chunk)
        offsets = chunk.offsets
        for secondary, first, last, is_monotone in zip(
            secondaries.tolist(),
            offsets[starts].tolist(),
            offsets[starts + 1].tolist(),
            monotone.tolist(),
            strict=True,
        ):
            intervals_by_pair.setdefault((chunk.row, secondary), []).append(
                Interval(first, last, is_monotone)
            )

    def _sieve(self, chunk: _PrimaryChunk) -> tuple[IndexArray, IndexArray, BoolArray]:
        """The intervals of one primary's pairs that pass both sieves.

        Returns the catalog positions and interval indices (within
        ``chunk.offsets``) that pass, and whether ``f`` is monotone over each.
        """
        row, secondaries = chunk.row, chunk.secondaries
        rho = chunk.r_secondaries - chunk.r_primary
        distance = np.sqrt(dot(rho, rho))
        length = np.diff(chunk.offsets)
        speed = self._speed_primary[row] + self._speed_catalog[secondaries]
        lowest = np.minimum(
            self._lowest_primary[row], self._lowest_catalog[secondaries]
        )
        bounding_radius = self._bounding_radii[row]
        pair_rows, interval_indices = speed_sieve(
            distance, length, speed, bounding_radius
        )
        keep, monotone = chord_sieve(
            rho,
            distance,
            length,
            speed,
            lowest,
            bounding_radius,
            (pair_rows, interval_indices),
        )
        return (
            secondaries[pair_rows][keep],
            interval_indices[keep],
            monotone[keep],
        )

    def _refine(
        self,
        states_of: PairStates,
        primary: int,
        secondary: int,
        intervals: list[Interval],
        failures: Failures,
    ) -> list[RangeMinimum]:
        """The minima of one pair in its intervals that passed the sieves.

        A failure while refining an interval is recorded in ``failures`` and
        loses that interval's minima only.
        """
        pair = Pair(states_of(primary, secondary), self._duration)
        brackets = [interval_samples(interval) for interval in intervals]
        # Every interval's samples in one evaluation, kept by the pair; after a
        # failure, each interval is evaluated, and its failure found, on its own.
        with contextlib.suppress(Failed):
            pair.f_values(np.unique(np.concatenate(brackets)))
        found = [roots(pair, samples) for samples in brackets]
        tcas = np.array([tca for interval_tcas, _ in found for tca in interval_tcas])
        if tcas.size == 0:
            states = (np.empty((0, 3)),) * 4
            errors = np.zeros((2, 0), dtype=np.uint8)
        else:
            errors, r, v = pair.sample(tcas)
            states = r[0], v[0], r[1], v[1]
        volume = self._volumes[primary]
        minima: list[RangeMinimum] = []
        first = 0
        for interval_tcas, failed in found:
            span = slice(first, first + len(interval_tcas))
            first = span.stop
            try:
                minima.extend(
                    minima_inside_volume(
                        volume,
                        primary,
                        secondary,
                        tcas[span],
                        errors[:, span],
                        tuple(state[span] for state in states),
                        failed,
                    )
                )
            except Failed as exc:
                failures.note(primary, secondary, exc)
        return minima
