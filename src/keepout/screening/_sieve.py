"""The speed and chord sieves of :class:`~keepout.screening.tca.ApproachSearch`:
which grid intervals of a pair may hold a point inside the volume (see
:mod:`keepout.screening.tca` for the bounds)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np

from keepout._linalg import dot
from keepout.orbits import EARTH_MU

if TYPE_CHECKING:
    from keepout._types import BoolArray, FloatArray, IndexArray
    from keepout.propagation import RadiusBounds

__all__ = [
    "GRAVITY_MARGIN",
    "MIN_RADIUS",
    "PERTURBATION",
    "SPEED_BOUND_MARGIN",
    "chord_sieve",
    "floored_lowest",
    "speed_bounds",
    "speed_sieve",
]

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


def floored_lowest(bounds: RadiusBounds) -> FloatArray:
    """Each object's lowest distance from the Earth's centre [m], taken at
    least :data:`MIN_RADIUS`."""
    lowest: FloatArray = np.maximum(bounds.lowest, MIN_RADIUS)
    return lowest


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
    lowest = floored_lowest(bounds)
    highest = np.maximum(bounds.highest, lowest)
    # highest / (lowest + highest), written to give 1 for an infinite highest.
    share = 1.0 / (1.0 + lowest / highest)
    speeds: FloatArray = np.sqrt(2.0 * EARTH_MU * GRAVITY_MARGIN * share / lowest)
    return speeds * (1.0 + SPEED_BOUND_MARGIN)


def speed_sieve(
    distance: FloatArray,
    length: FloatArray,
    speed: FloatArray,
    bounding_radius: float,
) -> tuple[IndexArray, IndexArray]:
    """The pairs (rows of ``distance``) and intervals that pass the speed
    sieve, from each pair's range on the grid, the intervals' lengths, each
    pair's relative speed bound and the volume's bounding radius."""
    with np.errstate(invalid="ignore"):
        near = distance[:, :-1] + distance[:, 1:] <= 2.0 * bounding_radius + np.outer(
            speed, length
        )
    pair_rows, interval_indices = np.nonzero(near)
    return pair_rows, interval_indices


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
        The speed sieve's bound on the range over each interval
        (``rho_up``) [m].
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


def chord_sieve(
    rho: FloatArray,
    distance: FloatArray,
    length: FloatArray,
    speed: FloatArray,
    lowest: FloatArray,
    bounding_radius: float,
    passed: tuple[IndexArray, IndexArray],
) -> tuple[BoolArray, BoolArray]:
    """Which of the intervals that passed the speed sieve pass the chord
    sieve, and whether ``f`` is monotone over each.

    ``rho`` and ``distance`` are the relative positions and ranges of the
    pairs (rows) on the grid, as :func:`speed_sieve` takes them, ``length``
    the intervals' lengths, ``speed`` each pair's relative speed bound,
    ``lowest`` the lowest radius of either object of each pair and
    ``bounding_radius`` the volume's; ``passed`` is what
    :func:`speed_sieve` returned. A pair at exactly the same position at
    both ends of an interval is taken as one object and fails.
    """
    pair_rows, interval_indices = passed
    rho_start = rho[pair_rows, interval_indices]
    rho_end = rho[pair_rows, interval_indices + 1]
    range_start = distance[pair_rows, interval_indices]
    range_end = distance[pair_rows, interval_indices + 1]
    interval_length = length[interval_indices]
    speed = speed[pair_rows]
    lowest = lowest[pair_rows]
    chord = rho_end - rho_start
    squared = dot(chord, chord)
    along = np.zeros_like(squared)
    moving = squared > 0.0
    along[moving] = -dot(rho_start[moving], chord[moving]) / squared[moving]
    closest = rho_start + np.clip(along, 0.0, 1.0)[:, None] * chord
    chord_distance = np.sqrt(dot(closest, closest))
    speed_range_bound = 0.5 * (range_start + range_end + speed * interval_length)
    acceleration, range_bound = _acceleration_and_range_bounds(
        np.maximum(range_start, range_end),
        speed_range_bound,
        lowest,
        interval_length,
    )
    chord_sag = acceleration * interval_length * interval_length / 8.0
    same = np.all(rho_start == 0.0, axis=1) & np.all(rho_end == 0.0, axis=1)
    keep = (chord_distance - chord_sag <= bounding_radius) & ~same
    min_relative_speed = (
        np.sqrt(squared) / interval_length - 0.5 * acceleration * interval_length
    )
    monotone = (min_relative_speed > 0.0) & (
        min_relative_speed * min_relative_speed > range_bound * acceleration
    )
    return keep, monotone
