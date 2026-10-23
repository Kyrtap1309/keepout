"""Where Hall's 3D-Nc rate is exactly 0: the mean-centred test and the screen
that rules whole stretches of a window out before the rate is scanned."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final, NamedTuple

import numpy as np

from keepout._types import BoolArray, FloatArray
from keepout.pc._overlap import Gaussians
from keepout.pc._timeline import Candidate, regions

if TYPE_CHECKING:
    from keepout.pc._rate import CollisionRate

__all__ = [
    "NEGLIGIBLE_DISTANCE",
    "SCREEN_CHUNK",
    "SCREEN_MAX_CELLS",
    "SCREEN_MIN_CELLS",
    "SCREEN_PERIOD_FRACTION",
    "SCREEN_VARIANCE_MARGIN",
    "ScreenCells",
    "negligible",
    "reach",
    "screen",
]

NEGLIGIBLE_DISTANCE: Final = 1500.0
"""Squared Mahalanobis distance from the hard-body sphere to the mean relative
position beyond which the rate underflows (``exp(-750)``) and is taken as 0."""

SCREEN_MIN_CELLS: Final = 64
"""Fewest screen cells in a window."""

SCREEN_MAX_CELLS: Final = 50_000
"""Most screen cells in a window."""

SCREEN_PERIOD_FRACTION: Final = 1.0 / 64.0
"""Largest screen cell, as a fraction of the shorter orbital period."""

SCREEN_VARIANCE_MARGIN: Final = 2.0
"""Factor on the largest position variance of neighbouring screen cells."""

SCREEN_CHUNK: Final = 4096
"""Screen cells evaluated in one vectorised call."""


def _gap(gaussians: Gaussians, radius: float) -> FloatArray:
    """Distance [m] from the hard-body sphere of ``radius`` to the mean
    relative positions."""
    distance = np.linalg.norm(gaussians.relative_position, axis=1)
    gap: FloatArray = np.maximum(distance - radius, 0.0)
    return gap


def negligible(gaussians: Gaussians, radius: float) -> BoolArray:
    """Where the rate is taken as 0: the mean relative position farther from
    the hard-body sphere than :data:`NEGLIGIBLE_DISTANCE` allows."""
    largest = np.linalg.eigvalsh(gaussians.position_covariance)[:, -1]
    gap = _gap(gaussians, radius)
    result: BoolArray = gap * gap > NEGLIGIBLE_DISTANCE * largest
    return result


def reach(largest_variance: float, radius: float) -> float:
    """Distance [m] beyond which :func:`negligible` holds."""
    return math.sqrt(NEGLIGIBLE_DISTANCE * largest_variance) + radius


class ScreenCells(NamedTuple):
    """Mean-centred quantities at the centres of screen cells, each ``(m,)``.

    Parameters
    ----------
    gap
        Distance [m] from the hard-body sphere to the mean relative position.
    speed
        Mean relative speed [m/s].
    largest
        Largest combined position variance [m^2].
    crossing
        Straight-line crossing time [s] (infinite at zero relative speed).
    width
        Time standard deviation of the rate [s] (infinite at zero relative
        speed).
    """

    gap: FloatArray
    speed: FloatArray
    largest: FloatArray
    crossing: FloatArray
    width: FloatArray


def screen(rate: CollisionRate, window: tuple[float, float]) -> list[Candidate]:
    """Candidate intervals of ``window``: where ``rate`` may not be 0.

    The window is cut into cells. On each, the gap between the mean
    positions (less the radius) is bounded from below by its value at the
    cell centre minus how far two-body motion can close it within half a
    cell: ``|dv| h/2 + a_max (h/2)^2 / 2``, with ``a_max`` the sum of the
    two perigee accelerations. This is exact for the mean trajectories.
    The largest position variance over a cell is taken as
    :data:`SCREEN_VARIANCE_MARGIN` times the largest at the cell and its
    two neighbours (it varies smoothly over a cell of at most
    :data:`SCREEN_PERIOD_FRACTION` of an orbit, and grows convexly along
    track). Where even this bounded gap exceeds the reach of
    :data:`NEGLIGIBLE_DISTANCE`, :func:`negligible` holds throughout the
    cell and the rate is exactly 0. The cell size is the time the mean
    relative position takes to cross that reach, within
    :data:`SCREEN_MIN_CELLS` and :data:`SCREEN_MAX_CELLS` cells.
    """
    start, end = window
    count = _cell_count(rate, window)
    cell = (end - start) / count
    centres = start + (np.arange(count, dtype=np.float64) + 0.5) * cell
    parts = [
        _screen_cells(rate, centres[i : i + SCREEN_CHUNK])
        for i in range(0, count, SCREEN_CHUNK)
    ]
    cells = ScreenCells(*(np.concatenate(p) for p in zip(*parts, strict=True)))
    half = 0.5 * cell
    acceleration = rate.primary.max_acceleration + rate.secondary.max_acceleration
    closest = np.maximum(
        cells.gap - cells.speed * half - 0.5 * acceleration * half * half, 0.0
    )
    largest = cells.largest
    padded = np.concatenate([largest[:1], largest, largest[-1:]])
    envelope = SCREEN_VARIANCE_MARGIN * np.maximum.reduce(
        [padded[:-2], padded[1:-1], padded[2:]]
    )
    possible = closest * closest <= NEGLIGIBLE_DISTANCE * envelope
    return [
        Candidate(
            start if first == 0 else start + first * cell,
            end if last == count - 1 else start + (last + 1) * cell,
            float(np.min(cells.crossing[first : last + 1])),
            float(np.min(cells.width[first : last + 1])),
        )
        for first, last in regions(possible)
    ]


def _cell_count(rate: CollisionRate, window: tuple[float, float]) -> int:
    """Number of screen cells in ``window`` (see :func:`screen`)."""
    start, end = window
    span = end - start
    centre = _screen_cells(rate, np.array([0.5 * (start + end)]))
    speed = float(centre.speed[0])
    extent = reach(float(centre.largest[0]), rate.radius)
    natural = extent / speed if speed > 0.0 else math.inf
    longest = SCREEN_PERIOD_FRACTION * min(rate.primary.period, rate.secondary.period)
    cell = max(span / SCREEN_MAX_CELLS, min(natural, span / SCREEN_MIN_CELLS, longest))
    return math.ceil(span / cell)


def _screen_cells(rate: CollisionRate, times: FloatArray) -> ScreenCells:
    """Gap, relative speed, largest variance, crossing time and time width.

    All at ``times``, from the mean-centred expansion that :func:`negligible`
    uses. The time width is the standard deviation ``1 / (|v| sqrt(u' A^-1
    u))`` of the straight-line passage through the density, with ``A``
    floored as in the rate (``rate.floor_variance``).
    """
    gaussians = rate.mean_centred(times)
    dv = gaussians.relative_velocity
    a = gaussians.position_covariance
    gap = _gap(gaussians, rate.radius)
    speed = np.linalg.norm(dv, axis=1)
    values, vectors = np.linalg.eigh(a)
    largest = values[:, -1]
    direction = dv / np.where(speed > 0.0, speed, 1.0)[:, None]
    # u' A u and u' A^-1 u (A floored) in the principal axes of A.
    along2 = np.einsum("ij,ijk->ik", direction, vectors) ** 2
    sigma = np.sqrt(np.maximum(np.einsum("ik,ik->i", along2, values), 0.0))
    precision = np.einsum(
        "ik,ik->i", along2, 1.0 / np.maximum(values, rate.floor_variance)
    )
    with np.errstate(divide="ignore"):
        crossing = np.where(speed > 0.0, (sigma + rate.radius) / speed, math.inf)
        width = np.where(speed > 0.0, 1.0 / (speed * np.sqrt(precision)), math.inf)
    return ScreenCells(gap, speed, largest, crossing, width)
