"""Time windows, encounter segments and the time integral of a collision rate.

A cheap screen first marks the stretches of the window where the rate can be
non-negligible (candidate intervals). Each is scanned on a detection grid
fine enough that no encounter can fall between two nodes
(:data:`DETECTION_SIGMAS`), which is then refined to an eighth of a crossing
time around the nodes where the rate is significant. The stretches where the
rate is not negligible become encounter segments, split at their local maxima
and integrated by Gauss-Legendre rules of doubling order. Without an explicit
window the half-width starts at the linear-motion bound and doubles until the
rate at both edges is negligible, but the window never spans more than the
shorter orbital period.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from keepout._types import FloatArray
from keepout.errors import PcConvergenceError
from keepout.pc._sphere import gauss_legendre

RATE_FLOOR = 1e-16
"""Rate, relative to the peak, below which the rate is negligible."""

SIGMA_REACH = math.sqrt(2.0 * math.log(1.0 / RATE_FLOOR))
"""Standard deviations after which a Gaussian falls to :data:`RATE_FLOOR`."""

MIN_SCAN_POINTS = 401
"""Points of a scan over the whole window: the coarsest spacing is
``window / (MIN_SCAN_POINTS - 1)``."""

MAX_SCAN_POINTS = 20001
"""Largest number of detection-grid points over all candidate intervals."""

MAX_REFINED_POINTS = 20001
"""Largest number of points added by refinement; beyond it the refinement is
coarser (detection is already guaranteed by the detection grid)."""

DETECTION_LEVEL = 1e-8
"""Smallest encounter peak, relative to the largest, the scan must detect."""

DETECTION_SIGMAS = 0.5 * 2.0 * math.sqrt(2.0 * math.log(DETECTION_LEVEL / RATE_FLOOR))
"""Detection-grid spacing in time standard deviations of the rate (6.07).

Over one straight-line crossing the rate is, as a function of time, a
Gaussian of standard deviation ``tau = 1 / (|v| sqrt(u' A^-1 u))`` (``u`` the
unit relative velocity, ``A`` the combined position covariance) convolved
with the passage through the hard-body sphere, which only widens it. An
encounter whose peak is at least :data:`DETECTION_LEVEL` of the largest peak
is therefore above :data:`RATE_FLOOR` of it over at least
``2 tau sqrt(2 ln(DETECTION_LEVEL / RATE_FLOOR))`` = 12.1 ``tau``; a grid half
that wide puts at least one node inside it. The guarantee rests on ``tau``
being the minimum over the candidate interval (:attr:`Candidate.width`, the
smallest over its screen cells) and on the straight-line model; the factor 2
for the departure of curved two-body motion from a straight line is a
heuristic margin, not a bound. A missed encounter contributes at most
~``DETECTION_LEVEL`` of the Nc of the largest one."""

QUAD_RTOL = 1e-6
"""Default relative agreement of two Gauss-Legendre orders."""

MAX_BREAKPOINTS = 50


@dataclass(frozen=True)
class Segment:
    """A stretch of time where the collision rate is not negligible.

    Parameters
    ----------
    start, end
        Bounds [s from TCA].
    nc
        Integral of the rate over the segment.
    """

    start: float
    end: float
    nc: float


@dataclass(frozen=True)
class RateIntegral:
    """The time integral of a collision rate over a window.

    Parameters
    ----------
    nc
        Integral over the whole window.
    segments
        Where the rate is not negligible.
    window
        Integration window [s from TCA].
    edge
        Largest rate at the window edges, relative to the peak.
    peak
        Largest rate on the scan grid [1/s].
    agreement
        Largest difference of the last two Gauss-Legendre orders of a piece
        that never agreed to ``rtol``, relative to ``nc`` (0 if every piece
        agreed; infinite if ``nc`` is 0 or a piece had a single order).
    """

    nc: float
    segments: tuple[Segment, ...]
    window: tuple[float, float]
    edge: float
    peak: float
    agreement: float = 0.0


@dataclass(frozen=True)
class Candidate:
    """A stretch of time where a screen could not rule the rate negligible.

    Parameters
    ----------
    start, end
        Bounds [s from TCA].
    crossing
        Shortest straight-line crossing time in the stretch [s] (infinite
        for zero relative velocity); refined scans resolve it.
    width
        Smallest time standard deviation of the rate in the stretch [s]
        (infinite for zero relative velocity); see :data:`DETECTION_SIGMAS`.
    """

    start: float
    end: float
    crossing: float
    width: float


Screen: TypeAlias = Callable[[tuple[float, float]], list[Candidate]]
"""Candidate intervals of a window, in increasing time and disjoint."""


def straight_line_half_window(
    relative_velocity: FloatArray, position_covariance: FloatArray, radius: float
) -> float:
    """Half-window of a straight-line encounter [s].

    With ``sigma_v`` the combined position standard deviation along the
    relative velocity, the rate is negligible beyond
    ``(SIGMA_REACH sigma_v + R) / |v|``; infinite for zero relative velocity.
    """
    speed = float(np.linalg.norm(relative_velocity))
    if speed == 0.0:
        return math.inf
    direction = relative_velocity / speed
    sigma = math.sqrt(max(float(direction @ position_covariance @ direction), 0.0))
    return (SIGMA_REACH * sigma + radius) / speed


def regions(significant: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Index ranges ``[first, last]`` of consecutive True values."""
    indices = np.flatnonzero(significant)
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.concatenate([[indices[0]], indices[breaks + 1]])
    ends = np.concatenate([indices[breaks], [indices[-1]]])
    return [(int(a), int(b)) for a, b in zip(starts, ends, strict=True)]


@dataclass(frozen=True, eq=False)
class RateScan:
    """A collision rate sampled over a window, with its encounter segments.

    Parameters
    ----------
    times, values
        The scan grid [s from TCA], increasing, and the rate on it [1/s].
        It starts and ends at the window edges; between them it covers the
        candidate intervals of the screen with a detection grid, refined
        where the rate is significant, and is absent where the screen ruled
        the rate negligible.
    segments
        Index ranges ``[first, last]`` (inclusive, padded by one grid step) of
        the stretches where the rate is not negligible; each lies within one
        candidate interval.
    """

    times: FloatArray
    values: FloatArray
    segments: tuple[tuple[int, int], ...]

    @property
    def window(self) -> tuple[float, float]:
        """Scanned window [s from TCA]."""
        return float(self.times[0]), float(self.times[-1])

    @property
    def peak(self) -> float:
        """Largest rate on the grid [1/s]."""
        return float(self.values.max())

    @property
    def edge(self) -> float:
        """Largest rate at the window edges, relative to the peak (0 if none)."""
        return _edge(self.values)


def _edge(values: FloatArray) -> float:
    """Largest of the first and last value, relative to the largest (0 if none)."""
    peak = float(values.max())
    return float(max(values[0], values[-1])) / peak if peak > 0.0 else 0.0


def _detection_grids(
    window: tuple[float, float], candidates: list[Candidate]
) -> list[tuple[FloatArray, float]]:
    """Detection grids over the candidates, clipped to ``window``.

    Each is uniform with spacing ``DETECTION_SIGMAS * width``, at most
    ``window / (MIN_SCAN_POINTS - 1)``; it is returned with the spacing its
    refinement needs (an eighth of the crossing time, not coarser than the
    grid itself).

    Raises
    ------
    PcConvergenceError
        If detection needs more than :data:`MAX_SCAN_POINTS` points.
    """
    start, end = window
    coarsest = (end - start) / (MIN_SCAN_POINTS - 1)
    plan: list[tuple[float, float, int, float]] = []
    finest = math.inf
    for candidate in candidates:
        a, b = max(candidate.start, start), min(candidate.end, end)
        if b <= a:
            continue
        spacing = min(DETECTION_SIGMAS * candidate.width, coarsest)
        finest = min(finest, spacing)
        count = math.ceil((b - a) / spacing) + 1
        plan.append((a, b, count, min(candidate.crossing / 8.0, spacing)))
    if sum(count for _, _, count, _ in plan) > MAX_SCAN_POINTS:
        raise PcConvergenceError(
            f"cannot resolve the collision rate over the window [{start:.6g}, "
            f"{end:.6g}] s: detecting its possible encounters needs a scan "
            f"spacing of {finest:.3g} s, more than {MAX_SCAN_POINTS} points; "
            "pass a shorter window around the encounter, e.g. "
            "compute_pc(conjunction, 'hall', window=(a, b)) (or 'montecarlo')"
        )
    return [(np.linspace(a, b, count), fine) for a, b, count, fine in plan]


def _refine(
    rate: Callable[[float], float],
    times: FloatArray,
    values: FloatArray,
    fine: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Refine the grid around its significant nodes.

    Every gap next to a node above :data:`RATE_FLOOR` of the peak that is
    wider than the node's refinement spacing ``fine`` is filled uniformly.
    If that needs more than :data:`MAX_REFINED_POINTS` points, all spacings
    grow by the same factor.
    """
    significant = values > RATE_FLOOR * float(values.max())
    # Rounding slack: a grid already at its refinement spacing gains no node.
    ratios = np.diff(times) / fine[:-1] * (1.0 - 1e-9)
    gaps = [
        (k, math.ceil(float(ratios[k])) - 1)
        for k in range(times.size - 1)
        if (significant[k] or significant[k + 1]) and ratios[k] > 1.0
    ]
    total = sum(count for _, count in gaps)
    scale = max(1.0, total / MAX_REFINED_POINTS)
    added = [
        np.linspace(times[k], times[k + 1], math.floor(count / scale) + 2)[1:-1]
        for k, count in gaps
    ]
    if not added:
        return times, values
    extra = np.concatenate(added)
    merged = np.concatenate([times, extra])
    order = np.argsort(merged, kind="stable")
    new_values = np.array([rate(float(t)) for t in extra])
    return merged[order], np.concatenate([values, new_values])[order]


def _detect(
    rate: Callable[[float], float], window: tuple[float, float], screen: Screen
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """The rate on the detection grid of ``window``.

    Returns
    -------
    tuple
        Times, rates and the refinement spacing of each node (infinite where
        the next node is across a gap the screen ruled negligible).

    Raises
    ------
    PcConvergenceError
        If detection would need more than :data:`MAX_SCAN_POINTS` points, or
        the rate is not negligible where the screen ruled it negligible.
    """
    start, end = window
    planned = _detection_grids(window, screen(window))
    grids = [grid for grid, _ in planned]
    spacings = [fine for _, fine in planned]
    # Window edges outside every candidate are scanned too, as one-point
    # blocks, so that the scan always spans the window.
    if not grids or grids[0][0] > start:
        grids.insert(0, np.array([start]))
        spacings.insert(0, math.inf)
    if grids[-1][-1] < end:
        grids.append(np.array([end]))
        spacings.append(math.inf)
    times = np.concatenate(grids)
    values = np.array([rate(float(t)) for t in times])
    fine = np.concatenate(
        [
            np.full(grid.size, spacing)
            for grid, spacing in zip(grids, spacings, strict=True)
        ]
    )
    firsts = np.cumsum([0, *(grid.size for grid in grids[:-1])])
    lasts = firsts + np.array([grid.size for grid in grids]) - 1
    # Gaps between blocks are never refined: their ends are negligible.
    fine[lasts] = math.inf
    peak = float(values.max())
    last = times.size - 1
    interior = set(firsts[firsts > 0].tolist()) | set(lasts[lasts < last].tolist())
    for index in sorted(interior):
        if values[index] > RATE_FLOOR * peak:
            raise PcConvergenceError(
                f"the collision rate at t = {times[index]:.6g} s from TCA is "
                f"{values[index] / peak:.1e} of its peak where the screen "
                "ruled it negligible"
            )
    return times, values, fine


def _segments(
    rate: Callable[[float], float],
    times: FloatArray,
    values: FloatArray,
    fine: FloatArray,
) -> RateScan:
    """Refine a detection scan around its significant nodes; find segments."""
    if float(values.max()) <= 0.0:
        return RateScan(times, values, ())
    times, values = _refine(rate, times, values, fine)
    peak = float(values.max())
    last = times.size - 1
    segments = tuple(
        (max(first - 1, 0), min(final + 1, last))
        for first, final in regions(values > RATE_FLOOR * peak)
    )
    return RateScan(times, values, segments)


def locate(
    rate: Callable[[float], float], window: tuple[float, float], screen: Screen
) -> RateScan:
    """Scan ``rate`` over the screen's candidates in ``window``; find segments.

    A detection grid (see :data:`DETECTION_SIGMAS`) over the candidates finds
    every encounter; the grid is then refined to an eighth of a crossing time
    around the nodes where the rate is significant, to place the segment
    bounds and the Gauss-Legendre breakpoints.

    Raises
    ------
    PcConvergenceError
        If detection would need more than :data:`MAX_SCAN_POINTS` points, or
        the rate is not negligible where the screen ruled it negligible.
    """
    return _segments(rate, *_detect(rate, window, screen))


def locate_auto(
    rate: Callable[[float], float],
    half_width: float,
    limit: float,
    screen: Screen,
    centre: tuple[float, float] = (0.0, 0.0),
) -> RateScan:
    """Scan a window that grows until the rate at its edges vanishes.

    The window is ``centre`` widened by ``half`` on both sides; ``half``
    starts at ``half_width`` and doubles, up to ``limit``, while the rate at
    an edge is not negligible or the rate is 0 everywhere scanned (an
    encounter may lie beyond). Only the detection grid is scanned while the
    window grows (the edges are on it); the final window is then refined as
    in :func:`locate`.

    Parameters
    ----------
    rate
        Collision rate [1/s] as a function of time from TCA [s].
    half_width
        Initial widening [s].
    limit
        Largest widening [s] (the window then spans the shorter orbital
        period).
    screen
        Candidate intervals of a window.
    centre
        Stretch ``(a, b)`` the window always covers [s from TCA].
    """
    a, b = centre
    half = min(half_width, limit)
    while True:
        times, values, fine = _detect(rate, (a - half, b + half), screen)
        if (values.max() > 0.0 and _edge(values) < RATE_FLOOR) or half >= limit:
            return _segments(rate, times, values, fine)
        half = min(2.0 * half, limit)


START_ORDER = 32
"""First Gauss-Legendre order of the time integral of one piece."""

MAX_ORDER = 1024
"""Largest Gauss-Legendre order of the time integral of one piece."""


def _gauss_legendre_integral(
    rate: Callable[[float], float], start: float, end: float, rtol: float
) -> tuple[float, float]:
    """Integral of ``rate`` over ``[start, end]`` by Gauss-Legendre rules.

    The order doubles from :data:`START_ORDER` until two orders agree to
    ``rtol`` or :data:`MAX_ORDER` is reached. Unlike QUADPACK's error
    estimate, the comparison is not upset by the ~1e-8 relative jitter of
    the rate (its sphere rule is rebuilt at every time).

    Returns
    -------
    tuple
        The integral and the difference of the last two orders (infinite if
        :data:`MAX_ORDER` allows a single order).
    """
    half, middle = 0.5 * (end - start), 0.5 * (end + start)

    def rule(order: int) -> float:
        nodes, weights = gauss_legendre(order)
        values = np.array([rate(middle + half * float(x)) for x in nodes])
        return half * float(weights @ values)

    order = START_ORDER
    value, difference = rule(order), math.inf
    while order < MAX_ORDER:
        order *= 2
        previous, value = value, rule(order)
        difference = abs(value - previous)
        if difference <= rtol * abs(value):
            break
    return value, difference


def integrate_segments(
    rate: Callable[[float], float], found: RateScan, *, rtol: float = QUAD_RTOL
) -> RateIntegral:
    """Integrate ``rate`` over the segments of a scan.

    Each segment is split at the local maxima of the scan and every piece is
    integrated by Gauss-Legendre rules of doubling order.
    """
    segments: list[Segment] = []
    shortfalls: list[float] = []
    for lo, hi in found.segments:
        inner = found.values[lo : hi + 1]
        cuts = [
            float(found.times[lo + j])
            for j in range(1, inner.size - 1)
            if inner[j] > inner[j - 1] and inner[j] >= inner[j + 1]
        ][:MAX_BREAKPOINTS]
        start, end = float(found.times[lo]), float(found.times[hi])
        pieces = [
            _gauss_legendre_integral(rate, a, b, rtol)
            for a, b in pairwise([start, *cuts, end])
        ]
        shortfalls += [d for v, d in pieces if not d <= rtol * abs(v)]
        segments.append(Segment(start, end, math.fsum(v for v, _ in pieces)))
    total = math.fsum(s.nc for s in segments)
    worst = max(shortfalls, default=0.0)
    agreement = worst / abs(total) if total != 0.0 else (math.inf if worst else 0.0)
    return RateIntegral(
        total, tuple(segments), found.window, found.edge, found.peak, agreement
    )
