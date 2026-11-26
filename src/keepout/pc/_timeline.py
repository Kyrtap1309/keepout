"""Time windows and the encounter segments of a collision rate.

A cheap screen first marks the stretches of the window where the rate can be
non-negligible (candidate intervals). Each is scanned on a detection grid
fine enough that no encounter can fall between two nodes
(:data:`DETECTION_SIGMAS`), which is then refined to an eighth of a crossing
time around the nodes where the rate is significant. The stretches where the
rate is not negligible become encounter segments (integrated by
:mod:`keepout.pc._quadrature`). Without an explicit window the half-width
starts at the linear-motion bound and doubles until the rate at both edges
is negligible, but the window never spans more than the shorter orbital
period.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, NamedTuple, TypeAlias

import numpy as np

from keepout._types import BoolArray, FloatArray, IndexArray
from keepout.errors import PcConvergenceError

RATE_FLOOR: Final = 1e-16
"""Rate, relative to the peak, below which the rate is negligible."""

SIGMA_REACH: Final = math.sqrt(2.0 * math.log(1.0 / RATE_FLOOR))
"""Standard deviations after which a Gaussian falls to :data:`RATE_FLOOR`."""

MIN_SCAN_POINTS: Final = 401
"""Points of a scan over the whole window: the coarsest spacing is
``window / (MIN_SCAN_POINTS - 1)``."""

MAX_SCAN_POINTS: Final = 20001
"""Largest number of detection-grid points over all candidate intervals."""

MAX_REFINED_POINTS: Final = 20001
"""Largest number of points added by refinement; beyond it the refinement is
coarser (detection is already guaranteed by the detection grid)."""

DETECTION_LEVEL: Final = 1e-8
"""Smallest encounter peak, relative to the largest, the scan must detect."""

DETECTION_SIGMAS: Final = (
    0.5 * 2.0 * math.sqrt(2.0 * math.log(DETECTION_LEVEL / RATE_FLOOR))
)
"""Detection-grid spacing in time standard deviations of the rate (6.07).

Over one straight-line crossing the rate is, as a function of time, a
Gaussian of standard deviation ``tau = 1 / (|v| sqrt(u' A^-1 u))`` (``u`` the
unit relative velocity, ``A`` the combined position covariance) convolved
with the passage through the hard-body sphere, which only widens it. An
encounter whose peak is at least :data:`DETECTION_LEVEL` of the largest peak
is therefore above :data:`RATE_FLOOR` of it over at least
``2 tau sqrt(2 ln(DETECTION_LEVEL / RATE_FLOOR))`` = 12.1 ``tau``, and a
grid of that spacing already puts a node inside it. The guarantee rests on
``tau`` being the minimum over the candidate interval
(:attr:`Candidate.width`, the smallest over its screen cells) and on the
straight-line model; the grid is half that wide (the leading ``0.5``) as a
margin for the departure of curved two-body motion from a straight line, a
heuristic, not a bound. A missed encounter contributes at most
~``DETECTION_LEVEL`` of the Nc of the largest one."""

REFINE_CROSSING_FRACTION: Final = 1.0 / 8.0
"""Refinement spacing around significant nodes, as a fraction of the
crossing time."""

_ROUNDING_SLACK: Final = 1e-9
"""Relative slack on grid gaps: a grid already at its refinement spacing
gains no node from rounding."""


@dataclass(frozen=True, slots=True)
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


Rate: TypeAlias = Callable[[FloatArray], FloatArray]
"""A collision rate [1/s] evaluated at an array of times [s from TCA]."""

Screen: TypeAlias = Callable[[tuple[float, float]], list[Candidate]]
"""Candidate intervals of a window, in increasing time and disjoint."""


def regions(significant: BoolArray) -> list[tuple[int, int]]:
    """Index ranges ``[first, last]`` of consecutive True values."""
    indices = np.flatnonzero(significant)
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.concatenate([[indices[0]], indices[breaks + 1]])
    ends = np.concatenate([indices[breaks], [indices[-1]]])
    return [(int(a), int(b)) for a, b in zip(starts, ends, strict=True)]


@dataclass(frozen=True, eq=False, slots=True)
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
    index_ranges
        Index ranges ``[first, last]`` (inclusive, padded by one grid step) of
        the stretches where the rate is not negligible; each lies within one
        candidate interval.
    """

    times: FloatArray
    values: FloatArray
    index_ranges: tuple[tuple[int, int], ...]

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


def negligible_note(scan: RateScan) -> str:
    """Diagnostic for a scan on which the rate is 0 everywhere."""
    start, end = scan.window
    return f"collision rate negligible over the whole window [{start:.6g}, {end:.6g}] s"


def _edge(values: FloatArray) -> float:
    """Largest of the first and last value, relative to the largest (0 if none)."""
    peak = float(values.max())
    return float(max(values[0], values[-1])) / peak if peak > 0.0 else 0.0


class _GridPlan(NamedTuple):
    """A planned detection grid: ``points`` nodes uniformly over ``[start, end]``,
    with the spacing its refinement needs."""

    start: float
    end: float
    points: int
    refine_spacing: float


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
    plan: list[_GridPlan] = []
    finest = math.inf
    for candidate in candidates:
        a, b = max(candidate.start, start), min(candidate.end, end)
        if b <= a:
            continue
        spacing = min(DETECTION_SIGMAS * candidate.width, coarsest)
        finest = min(finest, spacing)
        count = math.ceil((b - a) / spacing) + 1
        refine_spacing = min(candidate.crossing * REFINE_CROSSING_FRACTION, spacing)
        plan.append(_GridPlan(a, b, count, refine_spacing))
    if sum(grid.points for grid in plan) > MAX_SCAN_POINTS:
        raise PcConvergenceError(
            f"cannot resolve the collision rate over the window [{start:.6g}, "
            f"{end:.6g}] s: detecting its possible encounters needs a scan "
            f"spacing of {finest:.3g} s, more than {MAX_SCAN_POINTS} points; "
            "pass a shorter window around the encounter, e.g. "
            "compute_pc(conjunction, 'hall3d', window=(a, b)) (or 'montecarlo')"
        )
    return [
        (np.linspace(grid.start, grid.end, grid.points), grid.refine_spacing)
        for grid in plan
    ]


def _refine(
    rate: Rate,
    times: FloatArray,
    values: FloatArray,
    fine: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """The grid, refined around its significant nodes.

    Every gap next to a node above :data:`RATE_FLOOR` of the peak that is
    wider than the node's refinement spacing ``fine`` is filled uniformly.
    If that needs more than :data:`MAX_REFINED_POINTS` points, all spacings
    grow by the same factor.
    """
    significant = values > RATE_FLOOR * float(values.max())
    ratios = np.diff(times) / fine[:-1] * (1.0 - _ROUNDING_SLACK)
    gaps = np.flatnonzero((significant[:-1] | significant[1:]) & (ratios > 1.0))
    counts = np.ceil(ratios[gaps]) - 1.0
    scale = max(1.0, float(counts.sum()) / MAX_REFINED_POINTS)
    inner = np.floor(counts / scale).astype(np.intp)
    if not inner.any():
        return times, values
    # Each gap's inner points of np.linspace(start, end, n + 2), n = inner,
    # computed as it does: start + j * ((end - start) / (n + 1)).
    owner = np.repeat(gaps, inner)
    divisions = np.repeat(inner + 1, inner)
    j = np.arange(1, owner.size + 1) - np.repeat(np.cumsum(inner) - inner, inner)
    start = times[owner]
    extra = j * ((times[owner + 1] - start) / divisions) + start
    merged = np.concatenate([times, extra])
    order = np.argsort(merged, kind="stable")
    new_values = rate(extra)
    return merged[order], np.concatenate([values, new_values])[order]


def _detect(
    rate: Rate, window: tuple[float, float], screen: Screen
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Times, rates and refinement spacings (infinite where the next node is
    across a gap the screen ruled negligible) of the detection grid of
    ``window``.

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
    values = rate(times)
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
    _require_negligible_gaps(times, values, firsts, lasts)
    return times, values, fine


def _require_negligible_gaps(
    times: FloatArray, values: FloatArray, firsts: IndexArray, lasts: IndexArray
) -> None:
    """A check that the rate is negligible at the interior block edges.

    Raises
    ------
    PcConvergenceError
        If the rate at a block edge next to a gap the screen ruled negligible
        is above :data:`RATE_FLOOR` of the peak.
    """
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


def _refined_scan(
    rate: Rate,
    times: FloatArray,
    values: FloatArray,
    fine: FloatArray,
) -> RateScan:
    """A detection scan refined around its significant nodes, with its segments."""
    if float(values.max()) <= 0.0:
        return RateScan(times, values, ())
    times, values = _refine(rate, times, values, fine)
    peak = float(values.max())
    last = times.size - 1
    index_ranges = tuple(
        (max(first - 1, 0), min(last_index + 1, last))
        for first, last_index in regions(values > RATE_FLOOR * peak)
    )
    return RateScan(times, values, index_ranges)


def locate(rate: Rate, window: tuple[float, float], screen: Screen) -> RateScan:
    """The scan of ``rate`` over the screen's candidates in ``window``, with segments.

    A detection grid (see :data:`DETECTION_SIGMAS`) over the candidates finds
    every encounter; the grid is then refined to an eighth of a crossing time
    around the nodes where the rate is significant, to place the segment
    bounds and the quadrature breakpoints.

    Raises
    ------
    PcConvergenceError
        If detection would need more than :data:`MAX_SCAN_POINTS` points, or
        the rate is not negligible where the screen ruled it negligible.
    """
    return _refined_scan(rate, *_detect(rate, window, screen))


def locate_auto(
    rate: Rate,
    half_width: float,
    limit: float,
    screen: Screen,
    centre: tuple[float, float] = (0.0, 0.0),
) -> RateScan:
    """The scan of a window grown until the rate at its edges vanishes.

    The window is ``centre`` widened by ``half`` on both sides; ``half``
    starts at ``half_width`` and doubles, up to ``limit``, while the rate at
    an edge is not negligible or the rate is 0 everywhere scanned (an
    encounter may lie beyond). Only the detection grid is scanned while the
    window grows (the edges are on it); the final window is then refined as
    in :func:`locate`.

    Parameters
    ----------
    rate
        Collision rate [1/s] at an array of times from TCA [s].
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
            return _refined_scan(rate, times, values, fine)
        half = min(2.0 * half, limit)
