"""The time integral of a collision rate over the encounter segments of a
scan (:mod:`keepout.pc._timeline`): each segment split at its local maxima
and every piece integrated by nested Clenshaw-Curtis rules of doubling
order."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from typing import Final

import numpy as np

from keepout._types import FloatArray, read_only
from keepout.pc._timeline import Rate, RateScan

MAX_BREAKPOINTS: Final = 50
"""Largest number of local maxima a segment is split at."""

CC_START_ORDER: Final = 32
"""Intervals of the first Clenshaw-Curtis rule of the time integral of one
piece (``CC_START_ORDER + 1`` nodes)."""

CC_MAX_ORDER: Final = 1024
"""Intervals of the finest Clenshaw-Curtis rule of one piece."""

STAGNATION_RTOL: Final = 1e-4
"""Relative difference of two rules below which a difference that no longer
halves is taken as the rate's own noise (its sphere quadrature, ~1e-5 of the
peak): doubling further would not reduce it."""


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class RateIntegral:
    """The time integral of a collision rate over a window.

    Parameters
    ----------
    nc
        Integral over the whole window.
    segments
        Where the rate is not negligible.
    scan
        The scan the integral was computed on (its window, peak and edge).
    worst_disagreement
        Largest difference of the last two Clenshaw-Curtis rules of a piece
        that never agreed to ``rtol``, relative to ``nc`` (0 if every piece
        agreed; infinite if ``nc`` is 0 or a piece had a single order).
    """

    nc: float
    segments: tuple[Segment, ...]
    scan: RateScan
    worst_disagreement: float = 0.0


@lru_cache(maxsize=8)
def clenshaw_curtis(order: int) -> tuple[FloatArray, FloatArray]:
    """Read-only Clenshaw-Curtis nodes ``cos(k pi / order)`` and weights.

    ``order`` is even; the rule has ``order + 1`` nodes on ``[-1, 1]``, from
    ``+1`` down to ``-1``, and those of ``order / 2`` are its even-indexed
    ones, so doubling the order reuses every value already computed.
    """
    angles = np.pi * np.arange(order + 1) / order
    j = np.arange(1, order // 2 + 1)
    b = np.where(j == order // 2, 1.0, 2.0)
    sums = (b / (4.0 * j * j - 1.0)) @ np.cos(2.0 * np.outer(j, angles))
    weights = np.where((angles == 0.0) | (angles == np.pi), 1.0, 2.0) / order
    weights = weights * (1.0 - sums)
    nodes = np.cos(angles)
    read_only(nodes, weights)
    return nodes, weights


def _nested_integral(
    rate: Rate, start: float, end: float, rtol: float
) -> tuple[float, float]:
    """Integral of ``rate`` over ``[start, end]`` by nested Clenshaw-Curtis rules.

    The order doubles from :data:`CC_START_ORDER` until two rules agree to
    ``rtol``, their difference stops halving below :data:`STAGNATION_RTOL`
    (the rate's noise floor), or :data:`CC_MAX_ORDER` is reached. Each doubling
    evaluates the rate only at the new nodes.

    Returns
    -------
    tuple
        The integral and the difference of the last two rules (infinite if
        :data:`CC_MAX_ORDER` allows a single rule).
    """
    half, middle = 0.5 * (end - start), 0.5 * (end + start)

    def at(nodes: FloatArray) -> FloatArray:
        return rate(middle + half * nodes)

    order = CC_START_ORDER
    nodes, weights = clenshaw_curtis(order)
    values = at(nodes)
    value = half * float(weights @ values)
    difference = previous_difference = math.inf
    while order < CC_MAX_ORDER:
        order *= 2
        nodes, weights = clenshaw_curtis(order)
        merged = np.empty(order + 1)
        merged[::2], merged[1::2] = values, at(nodes[1::2])
        values = merged
        previous, value = value, half * float(weights @ values)
        previous_difference, difference = difference, abs(value - previous)
        if difference <= rtol * abs(value):
            break
        if (
            difference <= STAGNATION_RTOL * abs(value)
            and difference > 0.5 * previous_difference
        ):
            break
    return value, difference


def integrate_segments(rate: Rate, scan: RateScan, *, rtol: float) -> RateIntegral:
    """The integral of ``rate`` over the segments of a scan.

    Each segment is split at the local maxima of the scan and every piece is
    integrated by nested Clenshaw-Curtis rules of doubling order until two
    agree to ``rtol``.
    """
    segments: list[Segment] = []
    shortfalls: list[float] = []
    for first, last in scan.index_ranges:
        inner = scan.values[first : last + 1]
        maxima = (inner[1:-1] > inner[:-2]) & (inner[1:-1] >= inner[2:])
        cuts = scan.times[first + 1 + np.flatnonzero(maxima)][:MAX_BREAKPOINTS]
        start, end = float(scan.times[first]), float(scan.times[last])
        pieces = [
            _nested_integral(rate, a, b, rtol)
            for a, b in pairwise([start, *cuts.tolist(), end])
        ]
        # Written ``not d <= ...`` so that a NaN difference counts as a shortfall.
        shortfalls += [d for v, d in pieces if not d <= rtol * abs(v)]
        segments.append(Segment(start, end, math.fsum(v for v, _ in pieces)))
    total = math.fsum(s.nc for s in segments)
    worst = max(shortfalls, default=0.0)
    disagreement = worst / abs(total) if total != 0.0 else (math.inf if worst else 0.0)
    return RateIntegral(total, tuple(segments), scan, disagreement)
