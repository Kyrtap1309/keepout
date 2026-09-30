"""Hall's 3D-Nc collision rate and the scan that locates its encounters.

The rate ``Ncdot(t)`` [1/s] is defined in :mod:`keepout.pc.hall3d`; it is
shared by :func:`~keepout.pc.hall3d.hall3d`, which integrates it, and
:func:`~keepout.pc.montecarlo.montecarlo`, which samples trials only where
it is not negligible.
"""

from __future__ import annotations

import math
from typing import TypeAlias

import numpy as np
from scipy import special

from keepout._types import FloatArray
from keepout.conjunction import Conjunction
from keepout.errors import GeometryError, PcConvergenceError
from keepout.pc._orbit import OrbitUncertainty
from keepout.pc._plane import require_known_covariance
from keepout.pc._sphere import sphere_rule_principal
from keepout.pc._timeline import (
    Candidate,
    RateScan,
    locate,
    locate_auto,
    regions,
    straight_line_half_window,
)
from keepout.pc.result import Method

OVERLAP_TOLERANCE = 1e-6
"""Convergence of the peak-overlap iteration: squared Mahalanobis step of
the peak position (Hall 2021, sec. VII.B step 7)."""

OVERLAP_MAX_ITERATIONS = 100

NEGLIGIBLE_DISTANCE = 1500.0
"""Squared Mahalanobis distance from the hard-body sphere to the mean relative
position beyond which the rate underflows (``exp(-750)``) and is taken as 0."""

FALLBACK_TOLERANCE = 1e-10
"""Largest rate, relative to the peak, allowed where the peak-overlap
iteration failed and the mean-centred expansion was used instead."""

VARIANCE_FLOOR = 1e-4
"""Smallest position standard deviation used when inverting a covariance, as
a fraction of the hard-body radius (as in 2-D Pc and CARA's ``Fclip``): a
perfectly known object then has a finite, tiny extent."""

SCREEN_MIN_CELLS, SCREEN_MAX_CELLS = 64, 50_000
"""Bounds on the number of screen cells in a window."""

SCREEN_PERIOD_FRACTION = 1.0 / 64.0
"""Largest screen cell, as a fraction of the shorter orbital period."""

SCREEN_VARIANCE_MARGIN = 2.0
"""Factor on the largest position variance of neighbouring screen cells."""

SCREEN_CHUNK = 4096
"""Screen cells evaluated in one vectorised call."""

_SQRT_PI = math.sqrt(math.pi)
_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)

_TINY_VARIANCE = 1e-300
"""Floor [m^2/s^2] of the conditional radial-velocity variance, which divides
the mean radial velocity in the flux."""

Gaussians: TypeAlias = tuple[FloatArray, FloatArray, FloatArray, FloatArray]
"""Mean and covariance of both objects' states at one time: ``x1, p1, x2, p2``."""


def _floored_inverse(matrix: FloatArray, floor: float) -> FloatArray:
    """Inverse of a symmetric matrix with its eigenvalues raised to ``floor``.

    Built from the eigen-decomposition, so a tiny floored eigenvalue next to
    a huge one is not lost to rounding as it would be by inverting the
    reconstructed matrix.
    """
    values, vectors = np.linalg.eigh(matrix)
    result: FloatArray = (vectors / np.maximum(values, floor)) @ vectors.T
    return result


def _peak_overlap(
    primary: OrbitUncertainty,
    secondary: OrbitUncertainty,
    t: float,
    floor: float,
    mean_centred: Gaussians,
) -> Gaussians | None:
    """Converged expansion-centre Gaussians of both objects at ``t``, or None.

    ``floor`` is the smallest position variance used when inverting;
    ``mean_centred`` (the expansion about the mean elements) is the first
    iterate.
    """
    x1, p1, x2, p2 = mean_centred
    previous: FloatArray | None = None
    for _ in range(OVERLAP_MAX_ITERATIONS):
        inverse1, inverse2 = (_floored_inverse(p[:3, :3], floor) for p in (p1, p2))
        precision = inverse1 + inverse2
        peak = np.linalg.solve(precision, inverse1 @ x1[:3] + inverse2 @ x2[:3])
        if previous is not None:
            step = peak - previous
            if float(step @ precision @ step) <= OVERLAP_TOLERANCE:
                return x1, p1, x2, p2
        previous = peak
        centres = [
            orbit.centre_at(
                np.concatenate([peak, x[3:] + p[3:, :3] @ inv @ (peak - x[:3])]), t
            )
            for orbit, x, p, inv in (
                (primary, x1, p1, inverse1),
                (secondary, x2, p2, inverse2),
            )
        ]
        (x1, p1), (x2, p2) = (
            orbit.expansion(centre, t)
            for orbit, centre in zip((primary, secondary), centres, strict=True)
        )
    return None


class CollisionRate:
    """Hall's 3D-Nc collision rate ``Ncdot(t)`` [1/s] of a conjunction.

    Parameters
    ----------
    primary, secondary
        The two orbits with their uncertainty.
    radius
        Combined hard-body radius [m].
    """

    def __init__(
        self, primary: OrbitUncertainty, secondary: OrbitUncertainty, radius: float
    ) -> None:
        self.primary, self.secondary, self.radius = primary, secondary, radius
        self.floor = (VARIANCE_FLOOR * radius) ** 2
        self.evaluations = 0
        self.floored = math.inf
        """Smallest combined position variance [m^2] raised to :attr:`floor`
        by :meth:`_rate` (infinite if the floor never engaged)."""
        self.fallbacks: list[tuple[float, float]] = []
        """Times [s] and rates where the peak-overlap iteration failed."""

    def _mean_centred(self, t: float) -> Gaussians:
        x1, p1 = self.primary.expansion(self.primary.elements, t)
        x2, p2 = self.secondary.expansion(self.secondary.elements, t)
        return x1, p1, x2, p2

    def _negligible(
        self, x1: FloatArray, p1: FloatArray, x2: FloatArray, p2: FloatArray
    ) -> bool:
        largest = float(np.linalg.eigvalsh(p1[:3, :3] + p2[:3, :3])[-1])
        gap = max(float(np.linalg.norm(x2[:3] - x1[:3])) - self.radius, 0.0)
        return gap * gap > NEGLIGIBLE_DISTANCE * largest

    def __call__(self, t: float) -> float:
        self.evaluations += 1
        mean_centred = self._mean_centred(t)
        if self._negligible(*mean_centred):
            return 0.0
        try:
            converged = _peak_overlap(
                self.primary, self.secondary, t, self.floor, mean_centred
            )
        except (np.linalg.LinAlgError, ArithmeticError, ValueError, GeometryError):
            # A diverging iteration can leave the space of elliptic orbits.
            converged = None
        if converged is not None and all(np.all(np.isfinite(a)) for a in converged):
            return self._checked_rate(t, converged)
        value = self._checked_rate(t, mean_centred)
        self.fallbacks.append((t, value))
        return value

    def _checked_rate(self, t: float, gaussians: Gaussians) -> float:
        """:meth:`_rate`, with a failure raised as :class:`PcConvergenceError`."""
        try:
            value = self._rate(*gaussians)
        except (np.linalg.LinAlgError, ArithmeticError, ValueError) as error:
            raise PcConvergenceError(
                f"Hall 3D-Nc: the collision rate at t = {t:.6g} s from TCA "
                f"cannot be computed: {error}"
            ) from error
        if not math.isfinite(value):
            raise PcConvergenceError(
                f"Hall 3D-Nc: the collision rate at t = {t:.6g} s from TCA is "
                f"not finite ({value})"
            )
        return value

    def _screen_cells(
        self, times: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        """Gap, relative speed, largest variance, crossing time and time width.

        All at ``times``, from the mean-centred expansion that
        :meth:`_negligible` uses. The time width is the standard deviation
        ``1 / (|v| sqrt(u' A^-1 u))`` of the straight-line passage through
        the density, with ``A`` floored as in :meth:`_rate`.
        """
        x1, p1 = self.primary.mean_expansions(times)
        x2, p2 = self.secondary.mean_expansions(times)
        dr, dv = x2[:, :3] - x1[:, :3], x2[:, 3:] - x1[:, 3:]
        a = p1[:, :3, :3] + p2[:, :3, :3]
        gap = np.maximum(np.linalg.norm(dr, axis=1) - self.radius, 0.0)
        speed = np.linalg.norm(dv, axis=1)
        values, vectors = np.linalg.eigh(a)
        largest = values[:, -1]
        direction = dv / np.where(speed > 0.0, speed, 1.0)[:, None]
        sigma = np.sqrt(
            np.maximum(np.einsum("ij,ijk,ik->i", direction, a, direction), 0.0)
        )
        along = np.einsum("ij,ijk->ik", direction, vectors)
        precision = np.einsum(
            "ik,ik->i", along * along, 1.0 / np.maximum(values, self.floor)
        )
        with np.errstate(divide="ignore"):
            crossing = np.where(speed > 0.0, (sigma + self.radius) / speed, math.inf)
            width = np.where(speed > 0.0, 1.0 / (speed * np.sqrt(precision)), math.inf)
        return gap, speed, largest, crossing, width

    def screen(self, window: tuple[float, float]) -> list[Candidate]:
        """Candidate intervals of ``window``: where the rate may not be 0.

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
        :data:`NEGLIGIBLE_DISTANCE`, :meth:`_negligible` holds throughout the
        cell and the rate is exactly 0. The cell size is the time the mean
        relative position takes to cross that reach, within
        :data:`SCREEN_MIN_CELLS` and :data:`SCREEN_MAX_CELLS` cells.
        """
        start, end = window
        span = end - start
        middle = np.array([0.5 * (start + end)])
        _, speed, largest, _, _ = self._screen_cells(middle)
        extent = reach(float(largest[0]), self.radius)
        natural = extent / float(speed[0]) if speed[0] > 0.0 else math.inf
        longest = SCREEN_PERIOD_FRACTION * min(
            self.primary.period, self.secondary.period
        )
        cell = max(
            span / SCREEN_MAX_CELLS, min(natural, span / SCREEN_MIN_CELLS, longest)
        )
        count = math.ceil(span / cell)
        cell = span / count
        centres = start + (np.arange(count, dtype=np.float64) + 0.5) * cell
        parts = [
            self._screen_cells(centres[i : i + SCREEN_CHUNK])
            for i in range(0, count, SCREEN_CHUNK)
        ]
        gap, speed, largest, crossing, width = (
            np.concatenate(p) for p in zip(*parts, strict=True)
        )
        half = 0.5 * cell
        acceleration = self.primary.max_acceleration + self.secondary.max_acceleration
        closest = np.maximum(gap - speed * half - 0.5 * acceleration * half * half, 0.0)
        padded = np.concatenate([largest[:1], largest, largest[-1:]])
        envelope = SCREEN_VARIANCE_MARGIN * np.maximum.reduce(
            [padded[:-2], padded[1:-1], padded[2:]]
        )
        possible = closest * closest <= NEGLIGIBLE_DISTANCE * envelope
        return [
            Candidate(
                start if first == 0 else start + first * cell,
                end if last == count - 1 else start + (last + 1) * cell,
                float(np.min(crossing[first : last + 1])),
                float(np.min(width[first : last + 1])),
            )
            for first, last in regions(possible)
        ]

    def _rate(
        self, x1: FloatArray, p1: FloatArray, x2: FloatArray, p2: FloatArray
    ) -> float:
        mean, covariance = x2 - x1, p1 + p2
        r_mean, v_mean = mean[:3], mean[3:]
        values, vectors = np.linalg.eigh(covariance[:3, :3])
        if values[0] < self.floor:
            self.floored = min(self.floored, float(values[0]))
        # The floored position covariance A, used through its eigen-decomposition.
        floored = np.maximum(values, self.floor)
        a_inverse = (vectors / floored) @ vectors.T
        log_det = float(np.sum(np.log(floored)))
        b, c = covariance[3:, :3], covariance[3:, 3:]
        gain = b @ a_inverse
        conditional = c - gain @ b.T
        nodes, weights = sphere_rule_principal(floored, vectors, r_mean, self.radius)
        offset = self.radius * nodes - r_mean
        mahalanobis = np.einsum("ij,jk,ik->i", offset, a_inverse, offset)
        density = np.exp(-0.5 * mahalanobis - 0.5 * log_det) / (2.0 * math.pi) ** 1.5
        sigma = np.sqrt(
            np.maximum(
                np.einsum("ij,jk,ik->i", nodes, conditional, nodes), _TINY_VARIANCE
            )
        )
        x = np.einsum("ij,ij->i", nodes, v_mean + offset @ gain.T) / (_SQRT_2 * sigma)
        flux = sigma / _SQRT_2PI * (np.exp(-x * x) - _SQRT_PI * x * special.erfc(x))
        return self.radius**2 * float(np.sum(weights * density * flux))


def reach(largest_variance: float, radius: float) -> float:
    """Distance [m] beyond which :meth:`CollisionRate._negligible` holds."""
    return math.sqrt(NEGLIGIBLE_DISTANCE * largest_variance) + radius


def collision_rate(conjunction: Conjunction, method: Method) -> CollisionRate:
    """The 3D-Nc collision rate of a conjunction with both covariances known.

    Raises
    ------
    CovarianceError
        If a covariance is unknown; the message names ``method``.
    GeometryError
        If an orbit is not elliptic.
    """
    require_known_covariance(conjunction.unknown_covariance, method)
    return CollisionRate(
        OrbitUncertainty.of(conjunction.primary),
        OrbitUncertainty.of(conjunction.secondary),
        conjunction.hard_body_radius,
    )


def encounter_scan(
    conjunction: Conjunction, rate: CollisionRate, window: tuple[float, float] | None
) -> RateScan:
    """Scan the rate over ``window`` or an automatically sized one.

    The automatic window covers TCA (t = 0) and the straight-line closest
    approach of the states, ``t* = -(r . v) / |v|^2``, widened on both sides
    by the straight-line half-window of :func:`straight_line_half_window`, doubled
    until the rate at the edges is negligible (or, while the rate is 0
    everywhere scanned, until an encounter appears). ``t*`` is clipped to
    half the shorter orbital period ``P`` and the window is never wider than
    ``P``, so no encounter of the next revolution is counted twice.

    Raises
    ------
    ValueError
        If ``window`` is not finite with ``start < end``.
    PcConvergenceError
        If the encounters in the window cannot be resolved.
    """
    if window is not None:
        start, end = (float(v) for v in window)
        if not (math.isfinite(start) and math.isfinite(end) and start < end):
            raise ValueError(f"window must be finite with start < end, got {window}")
        return locate(rate, (start, end), rate.screen)
    half = straight_line_half_window(
        conjunction.relative_velocity,
        rate.primary.position_covariance + rate.secondary.position_covariance,
        conjunction.hard_body_radius,
    )
    period = min(rate.primary.period, rate.secondary.period)
    closest = min(
        max(straight_line_closest_time(conjunction), -0.5 * period), 0.5 * period
    )
    centre = (min(0.0, closest), max(0.0, closest))
    limit = 0.5 * (period - (centre[1] - centre[0]))
    return locate_auto(rate, half, limit, rate.screen, centre=centre)


def straight_line_closest_time(conjunction: Conjunction) -> float:
    """Time [s from TCA] of the straight-line closest approach of the states.

    ``-(r . v) / |v|^2`` with the relative position ``r`` and velocity ``v``;
    0 at zero relative velocity.
    """
    velocity = conjunction.relative_velocity
    speed2 = float(velocity @ velocity)
    if speed2 == 0.0:
        return 0.0
    return -float(conjunction.relative_position @ velocity) / speed2


def check_fallbacks(fallbacks: list[tuple[float, float]], found: RateScan) -> None:
    """Raise if the peak-overlap iteration failed where the rate may matter.

    A failure inside an encounter segment, or within one grid step of it, is
    fatal whatever the mean-centred rate there (that rate is what the
    iteration replaces, and with a large along-track uncertainty it can be
    ~0 where the true rate is near its peak); points where the mean-centred
    gap is negligible never reach the iteration. Elsewhere, in the far tail,
    a failure is fatal only if the mean-centred rate exceeds
    :data:`FALLBACK_TOLERANCE` of the peak.
    """
    times, peak = found.times, found.peak
    zones = [
        (
            float(times[lo] - (times[lo + 1] - times[lo])),
            float(times[hi] + (times[hi] - times[hi - 1])),
        )
        for lo, hi in found.segments
        if hi > lo
    ]
    for t, value in fallbacks:
        inside = any(a <= t <= b for a, b in zones)
        if inside or value > FALLBACK_TOLERANCE * peak:
            where = (
                "inside an encounter segment"
                if inside
                else f"where the collision rate is {value / peak:.1e} of its peak"
            )
            raise PcConvergenceError(
                f"Hall 3D-Nc: the peak-overlap iteration did not converge at "
                f"t = {t:.6g} s from TCA, {where}"
            )
