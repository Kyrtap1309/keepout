"""Hall's 3D-Nc collision rate and the scan that locates its encounters.

The rate ``Ncdot(t)`` [1/s] is defined in :mod:`keepout.pc.hall3d`; it is
shared by :func:`~keepout.pc.hall3d.hall3d`, which integrates it, and
:func:`~keepout.pc.montecarlo.montecarlo`, which samples trials only where
it is not negligible.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final, NamedTuple

import numpy as np
from scipy import special

from keepout._types import BoolArray, FloatArray, IndexArray, from_eigen
from keepout.conjunction import Conjunction
from keepout.errors import GeometryError, PcConvergenceError
from keepout.pc._orbit import OrbitUncertainty
from keepout.pc._plane import SIGMA_FLOOR, SQRT_2, SQRT_2PI, require_known_covariance
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

OVERLAP_TOLERANCE: Final = 1e-6
"""Convergence of the peak-overlap iteration: squared Mahalanobis step of
the peak position (Hall 2021, sec. VII.B step 7)."""

OVERLAP_MAX_ITERATIONS: Final = 100

NEGLIGIBLE_DISTANCE: Final = 1500.0
"""Squared Mahalanobis distance from the hard-body sphere to the mean relative
position beyond which the rate underflows (``exp(-750)``) and is taken as 0."""

FALLBACK_TOLERANCE: Final = 1e-10
"""Largest rate, relative to the peak, allowed where the peak-overlap
iteration failed and the mean-centred expansion was used instead."""

SCREEN_MIN_CELLS, SCREEN_MAX_CELLS = 64, 50_000
"""Bounds on the number of screen cells in a window."""

SCREEN_PERIOD_FRACTION: Final = 1.0 / 64.0
"""Largest screen cell, as a fraction of the shorter orbital period."""

SCREEN_VARIANCE_MARGIN: Final = 2.0
"""Factor on the largest position variance of neighbouring screen cells."""

SCREEN_CHUNK: Final = 4096
"""Screen cells evaluated in one vectorised call."""

_SQRT_PI: Final = math.sqrt(math.pi)

_TINY_VARIANCE: Final = 1e-300
"""Floor [m^2/s^2] of the conditional radial-velocity variance, which divides
the mean radial velocity in the flux."""

_RATE_ERRORS: Final = (np.linalg.LinAlgError, ArithmeticError, ValueError)
"""Failures of the rate computation itself."""

_ITERATION_ERRORS: Final = (*_RATE_ERRORS, GeometryError)
"""Failures of the peak-overlap iteration that fall back to the mean-centred
expansion instead of stopping the computation."""


class Gaussians(NamedTuple):
    """Means and covariances of both objects' states, at one time or a batch.

    Batched: ``x1, x2`` ``(m, 6)`` and ``p1, p2`` ``(m, 6, 6)``.
    """

    x1: FloatArray
    p1: FloatArray
    x2: FloatArray
    p2: FloatArray

    def take(self, index: int | slice | IndexArray | BoolArray) -> Gaussians:
        """The times ``index`` of a batch."""
        return Gaussians(*(a[index] for a in self))

    @property
    def relative_position(self) -> FloatArray:
        """Mean relative positions ``(m, 3)`` [m]."""
        result: FloatArray = self.x2[:, :3] - self.x1[:, :3]
        return result

    @property
    def relative_velocity(self) -> FloatArray:
        """Mean relative velocities ``(m, 3)`` [m/s]."""
        result: FloatArray = self.x2[:, 3:] - self.x1[:, 3:]
        return result

    @property
    def position_covariance(self) -> FloatArray:
        """Combined position covariances ``(m, 3, 3)`` [m^2]."""
        result: FloatArray = self.p1[:, :3, :3] + self.p2[:, :3, :3]
        return result


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


def _floored_inverse(matrix: FloatArray, floor: float) -> FloatArray:
    """Inverse of symmetric matrices ``(..., n, n)`` with eigenvalues raised to
    ``floor``.

    Built from the eigen-decomposition, so a tiny floored eigenvalue next to
    a huge one is not lost to rounding as it would be by inverting the
    reconstructed matrix.
    """
    values, vectors = np.linalg.eigh(matrix)
    return from_eigen(1.0 / np.maximum(values, floor), vectors)


def _apply(matrix: FloatArray, vector: FloatArray) -> FloatArray:
    """Batched matrix-vector products ``(m, i, j) x (m, j) -> (m, i)``."""
    result: FloatArray = (matrix @ vector[..., None])[..., 0]
    return result


def _iterate_overlap(
    primary: OrbitUncertainty,
    secondary: OrbitUncertainty,
    times: FloatArray,
    floor: float,
    mean_centred: Gaussians,
) -> tuple[Gaussians, BoolArray]:
    """The peak-overlap iteration on a batch of times; may raise."""
    result = Gaussians(*(a.copy() for a in mean_centred))
    converged = np.zeros(times.size, dtype=bool)
    rows = np.arange(times.size)
    current = mean_centred
    previous: FloatArray | None = None
    for _ in range(OVERLAP_MAX_ITERATIONS):
        x1, p1, x2, p2 = current
        inverse1 = _floored_inverse(p1[:, :3, :3], floor)
        inverse2 = _floored_inverse(p2[:, :3, :3], floor)
        precision = inverse1 + inverse2
        rhs = _apply(inverse1, x1[:, :3]) + _apply(inverse2, x2[:, :3])
        peak = np.linalg.solve(precision, rhs[..., None])[..., 0]
        if previous is not None:
            step = peak - previous
            distance = np.einsum("mi,mi->m", _apply(precision, step), step)
            done = distance <= OVERLAP_TOLERANCE
            for target, source in zip(result, current, strict=True):
                target[rows[done]] = source[done]
            converged[rows[done]] = True
            keep = ~done
            rows, peak, current = rows[keep], peak[keep], current.take(keep)
            inverse1, inverse2 = inverse1[keep], inverse2[keep]
            if rows.size == 0:
                break
        previous = peak
        at = times[rows]
        centre1 = primary.centre_at(
            _conditioned_state(current.x1, current.p1, inverse1, peak), at
        )
        centre2 = secondary.centre_at(
            _conditioned_state(current.x2, current.p2, inverse2, peak), at
        )
        current = Gaussians(
            *primary.expansions(centre1, at), *secondary.expansions(centre2, at)
        )
    return result, converged


def _conditioned_state(
    x: FloatArray, p: FloatArray, inverse: FloatArray, peak: FloatArray
) -> FloatArray:
    """States ``(m, 6)`` at the peak position, velocity conditioned on it.

    ``[peak, v + P_vr A^-1 (peak - r)]`` with the object's mean ``(r, v)``,
    covariance ``P`` and floored inverse position covariance ``A^-1``.
    """
    velocity = x[:, 3:] + _apply(p[:, 3:, :3], _apply(inverse, peak - x[:, :3]))
    return np.concatenate([peak, velocity], axis=1)


def _peak_overlap(
    primary: OrbitUncertainty,
    secondary: OrbitUncertainty,
    times: FloatArray,
    floor: float,
    mean_centred: Gaussians,
) -> tuple[Gaussians, BoolArray]:
    """Expansion-centre Gaussians of both objects at ``times`` (Hall 2021,
    sec. VII.B), and where the iteration converged to finite values.

    ``floor`` is the smallest position variance used when inverting;
    ``mean_centred`` (the batched expansion about the mean elements) is the
    first iterate and is returned where the iteration failed. If one time
    makes the batch fail (a diverging iterate can leave the space of elliptic
    orbits), the times are iterated one by one so the others still converge.
    """
    try:
        gaussians, converged = _iterate_overlap(
            primary, secondary, times, floor, mean_centred
        )
    except _ITERATION_ERRORS:
        rows = [
            _single_overlap(primary, secondary, times, floor, mean_centred, k)
            for k in range(times.size)
        ]
        gaussians = Gaussians(
            *(np.stack(parts) for parts in zip(*(row[0] for row in rows), strict=True))
        )
        converged = np.array([row[1] for row in rows], dtype=bool)
    finite = np.all(
        [np.isfinite(a).reshape(times.size, -1).all(axis=1) for a in gaussians], axis=0
    )
    return gaussians, converged & finite


def _single_overlap(
    primary: OrbitUncertainty,
    secondary: OrbitUncertainty,
    times: FloatArray,
    floor: float,
    mean_centred: Gaussians,
    k: int,
) -> tuple[Gaussians, bool]:
    one = mean_centred.take(slice(k, k + 1))
    try:
        gaussians, converged = _iterate_overlap(
            primary, secondary, times[k : k + 1], floor, one
        )
    except _ITERATION_ERRORS:
        return one.take(0), False
    return gaussians.take(0), bool(converged[0])


class CollisionRate:
    """Hall's 3D-Nc collision rate ``Ncdot(t)`` [1/s] of a conjunction.

    Called with an array of times [s from TCA], it returns the rate at each.

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
        self.floor_variance = (SIGMA_FLOOR * radius) ** 2
        """Smallest position variance [m^2] used when inverting a covariance:
        :data:`~keepout.pc._plane.SIGMA_FLOOR` of the radius, squared (as in
        2-D Pc and CARA's ``Fclip``), so a perfectly known object has a
        finite, tiny extent."""
        self.evaluations = 0
        self.smallest_floored_variance = math.inf
        """Smallest combined position variance [m^2] raised to
        :attr:`floor_variance` by :meth:`_rate` (infinite if the floor never
        engaged)."""
        self.fallbacks: list[tuple[float, float]] = []
        """Times [s] and rates where the peak-overlap iteration failed."""

    @property
    def position_covariance(self) -> FloatArray:
        """Combined Cartesian position covariance at TCA [m^2]."""
        result: FloatArray = (
            self.primary.position_covariance + self.secondary.position_covariance
        )
        return result

    def _mean_centred(self, times: FloatArray) -> Gaussians:
        """Both objects' expansions about their mean orbits at ``times``."""
        return Gaussians(
            *self.primary.mean_expansions(times), *self.secondary.mean_expansions(times)
        )

    def _gap(self, gaussians: Gaussians) -> FloatArray:
        """Distance [m] from the hard-body sphere to the mean relative positions."""
        distance = np.linalg.norm(gaussians.relative_position, axis=1)
        gap: FloatArray = np.maximum(distance - self.radius, 0.0)
        return gap

    def _negligible(self, gaussians: Gaussians) -> BoolArray:
        largest = np.linalg.eigvalsh(gaussians.position_covariance)[:, -1]
        gap = self._gap(gaussians)
        negligible: BoolArray = gap * gap > NEGLIGIBLE_DISTANCE * largest
        return negligible

    def __call__(self, t: FloatArray) -> FloatArray:
        times = np.atleast_1d(np.asarray(t, dtype=np.float64))
        self.evaluations += times.size
        gaussians = self._mean_centred(times)
        values = np.zeros(times.size)
        active = np.flatnonzero(~self._negligible(gaussians))
        if active.size == 0:
            return values
        mean_centred = gaussians.take(active)
        converged, ok = _peak_overlap(
            self.primary,
            self.secondary,
            times[active],
            self.floor_variance,
            mean_centred,
        )
        for k, index in enumerate(active):
            t_k = float(times[index])
            source = converged if ok[k] else mean_centred
            values[index] = self._checked_rate(t_k, source.take(k))
            if not ok[k]:
                self.fallbacks.append((t_k, float(values[index])))
        return values

    def _checked_rate(self, t: float, gaussians: Gaussians) -> float:
        """:meth:`_rate`, with a failure raised as :class:`PcConvergenceError`."""
        try:
            value = self._rate(*gaussians)
        except _RATE_ERRORS as error:
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

    def _screen_cells(self, times: FloatArray) -> ScreenCells:
        """Gap, relative speed, largest variance, crossing time and time width.

        All at ``times``, from the mean-centred expansion that
        :meth:`_negligible` uses. The time width is the standard deviation
        ``1 / (|v| sqrt(u' A^-1 u))`` of the straight-line passage through
        the density, with ``A`` floored as in :meth:`_rate`.
        """
        gaussians = self._mean_centred(times)
        dv = gaussians.relative_velocity
        a = gaussians.position_covariance
        gap = self._gap(gaussians)
        speed = np.linalg.norm(dv, axis=1)
        values, vectors = np.linalg.eigh(a)
        largest = values[:, -1]
        direction = dv / np.where(speed > 0.0, speed, 1.0)[:, None]
        # u' A u and u' A^-1 u (A floored) in the principal axes of A.
        along2 = np.einsum("ij,ijk->ik", direction, vectors) ** 2
        sigma = np.sqrt(np.maximum(np.einsum("ik,ik->i", along2, values), 0.0))
        precision = np.einsum(
            "ik,ik->i", along2, 1.0 / np.maximum(values, self.floor_variance)
        )
        with np.errstate(divide="ignore"):
            crossing = np.where(speed > 0.0, (sigma + self.radius) / speed, math.inf)
            width = np.where(speed > 0.0, 1.0 / (speed * np.sqrt(precision)), math.inf)
        return ScreenCells(gap, speed, largest, crossing, width)

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
        count = self._cell_count(window)
        cell = (end - start) / count
        centres = start + (np.arange(count, dtype=np.float64) + 0.5) * cell
        parts = [
            self._screen_cells(centres[i : i + SCREEN_CHUNK])
            for i in range(0, count, SCREEN_CHUNK)
        ]
        cells = ScreenCells(*(np.concatenate(p) for p in zip(*parts, strict=True)))
        half = 0.5 * cell
        acceleration = self.primary.max_acceleration + self.secondary.max_acceleration
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

    def _cell_count(self, window: tuple[float, float]) -> int:
        """Number of screen cells in ``window`` (see :meth:`screen`)."""
        start, end = window
        span = end - start
        centre = self._screen_cells(np.array([0.5 * (start + end)]))
        speed = float(centre.speed[0])
        extent = reach(float(centre.largest[0]), self.radius)
        natural = extent / speed if speed > 0.0 else math.inf
        longest = SCREEN_PERIOD_FRACTION * min(
            self.primary.period, self.secondary.period
        )
        cell = max(
            span / SCREEN_MAX_CELLS, min(natural, span / SCREEN_MIN_CELLS, longest)
        )
        return math.ceil(span / cell)

    def _rate(
        self, x1: FloatArray, p1: FloatArray, x2: FloatArray, p2: FloatArray
    ) -> float:
        mean, covariance = x2 - x1, p1 + p2
        r_mean, v_mean = mean[:3], mean[3:]
        values, vectors = np.linalg.eigh(covariance[:3, :3])
        if values[0] < self.floor_variance:
            self.smallest_floored_variance = min(
                self.smallest_floored_variance, float(values[0])
            )
        # The floored position covariance A, used through its eigen-decomposition.
        floored = np.maximum(values, self.floor_variance)
        a_inverse = from_eigen(1.0 / floored, vectors)
        log_det = float(np.sum(np.log(floored)))
        b, c = covariance[3:, :3], covariance[3:, 3:]
        gain = b @ a_inverse
        conditional = c - gain @ b.T
        nodes, weights = sphere_rule_principal(floored, vectors, r_mean, self.radius)
        offset = self.radius * nodes - r_mean
        mahalanobis = np.einsum("ij,ij->i", offset @ a_inverse, offset)
        density = np.exp(-0.5 * mahalanobis - 0.5 * log_det) / (2.0 * math.pi) ** 1.5
        variance = np.einsum("ij,ij->i", nodes @ conditional, nodes)
        sigma = np.sqrt(np.maximum(variance, _TINY_VARIANCE))
        x = np.einsum("ij,ij->i", nodes, v_mean + offset @ gain.T) / (SQRT_2 * sigma)
        flux = sigma / SQRT_2PI * (np.exp(-x * x) - _SQRT_PI * x * special.erfc(x))
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
        rate.position_covariance,
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


def check_fallbacks(fallbacks: Sequence[tuple[float, float]], found: RateScan) -> None:
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
