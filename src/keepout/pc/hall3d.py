"""3-D Pc by Hall's 3D-Nc method (Hall 2021, J. Spacecraft Rockets 58(3)).

The expected number of collisions ``Nc`` is the time integral of the rate at
which the secondary's centre enters the hard-body sphere around the primary
(eqs. 51e, 52):

    Ncdot(t) = R^2 * integral over the unit sphere of
               N(R r; r_rel, A) * nu(r),
    nu(r) = sigma / sqrt(2 pi) * [exp(-x^2) - sqrt(pi) x erfc(x)],

with the relative position mean ``r_rel`` and covariance ``A`` and the
conditional relative-velocity statistics (``sigma``, ``x``) taken from a
first-order expansion of two-body motion in equinoctial elements. The
expansion is centred not on the mean orbits but on the states at the peak
overlap of the two position densities, found by iteration at each time
(sec. VII.B).

``Nc`` counts every entry into the hard-body sphere, so it is an upper bound
on Pc, the probability of at least one entry. For a single, short encounter
(a straight-line pass, each sample entering at most once) the two agree. For
several encounters in the window, or a long or zero-relative-velocity
encounter in which a sample can leave and re-enter, ``Nc`` exceeds Pc; a
diagnostic says so.
"""

from __future__ import annotations

import math
from typing import TypeAlias

import numpy as np
from scipy import special

from keepout._types import FloatArray, require_positive
from keepout.conjunction import Conjunction
from keepout.errors import GeometryError, PcConvergenceError
from keepout.pc._orbit import OrbitUncertainty
from keepout.pc._plane import require_known_covariance
from keepout.pc._sphere import sphere_rule
from keepout.pc._timeline import (
    QUAD_RTOL,
    RATE_FLOOR,
    Candidate,
    RateIntegral,
    RateScan,
    integrate_segments,
    locate,
    locate_auto,
    regions,
    straight_line_half_window,
)
from keepout.pc.result import PcResult

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

QUIET_AGREEMENT = 1e-4
"""Time-integral agreement below which a missed ``rtol`` is not reported. It
is per piece (the last two Gauss-Legendre orders of one piece of an
encounter segment, relative to the total Nc) and 10 times inside Hall's
tightest validation criterion (1e-3 relative to CARA), so it cannot affect
a result at that accuracy."""

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

Gaussians: TypeAlias = tuple[FloatArray, FloatArray, FloatArray, FloatArray]
"""Mean and covariance of both objects' states at one time: ``x1, p1, x2, p2``."""


def _floored(matrix: FloatArray, floor: float) -> FloatArray:
    """A symmetric matrix with its eigenvalues raised to at least ``floor``."""
    values, vectors = np.linalg.eigh(matrix)
    result: FloatArray = (vectors * np.maximum(values, floor)) @ vectors.T
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
        inverse1, inverse2 = (
            np.linalg.inv(_floored(p[:3, :3], floor)) for p in (p1, p2)
        )
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
        reach = _reach(float(largest[0]), self.radius)
        natural = reach / float(speed[0]) if speed[0] > 0.0 else math.inf
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
        a = (vectors * np.maximum(values, self.floor)) @ vectors.T
        b, c = covariance[3:, :3], covariance[3:, 3:]
        a_inverse = np.linalg.inv(a)
        gain = b @ a_inverse
        conditional = c - gain @ b.T
        nodes, weights = sphere_rule(a, r_mean, self.radius)
        offset = self.radius * nodes - r_mean
        mahalanobis = np.einsum("ij,jk,ik->i", offset, a_inverse, offset)
        _, log_det = np.linalg.slogdet(a)
        density = np.exp(-0.5 * mahalanobis - 0.5 * log_det) / (2.0 * math.pi) ** 1.5
        sigma = np.sqrt(
            np.maximum(np.einsum("ij,jk,ik->i", nodes, conditional, nodes), 1e-300)
        )
        x = np.einsum("ij,ij->i", nodes, v_mean + offset @ gain.T) / (_SQRT_2 * sigma)
        flux = sigma / _SQRT_2PI * (np.exp(-x * x) - _SQRT_PI * x * special.erfc(x))
        return self.radius**2 * float(np.sum(weights * density * flux))


def _reach(largest_variance: float, radius: float) -> float:
    """Distance [m] beyond which :meth:`CollisionRate._negligible` holds."""
    return math.sqrt(NEGLIGIBLE_DISTANCE * largest_variance) + radius


def collision_rate(conjunction: Conjunction) -> CollisionRate:
    """The 3D-Nc collision rate of a conjunction with both covariances known."""
    require_known_covariance(conjunction.unknown_covariance, "hall")
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
    everywhere scanned, until an encounter appears), by at most half the
    shorter orbital period. ``t*`` is clipped to that half period.

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
        _relative_velocity(conjunction),
        rate.primary.position_covariance + rate.secondary.position_covariance,
        conjunction.hard_body_radius,
    )
    limit = 0.5 * min(rate.primary.period, rate.secondary.period)
    closest = min(max(straight_line_closest_time(conjunction), -limit), limit)
    centre = (min(0.0, closest), max(0.0, closest))
    return locate_auto(rate, half, limit, rate.screen, centre)


def _relative_velocity(conjunction: Conjunction) -> FloatArray:
    velocity: FloatArray = conjunction.secondary.state.v - conjunction.primary.state.v
    return velocity


def straight_line_closest_time(conjunction: Conjunction) -> float:
    """Time [s from TCA] of the straight-line closest approach of the states.

    ``-(r . v) / |v|^2`` with the relative position ``r`` and velocity ``v``;
    0 at zero relative velocity.
    """
    velocity = _relative_velocity(conjunction)
    speed2 = float(velocity @ velocity)
    if speed2 == 0.0:
        return 0.0
    position = conjunction.secondary.state.r - conjunction.primary.state.r
    return -float(position @ velocity) / speed2


def _diagnostics(
    integral: RateIntegral, window_given: bool, rtol: float = QUAD_RTOL
) -> list[str]:
    notes: list[str] = []
    nc = integral.nc
    if integral.peak <= 0.0:
        start, end = integral.window
        notes.append(
            f"collision rate negligible over the whole window [{start:.6g}, "
            f"{end:.6g}] s"
        )
    if nc > 1.0:
        notes.append(
            f"expected number of collisions Nc = {nc:.4g} exceeds 1; Pc set to 1"
        )
    if len(integral.segments) > 1:
        largest = max(s.nc for s in integral.segments)
        notes.append(
            f"{len(integral.segments)} encounters in the window: Pc lies between "
            f"the largest one's Nc ({largest:.3g}) and the total Nc ({nc:.3g}); "
            "the value is Nc (Hall 2021)"
        )
    if not window_given and integral.edge >= RATE_FLOOR:
        notes.append(
            f"the collision rate is {integral.edge:.1e} of its peak at the window "
            "edge (half an orbital period): the encounter may be longer; pass "
            "window="
        )
    if integral.agreement > max(rtol, QUIET_AGREEMENT):
        notes.append(
            f"the time integral did not reach rtol: its worst piece agrees to "
            f"{integral.agreement:.1e} of Nc (rtol {rtol:.1e})"
        )
    return notes


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


def hall3d(
    conjunction: Conjunction,
    *,
    window: tuple[float, float] | None = None,
    rtol: float = QUAD_RTOL,
) -> PcResult:
    """3-D Pc of a conjunction by Hall's 3D-Nc method.

    Parameters
    ----------
    conjunction
        The conjunction; both covariances must be known (6x6, including
        velocity) and both orbits elliptic.
    window
        Integration window ``(start, end)`` [s from TCA]. ``None`` (default)
        grows a symmetric window until the rate at its edges is negligible,
        up to half the shorter orbital period.
    rtol
        Relative tolerance of the time integral.

    Returns
    -------
    PcResult
        ``value = min(Nc, 1)``. ``details``: ``"nc"``, ``"window_start_s"``,
        ``"window_end_s"``, ``"encounters"`` (segments where the rate is not
        negligible), ``"largest_encounter_nc"`` (a lower bound on Pc),
        ``"evaluations"`` (rate evaluations) and ``"fallbacks"`` (the number
        of times the peak-overlap iteration failed in the far tail, outside
        every encounter).

    Raises
    ------
    CovarianceError
        If a covariance is unknown.
    GeometryError
        If an orbit is not elliptic.
    PcConvergenceError
        If the peak-overlap iteration fails where the rate matters, the rate
        cannot be computed, or the encounters in ``window`` need more scan
        points than allowed.
    ValueError
        If ``window`` is not finite with ``start < end``, or ``rtol`` is not
        finite and positive.
    """
    require_positive(rtol, "rtol")
    rate = collision_rate(conjunction)
    found = encounter_scan(conjunction, rate, window)
    integral = integrate_segments(rate, found, rtol=rtol)
    check_fallbacks(rate.fallbacks, found)
    if not math.isfinite(integral.nc):
        raise PcConvergenceError(f"Hall 3D-Nc: Nc is not finite ({integral.nc})")
    details = {
        "nc": integral.nc,
        "window_start_s": integral.window[0],
        "window_end_s": integral.window[1],
        "encounters": float(len(integral.segments)),
        "largest_encounter_nc": max((s.nc for s in integral.segments), default=0.0),
        "evaluations": float(rate.evaluations),
        "fallbacks": float(len(rate.fallbacks)),
    }
    notes = [
        *_floor_notes(rate),
        *_diagnostics(integral, window is not None, rtol),
        *_entry_notes(conjunction, rate, integral, window is not None),
    ]
    return PcResult.clamped(min(integral.nc, 1.0), "hall", tuple(notes), details)


def _floor_notes(rate: CollisionRate) -> list[str]:
    """Where :data:`VARIANCE_FLOOR` raised the combined covariance.

    As in the 2-D methods, an object whose position covariance is zero is
    perfectly known (only an all-zero 6x6 CDM covariance means unknown), and
    only the floor on the combined covariance is reported.
    """
    if rate.floored < rate.floor:
        return [
            f"combined position covariance: smallest variance {rate.floored:.3g} "
            f"m^2 raised to {rate.floor:.3g} m^2"
        ]
    return []


def _entry_notes(
    conjunction: Conjunction,
    rate: CollisionRate,
    integral: RateIntegral,
    window_given: bool,
) -> list[str]:
    """Notes on what ``Nc`` counts and on an encounter the scan did not find."""
    # Deferred: keepout.pc.api imports this module.
    from keepout.pc.api import DURATION_LIMIT, encounter_duration

    notes: list[str] = []
    repeated = (
        "Nc counts repeated entries into the hard-body sphere, so it is an "
        "upper bound on Pc"
    )
    if conjunction.relative_speed == 0.0:
        notes.append(f"relative velocity is zero: {repeated}")
    else:
        duration = encounter_duration(conjunction)
        period = rate.primary.period
        if duration > DURATION_LIMIT * period:
            notes.append(
                f"long encounter (about {duration:.3g} s, "
                f"{100.0 * duration / period:.2g}% of the primary's orbital "
                f"period): {repeated}"
            )
        if integral.peak <= 0.0 and not window_given:
            covariance = (
                rate.primary.position_covariance + rate.secondary.position_covariance
            )
            reach = _reach(
                float(np.linalg.eigvalsh(covariance)[-1]), conjunction.hard_body_radius
            )
            miss = conjunction.plane.miss_distance
            if miss <= reach:
                notes.append(
                    f"the straight-line closest approach ({miss:.3g} m, "
                    f"t = {straight_line_closest_time(conjunction):.6g} s) is "
                    f"within {reach:.3g} m, but two-body motion keeps the "
                    "objects apart over the whole window: check that the "
                    "states are near TCA"
                )
    return notes
