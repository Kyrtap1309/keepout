"""2-D Pc by adaptive numerical integration (Foster & Estes 1992).

The 2-D integral over the hard-body circle is reduced exactly to a 1-D
integral along the wide principal axis ``u``:

    Pc = integral over u of N(u; miss_u, sigma_u)
                            * P(|v - miss_v| within chord at u) du

With ``u = R sin(theta)`` (``R`` the hard-body radius) the integrand is
smooth at the circle's edge, and
only the band where the density is not negligible is integrated.
"""

from __future__ import annotations

import math
import sys
from typing import Final

from scipy import integrate

from keepout.encounter import EncounterPlane
from keepout.pc._plane import PlaneGaussian, plane_gaussian_of
from keepout.pc.result import PcResult

__all__ = [
    "BREAKPOINT_GAP",
    "FOSTER_RTOL",
    "MIN_RTOL",
    "QUAD_LIMIT",
    "STEP_SIGMAS",
    "TAIL_SIGMAS",
    "foster",
    "gaussian_pc",
]

FOSTER_RTOL: Final = 1e-10
"""Default relative tolerance of the adaptive quadrature."""

MIN_RTOL: Final = 50.0 * sys.float_info.epsilon
"""Smallest relative tolerance QUADPACK accepts."""

STEP_SIGMAS: Final = 10.0
"""Half-width, in narrow-axis standard deviations, of the chord-mass step
around ``h = |miss_v|`` whose ends are given to the quadrature as breakpoints."""

BREAKPOINT_GAP: Final = 1e-6
"""Breakpoints closer than this fraction of the band to its ends are dropped."""

TAIL_SIGMAS: Final = 12.0
"""Half-width, in standard deviations, of the band along the wide axis that
is integrated; outside it the density is below exp(-72) of its peak."""

QUAD_LIMIT: Final = 200
"""Largest number of QUADPACK subintervals."""


def _band(gaussian: PlaneGaussian) -> tuple[float, float] | None:
    """Angles ``theta`` (``u = R sin(theta)``) of the part of the circle where
    the density along ``u`` matters.

    Returns ``None`` if the band misses the circle (Pc underflows to 0).
    """
    radius = gaussian.hard_body_radius
    low = max(gaussian.miss_u - TAIL_SIGMAS * gaussian.sigma_u, -radius)
    high = min(gaussian.miss_u + TAIL_SIGMAS * gaussian.sigma_u, radius)
    if low >= high:
        return None
    return math.asin(low / radius), math.asin(high / radius)


def _chord_steps(gaussian: PlaneGaussian, low: float, high: float) -> list[float]:
    """Angles in ``(low, high)`` where the chord mass rises from 0 to 1.

    With a narrow axis much thinner than the circle, ``M(h)`` is a step at
    ``h = |miss_v|``, i.e. ``theta = +-acos(|miss_v| / R)``, which adaptive
    quadrature can step over; the step's centre and ends become breakpoints.
    """
    radius = gaussian.hard_body_radius
    points: set[float] = set()
    for k in (-STEP_SIGMAS, 0.0, STEP_SIGMAS):
        half_chord = abs(gaussian.miss_v) + k * gaussian.sigma_v
        if 0.0 < half_chord < radius:
            angle = math.acos(half_chord / radius)
            points.update(p for p in (-angle, angle) if low < p < high)
    # A breakpoint a hair from an end of the band (``h = |miss_v|`` with a
    # tiny ``miss_v``, or ``h = 10 sigma_v`` with ``10 sigma_v ~ R``) leaves
    # QUADPACK a sliver interval, which it reports as bad integrand behaviour;
    # the step it marks is then at the end itself, so it is dropped.
    gap = BREAKPOINT_GAP * (high - low)
    return [p for p in sorted(points) if p - low > gap and high - p > gap]


def foster(
    plane: EncounterPlane, hard_body_radius: float, *, rtol: float = FOSTER_RTOL
) -> PcResult:
    """2-D Pc of a short-duration encounter by adaptive quadrature.

    Parameters
    ----------
    plane
        The encounter plane (miss vector and combined covariance).
    hard_body_radius
        Combined hard-body radius [m].
    rtol
        Relative tolerance passed to :func:`scipy.integrate.quad`.

    Returns
    -------
    PcResult
        ``details`` holds ``"error_estimate"`` (absolute, from QUADPACK) and
        ``"evaluations"``.

    Raises
    ------
    ValueError
        If ``hard_body_radius`` is not finite and positive, or ``rtol`` is
        below :data:`MIN_RTOL`.
    GeometryError
        If the miss vector is not finite.
    CovarianceError
        If a covariance is unknown, or the plane covariance is not finite or
        has no positive eigenvalue.
    """
    if not rtol >= MIN_RTOL:
        raise ValueError(f"rtol must be >= {MIN_RTOL:.3g}, got {rtol}")
    gaussian = plane_gaussian_of(plane, hard_body_radius, "foster")
    return gaussian_pc(gaussian, rtol=rtol)


def gaussian_pc(gaussian: PlaneGaussian, *, rtol: float = FOSTER_RTOL) -> PcResult:
    """Foster's quadrature of a principal-axes Gaussian over its hard body.

    The core of :func:`foster`, without its argument checks: for callers that
    integrate covariances other than the plane's own (see
    :mod:`keepout.pc.maxpc`).
    """
    diagnostics = list(gaussian.diagnostics)
    band = _band(gaussian)
    if band is None:
        details = {"error_estimate": 0.0, "evaluations": 0.0}
        return PcResult(0.0, "foster", tuple(diagnostics), details)
    low, high = band
    radius = gaussian.hard_body_radius

    def integrand(theta: float) -> float:
        # u = R sin(theta), du = R cos(theta) dtheta, half-chord = R cos(theta).
        half_chord = radius * math.cos(theta)
        density = gaussian.density_u_at(radius * math.sin(theta))
        return half_chord * density * gaussian.chord_mass_at(half_chord)

    value, error, info, *message = integrate.quad(
        integrand,
        low,
        high,
        points=_chord_steps(gaussian, low, high) or None,
        epsabs=0.0,
        epsrel=rtol,
        limit=QUAD_LIMIT,
        full_output=1,
    )
    if message:
        diagnostics.append(f"Foster quadrature: {' '.join(str(message[0]).split())}")
    return PcResult.clamped(
        value,
        "foster",
        tuple(diagnostics),
        {"error_estimate": float(error), "evaluations": float(info["neval"])},
    )
