"""2-D Pc by adaptive numerical integration (Foster & Estes 1992).

The 2-D integral over the hard-body circle is reduced exactly to a 1-D
integral along the wide principal axis ``u``:

    Pc = integral over u of  N(u; mu, su) * P(|v - mv| within chord at u) du

With ``u = R sin(theta)`` the integrand is smooth at the circle's edge, and
only the band where the density is not negligible is integrated.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import integrate

from keepout.encounter import EncounterPlane
from keepout.pc._plane import PlaneGaussian, plane_gaussian, require_positive
from keepout.pc.result import PcResult

FOSTER_RTOL = 1e-10
"""Default relative tolerance of the adaptive quadrature."""

STEP_SIGMAS = 10.0
"""Half-width, in narrow-axis standard deviations, of the chord-mass step
around ``h = |mv|`` whose ends are given to the quadrature as breakpoints."""


def _chord_steps(gaussian: PlaneGaussian, low: float, high: float) -> list[float]:
    """Angles in ``(low, high)`` where the chord mass rises from 0 to 1.

    With a narrow axis much thinner than the circle, ``M(h)`` is a step at
    ``h = |mv|``, i.e. ``theta = +-acos(|mv| / R)``, which adaptive
    quadrature can step over; the step's centre and ends become breakpoints.
    """
    radius = gaussian.radius
    points: set[float] = set()
    for k in (-STEP_SIGMAS, 0.0, STEP_SIGMAS):
        half_chord = abs(gaussian.mv) + k * gaussian.sv
        if 0.0 < half_chord < radius:
            angle = math.acos(half_chord / radius)
            points.update(p for p in (-angle, angle) if low < p < high)
    return sorted(points)


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
        If ``hard_body_radius`` is not finite and positive, or ``rtol`` is not
        positive.
    CovarianceError
        If the plane covariance has no positive eigenvalue.
    """
    require_positive(hard_body_radius, "hard_body_radius")
    require_positive(rtol, "rtol")
    gaussian = plane_gaussian(plane.miss_vector, plane.covariance, hard_body_radius)
    diagnostics = list(gaussian.diagnostics)
    band = gaussian.band()
    if band is None:
        return PcResult(0.0, "foster", tuple(diagnostics), {"error_estimate": 0.0})
    radius = hard_body_radius
    low = math.asin(band[0] / radius)
    high = math.asin(band[1] / radius)

    def integrand(theta: float) -> float:
        # u = R sin(theta), du = R cos(theta) dtheta, half-chord = R cos(theta).
        half_chord = np.array([radius * math.cos(theta)])
        density = gaussian.density_u(np.array([radius * math.sin(theta)]))
        return float((half_chord * density * gaussian.chord_mass(half_chord))[0])

    output = integrate.quad(
        integrand,
        low,
        high,
        points=_chord_steps(gaussian, low, high) or None,
        epsabs=0.0,
        epsrel=rtol,
        limit=200,
        full_output=1,
    )
    value, error, info = output[0], output[1], output[2]
    if len(output) > 3:
        diagnostics.append(f"Foster quadrature: {str(output[3]).splitlines()[0]}")
    return PcResult(
        min(max(value, 0.0), 1.0),
        "foster",
        tuple(diagnostics),
        {"error_estimate": float(error), "evaluations": float(info["neval"])},
    )
