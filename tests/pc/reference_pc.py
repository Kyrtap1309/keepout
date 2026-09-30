"""Independent reference Pc for isotropic covariance (Rice distribution)."""

import math
import warnings

from scipy import integrate, special, stats


def isotropic_pc(miss: float, sigma: float, radius: float) -> float:
    """Pc for plane covariance ``sigma^2 I``: the radial (Rice) integral.

    ``integral_0^R (r / s^2) exp(-(r^2 + m^2) / (2 s^2)) I0(r m / s^2) dr``,
    written with ``i0e`` so it neither overflows nor underflows early.
    """

    def integrand(r: float) -> float:
        scaled = r * miss / sigma**2
        exponent = -((r - miss) ** 2) / (2.0 * sigma**2)
        return float(r / sigma**2 * math.exp(exponent) * special.i0e(scaled))

    value, _ = integrate.quad(integrand, 0.0, radius, epsabs=0.0, epsrel=1e-12)
    return float(value)


def principal_axes_pc(
    miss_u: float, miss_v: float, sigma_u: float, sigma_v: float, radius: float
) -> float:
    """Pc for a plane Gaussian given in its principal axes, by brute force.

    Integrates ``N(v; miss_v, sigma_v) * P(|u - miss_u| < h(v))`` over the **narrow**
    axis ``v`` (keepout integrates over the wide axis ``u``), with ``v = R sin(phi)`` so
    the chord ``h = R cos(phi)`` has no square-root end point, breakpoints across the
    narrow density, and ``scipy.stats.norm`` rather than keepout's erf/erfc helpers.
    """

    def integrand(phi: float) -> float:
        v, half_chord = radius * math.sin(phi), radius * math.cos(phi)
        mass = stats.norm.cdf(half_chord, miss_u, sigma_u) - stats.norm.cdf(
            -half_chord, miss_u, sigma_u
        )
        return float(stats.norm.pdf(v, miss_v, sigma_v) * mass * half_chord)

    steps = [
        math.asin((miss_v + k * sigma_v) / radius)
        for k in range(-12, 13, 2)
        if -radius < miss_v + k * sigma_v < radius
    ]
    # Round-off can stop QUADPACK short of epsrel for tiny Pc; the test's own
    # comparison (rtol 1e-7) is the judge, so its warning is not shown.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", integrate.IntegrationWarning)
        value, _ = integrate.quad(
            integrand,
            -0.5 * math.pi,
            0.5 * math.pi,
            points=sorted(steps) or None,
            epsabs=0.0,
            epsrel=1e-10,
            limit=500,
        )
    return float(value)
