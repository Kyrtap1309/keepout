"""Independent reference Pc for isotropic covariance (Rice distribution)."""

import math

from scipy import integrate, special


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
    mu: float, mv: float, su: float, sv: float, radius: float
) -> float:
    """Pc for a plane Gaussian given in its principal axes, by brute force.

    Integrates ``N(u; mu, su) * P(|v| < h(u))`` over ``u`` directly, with
    breakpoints where the chord mass steps (``h(u) = |mv|``), using
    ``scipy.stats.norm`` rather than keepout's erf/erfc helpers.
    """
    from scipy import stats

    def integrand(u: float) -> float:
        half_chord = math.sqrt(max(radius * radius - u * u, 0.0))
        mass = stats.norm.cdf(half_chord, mv, sv) - stats.norm.cdf(-half_chord, mv, sv)
        return float(stats.norm.pdf(u, mu, su) * mass)

    steps = [
        edge * math.sqrt(radius * radius - (abs(mv) + k * sv) ** 2)
        for k in (-10.0, 0.0, 10.0)
        for edge in (-1.0, 1.0)
        if 0.0 < abs(mv) + k * sv < radius
    ]
    value, _ = integrate.quad(
        integrand,
        -radius,
        radius,
        points=sorted(steps),
        epsabs=0.0,
        epsrel=1e-12,
        limit=500,
    )
    return float(value)
