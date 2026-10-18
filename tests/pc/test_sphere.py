import math

import numpy as np
import pytest
from builders import approx
from scipy.integrate import lebedev_rule, quad
from scipy.stats import multivariate_normal

from keepout._types import FloatArray
from keepout.pc._sphere import sphere_grids, sphere_rule_principal


def sphere_rule(
    covariance: FloatArray, mean: FloatArray, radius: float
) -> tuple[FloatArray, FloatArray]:
    """:func:`sphere_rule_principal` of a covariance, decomposed as
    :class:`~keepout.pc._rate.CollisionRate` does it."""
    values, axes = np.linalg.eigh(covariance)
    return sphere_rule_principal(values, axes, mean, radius)


def test_weights_cover_the_sphere() -> None:
    nodes, weights = sphere_rule(np.diag([1.0, 4.0, 9.0]), np.zeros(3), 1.0)
    np.testing.assert_allclose(np.linalg.norm(nodes, axis=1), 1.0, atol=1e-14)
    assert weights.sum() == pytest.approx(4.0 * math.pi, rel=1e-12)


def surface_mass(
    nodes: FloatArray,
    weights: FloatArray,
    covariance: FloatArray,
    mean: FloatArray,
    radius: float,
) -> float:
    density = multivariate_normal(mean, covariance).pdf(radius * nodes)
    return radius**2 * float(np.sum(weights * density))


def isotropic_surface_mass(sigma: float, distance: float, radius: float) -> float:
    """Exact mass of ``N(x; m, sigma^2 I)`` on a sphere, ``|m| = distance``.

    With ``u = cos(angle to m)`` the integrand is ``exp(radius distance u /
    sigma^2)``, which integrates in closed form; written to avoid overflow.
    """
    k = radius * distance / sigma**2
    peak = math.exp(-((radius - distance) ** 2) / (2.0 * sigma**2))
    angular = peak * (1.0 - math.exp(-2.0 * k)) / k
    return 2.0 * math.pi * radius**2 * angular / math.pow(2.0 * math.pi * sigma**2, 1.5)


def normal_pdf(x: float, mean: float, sigma: float) -> float:
    z = (x - mean) / sigma
    return math.exp(-0.5 * z * z) / (math.sqrt(2.0 * math.pi) * sigma)


def principal_surface_mass(
    sigmas: tuple[float, ...], mean: tuple[float, ...], radius: float
) -> float:
    """Mass of ``N(x; mean, diag(sigmas^2))`` on a sphere, by adaptive quadrature.

    Independent of the rule: in the principal frame, with ``u = x1`` and
    ``x2 + i x3 = rho e^(i phi)``, ``rho = sqrt(radius^2 - u^2)``, the surface
    element is ``radius du dphi`` (Archimedes), so the mass is
    ``radius * integral of N(u; m1, s1) * ring(u) du`` with ``ring(u)`` the
    integral over ``phi`` of the other two factors. QUADPACK integrates both,
    with breakpoints at the band centre ``u = m1`` and at the peaks
    ``rho cos(phi) = m2`` of each ring; ``u`` beyond 12 ``s1`` of ``m1`` is
    left out (the density there is below ``exp(-72)`` of its peak).
    """
    s1, s2, s3 = sigmas
    m1, m2, m3 = mean

    def ring(u: float) -> float:
        rho = math.sqrt(max(radius**2 - u**2, 0.0))

        def density(phi: float) -> float:
            x2, x3 = rho * math.cos(phi), rho * math.sin(phi)
            return normal_pdf(x2, m2, s2) * normal_pdf(x3, m3, s3)

        peak = math.acos(m2 / rho) if abs(m2) < rho else None
        points = None if peak is None else (-peak, peak)
        mass, _ = quad(
            density, -math.pi, math.pi, points=points, epsabs=0.0, epsrel=1e-11
        )
        return normal_pdf(u, m1, s1) * mass

    low, high = max(-radius, m1 - 12.0 * s1), min(radius, m1 + 12.0 * s1)
    mass, _ = quad(ring, low, high, points=(m1,), epsabs=0.0, epsrel=1e-11)
    return radius * mass


@pytest.mark.parametrize(
    ("sigmas", "mean", "radius"),
    [
        ((50.0, 80.0, 200.0), (30.0, -10.0, 5.0), 10.0),  # smooth over the sphere
        ((0.1, 0.3, 125.0), (5.0, 3.0, 40.0), 10.0),  # thin band: sigma << R
        ((0.05, 2.0, 50.0), (-3.0, 1.0, 8.0), 10.0),  # thin band, wide in phi
        ((0.02, 0.5, 30.0), (9.9, 0.3, -2.0), 10.0),  # thin band round the pole
        ((0.01, 0.01, 0.05), (6.0, 8.0, 0.0), 10.0),  # small elongated blob
        ((0.2, 0.2, 0.2), (0.0, 0.0, 10.0), 10.0),  # small blob on the sphere
    ],
)
def test_density_integral_matches_a_reference(
    sigmas: tuple[float, ...], mean: tuple[float, ...], radius: float
) -> None:
    rotation = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))[0]
    covariance = rotation @ np.diag(np.square(sigmas)) @ rotation.T
    centre = rotation @ np.array(mean)
    nodes, weights = sphere_rule(covariance, centre, radius)
    assert weights.sum() == pytest.approx(4.0 * math.pi, rel=1e-12)
    got = surface_mass(nodes, weights, covariance, centre, radius)
    if min(sigmas) > radius:
        points, lebedev = lebedev_rule(131)
        expected = surface_mass(points.T, lebedev, covariance, centre, radius)
    elif len(set(sigmas)) == 1:
        expected = isotropic_surface_mass(
            sigmas[0], float(np.linalg.norm(centre)), radius
        )
    else:
        # No closed form for a thin band: nested adaptive quadrature in the
        # principal frame, which shares nothing with the rule.
        expected = principal_surface_mass(sigmas, mean, radius)
    assert got == approx(expected, rel=1e-9)


def test_batched_rules_are_the_rules_of_each_gaussian() -> None:
    # Smooth, thin band, band clipped at a pole (fewer panels), wide in phi.
    values = np.array(
        [
            [2500.0, 6400.0, 4e4],
            [0.01, 0.09, 1.5e4],
            [4e-4, 0.25, 900.0],
            [1.0, 4.0, 9.0],
        ]
    )
    along = np.array([30.0, 5.0, 9.9999, 0.0])
    batched = sphere_grids(values, along, 10.0)
    assert len({grid.cos_theta.size for grid in batched}) > 1
    for k, grid in enumerate(batched):
        (single,) = sphere_grids(values[k : k + 1], along[k : k + 1], 10.0)
        for got, expected in zip(grid, single, strict=True):
            np.testing.assert_array_equal(got, expected)
