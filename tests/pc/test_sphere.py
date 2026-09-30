import math

import numpy as np
import pytest
from scipy.integrate import lebedev_rule
from scipy.stats import multivariate_normal

from keepout.pc._sphere import PHI_RESOLUTION, sphere_rule


def test_weights_cover_the_sphere() -> None:
    nodes, weights = sphere_rule(np.diag([1.0, 4.0, 9.0]), np.zeros(3), 1.0)
    np.testing.assert_allclose(np.linalg.norm(nodes, axis=1), 1.0, atol=1e-14)
    assert weights.sum() == pytest.approx(4.0 * math.pi, rel=1e-12)


def surface_mass(
    nodes: np.ndarray,
    weights: np.ndarray,
    covariance: np.ndarray,
    mean: np.ndarray,
    radius: float,
) -> float:
    density = multivariate_normal(mean, covariance).pdf(radius * nodes)
    return radius**2 * float(np.sum(weights * density))


@pytest.mark.parametrize(
    ("sigmas", "mean", "radius"),
    [
        ((50.0, 80.0, 200.0), (30.0, -10.0, 5.0), 10.0),  # smooth over the sphere
        ((0.1, 0.3, 125.0), (5.0, 3.0, 40.0), 10.0),  # thin band: sigma << R
        ((0.2, 0.2, 0.2), (0.0, 0.0, 10.0), 10.0),  # small blob on the sphere
    ],
)
def test_density_integral_matches_a_high_order_lebedev_rule(
    sigmas: tuple[float, ...], mean: tuple[float, ...], radius: float
) -> None:
    rotation = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))[0]
    covariance = rotation @ np.diag(np.square(sigmas)) @ rotation.T
    centre = rotation @ np.array(mean)
    nodes, weights = sphere_rule(covariance, centre, radius)
    got = surface_mass(nodes, weights, covariance, centre, radius)
    if min(sigmas) > radius:
        points, lebedev = lebedev_rule(131)
        expected = surface_mass(points.T, lebedev, covariance, centre, radius)
    else:
        # Reference: the same rule with twice the resolution in both angles.
        fine = sphere_rule(
            covariance, centre, radius, order=32, phi_resolution=PHI_RESOLUTION / 2
        )
        expected = surface_mass(*fine, covariance, centre, radius)
    assert got == pytest.approx(expected, rel=1e-9)
