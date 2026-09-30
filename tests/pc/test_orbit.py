import numpy as np
import pytest
from builders import LEO_R, LEO_V, object_state

from keepout.errors import CovarianceError
from keepout.objects import ObjectState
from keepout.orbits import orbital_period
from keepout.pc._orbit import OrbitUncertainty

# Circular LEO along +y: 60 km along-track sigma, 30 m radial and cross-track.
ALONG_TRACK = object_state(LEO_R, LEO_V, np.diag([30.0**2, 60e3**2, 30.0**2]))


def test_samples_stay_on_the_curved_orbit() -> None:
    # A Cartesian Gaussian moves a 60 km along-track sample along the tangent,
    # (60 km)^2 / (2 * 7000 km) = 257 m above the orbit; in equinoctial
    # elements it stays on the orbit and the radius keeps its 30 m sigma.
    orbit = OrbitUncertainty.of(ALONG_TRACK)
    states = orbit.cartesian(orbit.sample(np.random.default_rng(1), 20_000), 0.0)
    radius = np.linalg.norm(states[:, :3], axis=1) - np.linalg.norm(LEO_R)
    along = states[:, 1]
    assert np.std(along) == pytest.approx(60e3, rel=0.03)
    assert np.std(radius) == pytest.approx(30.0, rel=0.1)
    cartesian_sigma = np.hypot(30.0, np.sqrt(2.0) * 60e3**2 / (2 * 7000e3))
    assert np.std(radius) < 0.2 * cartesian_sigma


def test_mean_centred_expansion_is_the_mean_orbit() -> None:
    orbit = OrbitUncertainty.of(ALONG_TRACK)
    mean, covariance = orbit.expansion(orbit.elements, 0.0)
    np.testing.assert_allclose(mean[:3], LEO_R, atol=1e-6)
    np.testing.assert_allclose(mean[3:], LEO_V, atol=1e-9)
    assert ALONG_TRACK.covariance is not None
    np.testing.assert_allclose(
        covariance, ALONG_TRACK.covariance.matrix, rtol=1e-8, atol=1e-6
    )


def test_centre_at_inverts_the_motion() -> None:
    orbit = OrbitUncertainty.of(ALONG_TRACK)
    later = orbit.cartesian(orbit.elements, 120.0)
    np.testing.assert_allclose(orbit.centre_at(later, 120.0), orbit.elements, atol=1e-9)
    assert orbit.period == pytest.approx(orbital_period(ALONG_TRACK.state), rel=1e-12)


def test_an_unknown_covariance_has_no_orbit_uncertainty() -> None:
    with pytest.raises(CovarianceError, match="covariance of X is unknown"):
        OrbitUncertainty.of(ObjectState("X", ALONG_TRACK.state, None))


def test_batched_expansions_match_one_at_a_time() -> None:
    orbit = OrbitUncertainty.of(
        object_state(LEO_R, LEO_V, np.diag([100.0, 900.0, 25.0]), velocity_sigma=0.1)
    )
    rng = np.random.default_rng(7)
    centres = orbit.sample(rng, 5)
    times = np.array([-300.0, -1.0, 0.0, 2.5, 600.0])
    means, covariances = orbit.expansions(centres, times)
    for centre, t, mean, covariance in zip(
        centres, times, means, covariances, strict=True
    ):
        one_mean, one_covariance = orbit.expansion(centre, float(t))
        np.testing.assert_allclose(mean, one_mean, rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(covariance, one_covariance, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        orbit.centre_at(means, times),
        [orbit.centre_at(m, float(t)) for m, t in zip(means, times, strict=True)],
        rtol=1e-12,
    )
