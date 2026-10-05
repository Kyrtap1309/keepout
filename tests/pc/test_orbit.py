import numpy as np
import pytest
from builders import LEO_R, LEO_V, known, object_state

from keepout._types import FloatArray, symmetric_part
from keepout.equinoctial import complex_step, propagate, to_cartesian, wrap_angle
from keepout.errors import CovarianceError
from keepout.objects import ObjectState
from keepout.orbits import orbital_period
from keepout.pc._orbit import OrbitUncertainty

# Circular LEO along +y: 60 km along-track sigma, 30 m radial and cross-track.
ALONG_TRACK = object_state(LEO_R, LEO_V, np.diag([30.0**2, 60e3**2, 30.0**2]))


def cartesian(orbit: OrbitUncertainty, elements: FloatArray, t: float) -> FloatArray:
    """Cartesian states at ``t`` [s from TCA] of TCA elements."""
    return np.asarray(to_cartesian(propagate(elements, t), orbit.fr))


def expansion(
    orbit: OrbitUncertainty, centre: FloatArray, t: float
) -> tuple[FloatArray, FloatArray]:
    """Hall's eq. 44 at one time, by a scalar complex step: the reference for
    the vectorised :meth:`OrbitUncertainty.expansions`."""
    state, jacobian = complex_step(
        lambda e: to_cartesian(propagate(e, t), orbit.fr), centre
    )
    offset = orbit.elements - centre
    offset[5] = wrap_angle(offset[5])
    return state + jacobian @ offset, symmetric_part(
        jacobian @ orbit.covariance @ jacobian.T
    )


def test_samples_stay_on_the_curved_orbit() -> None:
    # A Cartesian Gaussian moves a 60 km along-track sample along the tangent,
    # (60 km)^2 / (2 * 7000 km) = 257 m above the orbit; in equinoctial
    # elements it stays on the orbit and the radius keeps its 30 m sigma.
    orbit = OrbitUncertainty.of(ALONG_TRACK)
    states = cartesian(orbit, orbit.sample(np.random.default_rng(1), 20_000), 0.0)
    radius = np.linalg.norm(states[:, :3], axis=1) - np.linalg.norm(LEO_R)
    along = states[:, 1]
    assert np.std(along) == pytest.approx(60e3, rel=0.03)
    assert np.std(radius) == pytest.approx(30.0, rel=0.1)
    cartesian_sigma = np.hypot(30.0, np.sqrt(2.0) * 60e3**2 / (2 * 7000e3))
    assert np.std(radius) < 0.2 * cartesian_sigma


def test_mean_centred_expansion_is_the_mean_orbit() -> None:
    orbit = OrbitUncertainty.of(ALONG_TRACK)
    (mean,), (covariance,) = orbit.mean_expansions(np.array([0.0]))
    np.testing.assert_allclose(mean[:3], LEO_R, atol=1e-6)
    np.testing.assert_allclose(mean[3:], LEO_V, atol=1e-9)
    np.testing.assert_allclose(
        covariance, known(ALONG_TRACK.covariance).matrix, rtol=1e-8, atol=1e-6
    )


def test_centre_at_inverts_the_motion() -> None:
    orbit = OrbitUncertainty.of(ALONG_TRACK)
    later = cartesian(orbit, orbit.elements, 120.0)
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
        one_mean, one_covariance = expansion(orbit, centre, float(t))
        np.testing.assert_allclose(mean, one_mean, rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(covariance, one_covariance, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        orbit.centre_at(means, times),
        [orbit.centre_at(m, float(t)) for m, t in zip(means, times, strict=True)],
        rtol=1e-12,
    )


def test_samples_do_not_depend_on_the_eigenvector_signs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = OrbitUncertainty.of(ALONG_TRACK).sample(np.random.default_rng(3), 50)
    # A fresh orbit: the factor of the covariance is computed once per orbit.
    orbit = OrbitUncertainty.of(ALONG_TRACK)
    eigh = np.linalg.eigh
    calls: list[FloatArray] = []

    def flipped(matrix: FloatArray) -> tuple[FloatArray, FloatArray]:
        calls.append(matrix)
        values, vectors = eigh(matrix)
        return values, vectors * np.array([1.0, -1.0, 1.0, -1.0, -1.0, 1.0])

    monkeypatch.setattr(np.linalg, "eigh", flipped)
    np.testing.assert_allclose(
        orbit.sample(np.random.default_rng(3), 50), expected, rtol=1e-12, atol=1e-15
    )
    assert calls  # the flipped decomposition was used


def test_the_covariance_is_factored_once(monkeypatch: pytest.MonkeyPatch) -> None:
    orbit = OrbitUncertainty.of(ALONG_TRACK)
    orbit.sample(np.random.default_rng(1), 10)
    monkeypatch.setattr(np.linalg, "eigh", None)  # any further factoring fails
    assert orbit.sample(np.random.default_rng(1), 10).shape == (10, 6)
