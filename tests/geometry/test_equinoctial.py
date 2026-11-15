from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from scipy.integrate import solve_ivp

from keepout._types import ComplexArray, FloatArray
from keepout.equinoctial import (
    complex_step,
    propagate,
    retrograde_factor,
    to_cartesian,
    to_equinoctial,
    wrap_angle,
)
from keepout.errors import GeometryError
from keepout.orbits import EARTH_MU


def complex_step_jacobian(
    function: Callable[[ComplexArray], npt.NDArray[Any]], x: npt.ArrayLike
) -> FloatArray:
    """The Jacobian of :func:`complex_step`."""
    return complex_step(function, x)[1]


LEO = np.array([7000e3, 100e3, -50e3, 10.0, 7400.0, 1200.0])
# Retrograde and exactly equatorial (i = 180 deg): singular for fr = +1.
GEO_RETROGRADE = np.array([153446.76, 41874155.87, 0.0, 3066.87, -11.37, 0.0])
POLAR_ECCENTRIC = np.array([0.0, 7200e3, 0.0, 0.0, 0.0, 8500.0])
ESCAPING = np.array([7000e3, 0.0, 0.0, 0.0, 11500.0, 0.0])


def two_body(state: FloatArray, dt: float) -> FloatArray:
    """Independent oracle: numerically integrated two-body motion."""

    def rhs(_: float, y: FloatArray) -> FloatArray:
        r = y[:3]
        return np.concatenate([y[3:], -EARTH_MU * r / np.linalg.norm(r) ** 3])

    solution = solve_ivp(rhs, (0.0, dt), state, method="DOP853", rtol=1e-13, atol=1e-9)
    result: FloatArray = solution.y[:, -1]
    return result


@pytest.mark.parametrize("state", [LEO, GEO_RETROGRADE, POLAR_ECCENTRIC])
def test_cartesian_state_survives_the_equinoctial_round_trip(state: FloatArray) -> None:
    fr = retrograde_factor(state[:3], state[3:])
    back = to_cartesian(to_equinoctial(state, fr), fr)
    np.testing.assert_allclose(back[:3], state[:3], rtol=0, atol=1e-6)
    np.testing.assert_allclose(back[3:], state[3:], rtol=0, atol=1e-9)


def test_the_retrograde_factor_is_minus_one_for_a_retrograde_orbit() -> None:
    assert retrograde_factor(LEO[:3], LEO[3:]) == 1
    assert retrograde_factor(GEO_RETROGRADE[:3], GEO_RETROGRADE[3:]) == -1


@pytest.mark.parametrize("state", [LEO, GEO_RETROGRADE, POLAR_ECCENTRIC])
@pytest.mark.parametrize("dt", [-5000.0, 1.0, 3000.0, 20000.0])
def test_mean_longitude_drift_is_two_body_motion(state: FloatArray, dt: float) -> None:
    fr = retrograde_factor(state[:3], state[3:])
    moved = to_cartesian(propagate(to_equinoctial(state, fr), dt), fr)
    expected = two_body(state, dt)
    np.testing.assert_allclose(moved[:3], expected[:3], rtol=0, atol=1e-3)
    np.testing.assert_allclose(moved[3:], expected[3:], rtol=0, atol=1e-6)


def test_arrays_of_states_match_single_states() -> None:
    states = np.stack([LEO, POLAR_ECCENTRIC, LEO * 1.01])
    elements = to_equinoctial(states, 1)
    assert elements.shape == (3, 6)
    for row, state in zip(elements, states, strict=True):
        np.testing.assert_array_equal(row, to_equinoctial(state, 1))
    np.testing.assert_allclose(to_cartesian(elements, 1), states, rtol=1e-12, atol=1e-8)


def test_propagate_accepts_one_step_per_row() -> None:
    elements = to_equinoctial(np.stack([LEO, LEO]), 1)
    moved = propagate(elements, np.array([0.0, 10.0]))
    np.testing.assert_array_equal(moved[0], elements[0])
    assert moved[1, 5] == pytest.approx(elements[1, 5] + 10.0 * elements[1, 0])


@pytest.mark.parametrize(
    ("state", "tolerance"),
    [
        pytest.param(LEO, 1e-10, id="prograde"),
        pytest.param(GEO_RETROGRADE, 1e-9, id="retrograde"),
    ],
)
def test_complex_step_jacobian_is_exact(state: FloatArray, tolerance: float) -> None:
    fr = retrograde_factor(state[:3], state[3:])
    elements = to_equinoctial(state, fr)
    jacobian = complex_step_jacobian(lambda e: to_cartesian(e, fr), elements)
    # Moving the mean longitude by d is moving along the orbit for d / n
    # seconds: that column is (velocity, acceleration) / n.
    r = state[:3]
    acceleration = -EARTH_MU * r / np.linalg.norm(r) ** 3
    np.testing.assert_allclose(
        jacobian[:, 5],
        np.concatenate([state[3:], acceleration]) / elements[0],
        rtol=tolerance,
    )
    inverse = complex_step_jacobian(lambda x: to_equinoctial(x, fr), state)
    np.testing.assert_allclose(jacobian @ inverse, np.eye(6), atol=tolerance)


def test_unbound_state_is_rejected() -> None:
    with pytest.raises(GeometryError, match="unbound"):
        to_equinoctial(ESCAPING, 1)


def test_angles_wrap_into_one_turn_about_zero() -> None:
    assert wrap_angle(2 * np.pi + 0.1) == pytest.approx(0.1)
    assert wrap_angle(-np.pi - 0.1) == pytest.approx(np.pi - 0.1)


@pytest.mark.parametrize("perihelion_longitude", [0.0, 1.0, -2.5])
@pytest.mark.parametrize("eccentricity", [0.9, 0.975, 0.99, 0.999])
def test_high_eccentricity_round_trip(
    eccentricity: float, perihelion_longitude: float
) -> None:
    lam = np.linspace(-np.pi, np.pi, 20001)
    elements = np.zeros((lam.size, 6))
    elements[:, 0] = 1e-3
    elements[:, 1] = eccentricity * np.cos(perihelion_longitude)
    elements[:, 2] = eccentricity * np.sin(perihelion_longitude)
    elements[:, 5] = lam
    back = to_equinoctial(to_cartesian(elements, 1), 1)
    np.testing.assert_allclose(wrap_angle(back[:, 5] - lam), 0.0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(back[:, 1:3], elements[:, 1:3], rtol=0, atol=1e-9)


def test_large_mean_longitude_converges() -> None:
    elements = to_equinoctial(POLAR_ECCENTRIC, 1)
    far = propagate(elements, 1e7)  # ~ 1e4 rad of mean longitude
    period = 2 * np.pi / elements[0]
    near = propagate(elements, 1e7 - period * np.floor(1e7 / period))
    np.testing.assert_allclose(
        to_cartesian(far, 1)[:3], to_cartesian(near, 1)[:3], rtol=0, atol=1e-3
    )


def test_unconverged_kepler_is_rejected() -> None:
    elements = np.array([1e-3, 0.5, 0.0, 0.0, 0.0, np.nan])
    with pytest.raises(GeometryError, match="did not converge"):
        to_cartesian(elements, 1)


def test_rectilinear_state_is_rejected() -> None:
    with pytest.raises(GeometryError, match="angular momentum"):
        to_equinoctial(np.array([7000e3, 0.0, 0.0, 100.0, 0.0, 0.0]), 1)


def test_mismatched_retrograde_factor_is_rejected() -> None:
    prograde_equatorial = np.array([7000e3, 0.0, 0.0, 0.0, 7500.0, 0.0])
    with pytest.raises(GeometryError, match="retrograde factor"):
        to_equinoctial(prograde_equatorial, -1)


def test_one_unbound_row_rejects_the_batch() -> None:
    with pytest.raises(GeometryError, match="unbound"):
        to_equinoctial(np.stack([LEO, ESCAPING, POLAR_ECCENTRIC]), 1)


def test_complex_step_jacobian_rejects_real_output() -> None:
    with pytest.raises(TypeError, match="complex"):
        complex_step_jacobian(lambda x: np.real(x) * 2.0, np.ones(3))


def test_batched_complex_step_matches_each_point() -> None:
    states = np.stack([LEO, POLAR_ECCENTRIC])
    elements = to_equinoctial(states, 1)
    dt = np.array([0.0, 1500.0])
    # Each point's perturbed copies lie along axis 1, so its time broadcasts.
    values, jacobians = complex_step(
        lambda e: to_cartesian(propagate(e, dt[:, None]), 1), elements
    )
    assert values.shape == (2, 6)
    assert jacobians.shape == (2, 6, 6)
    for row, t in enumerate(dt):

        def at_t(e: ComplexArray, t: float = float(t)) -> ComplexArray:
            return to_cartesian(propagate(e, t), 1)

        value, jacobian = complex_step(at_t, elements[row])
        np.testing.assert_array_equal(values[row], value)
        np.testing.assert_array_equal(jacobians[row], jacobian)
