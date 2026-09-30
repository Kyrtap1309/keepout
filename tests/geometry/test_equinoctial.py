import numpy as np
import pytest
from scipy.integrate import solve_ivp

from keepout.equinoctial import (
    complex_step_jacobian,
    propagate,
    retrograde_factor,
    to_cartesian,
    to_equinoctial,
    wrap_angle,
)
from keepout.errors import GeometryError
from keepout.orbits import EARTH_MU

LEO = np.array([7000e3, 100e3, -50e3, 10.0, 7400.0, 1200.0])
# Retrograde and exactly equatorial (i = 180 deg): singular for fr = +1.
GEO_RETROGRADE = np.array([153446.76, 41874155.87, 0.0, 3066.87, -11.37, 0.0])
POLAR_ECCENTRIC = np.array([0.0, 7200e3, 0.0, 0.0, 0.0, 8500.0])


def two_body(state: np.ndarray, dt: float) -> np.ndarray:
    """Independent oracle: numerically integrated two-body motion."""

    def rhs(_: float, y: np.ndarray) -> np.ndarray:
        r = y[:3]
        return np.concatenate([y[3:], -EARTH_MU * r / np.linalg.norm(r) ** 3])

    solution = solve_ivp(rhs, (0.0, dt), state, method="DOP853", rtol=1e-13, atol=1e-9)
    result: np.ndarray = solution.y[:, -1]
    return result


@pytest.mark.parametrize("state", [LEO, GEO_RETROGRADE, POLAR_ECCENTRIC])
def test_round_trip(state: np.ndarray) -> None:
    fr = retrograde_factor(state[:3], state[3:])
    back = to_cartesian(to_equinoctial(state, fr), fr)
    np.testing.assert_allclose(back[:3], state[:3], rtol=0, atol=1e-6)
    np.testing.assert_allclose(back[3:], state[3:], rtol=0, atol=1e-9)


def test_retrograde_factor() -> None:
    assert retrograde_factor(LEO[:3], LEO[3:]) == 1
    assert retrograde_factor(GEO_RETROGRADE[:3], GEO_RETROGRADE[3:]) == -1


@pytest.mark.parametrize("state", [LEO, GEO_RETROGRADE, POLAR_ECCENTRIC])
@pytest.mark.parametrize("dt", [-5000.0, 1.0, 3000.0, 20000.0])
def test_mean_longitude_drift_is_two_body_motion(state: np.ndarray, dt: float) -> None:
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


def test_complex_step_jacobian_is_exact() -> None:
    fr = retrograde_factor(LEO[:3], LEO[3:])
    elements = to_equinoctial(LEO, fr)
    jacobian = complex_step_jacobian(lambda e: to_cartesian(e, fr), elements)
    # Moving the mean longitude by d is moving along the orbit for d / n
    # seconds: that column is (velocity, acceleration) / n.
    r = LEO[:3]
    acceleration = -EARTH_MU * r / np.linalg.norm(r) ** 3
    np.testing.assert_allclose(
        jacobian[:, 5],
        np.concatenate([LEO[3:], acceleration]) / elements[0],
        rtol=1e-10,
    )
    inverse = complex_step_jacobian(lambda x: to_equinoctial(x, fr), LEO)
    np.testing.assert_allclose(jacobian @ inverse, np.eye(6), atol=1e-10)


def test_unbound_state_is_rejected() -> None:
    escaping = np.array([7000e3, 0.0, 0.0, 0.0, 11500.0, 0.0])
    with pytest.raises(GeometryError, match="unbound"):
        to_equinoctial(escaping, 1)


def test_wrap_angle() -> None:
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
    escaping = np.array([7000e3, 0.0, 0.0, 0.0, 11500.0, 0.0])
    with pytest.raises(GeometryError, match="unbound"):
        to_equinoctial(np.stack([LEO, escaping, POLAR_ECCENTRIC]), 1)


def test_complex_step_jacobian_retrograde() -> None:
    fr = retrograde_factor(GEO_RETROGRADE[:3], GEO_RETROGRADE[3:])
    assert fr == -1
    elements = to_equinoctial(GEO_RETROGRADE, fr)
    jacobian = complex_step_jacobian(lambda e: to_cartesian(e, fr), elements)
    r = GEO_RETROGRADE[:3]
    acceleration = -EARTH_MU * r / np.linalg.norm(r) ** 3
    np.testing.assert_allclose(
        jacobian[:, 5],
        np.concatenate([GEO_RETROGRADE[3:], acceleration]) / elements[0],
        rtol=1e-9,
    )
    inverse = complex_step_jacobian(lambda x: to_equinoctial(x, fr), GEO_RETROGRADE)
    np.testing.assert_allclose(jacobian @ inverse, np.eye(6), atol=1e-9)


def test_complex_step_jacobian_rejects_real_output() -> None:
    with pytest.raises(TypeError, match="complex"):
        complex_step_jacobian(lambda x: np.real(x) * 2.0, np.ones(3))
