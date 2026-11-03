import numpy as np
import pytest
from builders import EPOCH, LEO_R
from hypothesis import given
from hypothesis import strategies as st

from keepout.orbits import EARTH_MU, orbital_period, perigee_radius
from keepout.state import StateVector


def test_orbital_period_of_a_circular_leo() -> None:
    circular_speed = float(np.sqrt(EARTH_MU / 7000e3))
    state = StateVector(EPOCH, LEO_R, [0.0, circular_speed, 0.0], "GCRF")
    assert orbital_period(state) == pytest.approx(
        2 * np.pi * 7000e3 / circular_speed, rel=1e-12
    )


def test_unbound_orbit_has_no_period() -> None:
    state = StateVector(EPOCH, LEO_R, [0.0, 15000.0, 0.0], "GCRF")
    assert orbital_period(state) is None


def test_perigee_of_a_circular_orbit_is_its_radius() -> None:
    circular_speed = float(np.sqrt(EARTH_MU / 7000e3))
    state = StateVector(EPOCH, LEO_R, [0.0, circular_speed, 0.0], "GCRF")
    assert perigee_radius(state) == pytest.approx(7000e3, rel=1e-12)


@given(
    radius=st.floats(6600e3, 40000e3),
    speed_ratio=st.floats(1.0, 1.4),
)
def test_a_state_at_perigee_has_its_own_radius_as_perigee(
    radius: float, speed_ratio: float
) -> None:
    # Perpendicular to the radius and above circular speed: at perigee.
    speed = speed_ratio * float(np.sqrt(EARTH_MU / radius))
    state = StateVector(EPOCH, [radius, 0.0, 0.0], [0.0, speed, 0.0], "GCRF")
    assert perigee_radius(state) == pytest.approx(radius, rel=1e-9)


@given(flight_path=st.floats(-1.2, 1.2), speed_ratio=st.floats(0.8, 1.3))
def test_perigee_is_never_above_the_current_radius(
    flight_path: float, speed_ratio: float
) -> None:
    speed = speed_ratio * float(np.sqrt(EARTH_MU / 7000e3))
    velocity = speed * np.array([np.sin(flight_path), np.cos(flight_path), 0.0])
    state = StateVector(EPOCH, [7000e3, 0.0, 0.0], velocity, "GCRF")
    perigee = perigee_radius(state)
    assert perigee is not None
    assert 0.0 <= perigee <= 7000e3 * (1.0 + 1e-12)


def test_unbound_orbit_has_no_perigee_radius() -> None:
    state = StateVector(EPOCH, LEO_R, [0.0, 15000.0, 0.0], "GCRF")
    assert perigee_radius(state) is None
