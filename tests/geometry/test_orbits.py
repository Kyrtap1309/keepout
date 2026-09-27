import numpy as np
import pytest
from builders import EPOCH, LEO_R

from keepout._types import StateVector
from keepout.orbits import EARTH_MU, orbital_period


def test_orbital_period_of_a_circular_leo() -> None:
    circular_speed = float(np.sqrt(EARTH_MU / 7000e3))
    state = StateVector(EPOCH, LEO_R, [0.0, circular_speed, 0.0], "GCRF")
    assert orbital_period(state) == pytest.approx(
        2 * np.pi * 7000e3 / circular_speed, rel=1e-12
    )


def test_unbound_orbit_has_no_period() -> None:
    state = StateVector(EPOCH, LEO_R, [0.0, 15000.0, 0.0], "GCRF")
    assert orbital_period(state) is None
