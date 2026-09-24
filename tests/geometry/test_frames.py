import numpy as np
import pytest
from geometry_builders import EPOCH
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from keepout._types import FloatArray, StateVector
from keepout.errors import GeometryError
from keepout.frames import (
    inertial_to_rtn_covariance,
    rtn_rotation,
    rtn_to_inertial_covariance,
    to_gcrf,
)

EARTH_ROTATION_RATE = 7.292115146706979e-5  # rad/s (IERS conventions)

vectors = arrays(np.float64, (3,), elements=st.floats(min_value=-1e7, max_value=1e7))
square_roots = arrays(
    np.float64, (6, 6), elements=st.floats(min_value=-100.0, max_value=100.0)
)


def well_posed(r: FloatArray, v: FloatArray) -> bool:
    r_norm = float(np.linalg.norm(r))
    v_norm = float(np.linalg.norm(v))
    h_norm = float(np.linalg.norm(np.cross(r, v)))
    return r_norm > 1.0 and v_norm > 1e-3 and h_norm > 1e-3 * r_norm * v_norm


def test_rtn_of_circular_equatorial_orbit_is_identity() -> None:
    rotation = rtn_rotation([7000e3, 0.0, 0.0], [0.0, 7500.0, 0.0])
    np.testing.assert_allclose(rotation, np.eye(3), atol=1e-15)


@given(vectors, vectors)
def test_rtn_rotation_is_proper_and_aligned(r: FloatArray, v: FloatArray) -> None:
    assume(well_posed(r, v))
    rotation = rtn_rotation(r, v)
    r_norm = float(np.linalg.norm(r))
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-12)
    radial = rotation @ r
    assert radial[0] == pytest.approx(r_norm, rel=1e-12)
    np.testing.assert_allclose(radial[1:], 0.0, atol=1e-9 * r_norm)
    assert abs((rotation @ v)[2]) <= 1e-9 * float(np.linalg.norm(v))


@pytest.mark.parametrize(
    ("r", "v"),
    [
        ([0.0, 0.0, 0.0], [0.0, 7500.0, 0.0]),
        ([7000e3, 0.0, 0.0], [7500.0, 0.0, 0.0]),
        ([7000e3, 0.0, 0.0], [0.0, 0.0, 0.0]),
    ],
)
def test_rtn_rotation_rejects_degenerate_states(r: list[float], v: list[float]) -> None:
    with pytest.raises(GeometryError, match="RTN frame is undefined"):
        rtn_rotation(r, v)


@given(vectors, vectors, square_roots)
def test_covariance_rtn_round_trip(
    r: FloatArray, v: FloatArray, root: FloatArray
) -> None:
    assume(well_posed(r, v))
    covariance = root @ root.T
    scale = max(float(np.max(np.abs(covariance))), 1.0)
    rtn = inertial_to_rtn_covariance(covariance, r, v)
    back = rtn_to_inertial_covariance(rtn, r, v)
    np.testing.assert_allclose(back, covariance, atol=1e-12 * scale)
    assert np.trace(rtn[:3, :3]) == pytest.approx(
        np.trace(covariance[:3, :3]), rel=1e-12, abs=1e-12 * scale
    )


def test_gcrf_state_is_returned_unchanged() -> None:
    state = StateVector(EPOCH, [7000e3, 0.0, 0.0], [0.0, 7500.0, 0.0], "GCRF")
    assert to_gcrf(state) is state


def test_eme2000_is_relabelled_as_gcrf() -> None:
    state = StateVector(EPOCH, [7000e3, 1.0, 2.0], [3.0, 7500.0, 4.0], "EME2000")
    converted = to_gcrf(state)
    assert converted.frame == "GCRF"
    assert converted.epoch is state.epoch
    np.testing.assert_array_equal(converted.r, state.r)
    np.testing.assert_array_equal(converted.v, state.v)


def test_itrf_point_fixed_on_the_equator_moves_with_earth_rotation() -> None:
    radius = 6378137.0
    state = StateVector(EPOCH, [radius, 0.0, 0.0], [0.0, 0.0, 0.0], "ITRF")
    converted = to_gcrf(state)
    assert converted.frame == "GCRF"
    assert np.linalg.norm(converted.r) == pytest.approx(radius, abs=1e-3)
    assert np.linalg.norm(converted.v) == pytest.approx(
        EARTH_ROTATION_RATE * radius, abs=1e-3
    )


def test_itrf_velocity_gains_the_earth_rotation_term() -> None:
    r = np.array([6778137.0, 0.0, 0.0])
    v = np.array([0.0, 5900.0, 4400.0])
    converted = to_gcrf(StateVector(EPOCH, r, v, "ITRF"))
    inertial = v + np.cross([0.0, 0.0, EARTH_ROTATION_RATE], r)
    assert np.linalg.norm(converted.r) == pytest.approx(np.linalg.norm(r), abs=1e-3)
    assert np.linalg.norm(converted.v) == pytest.approx(
        np.linalg.norm(inertial), abs=1e-2
    )
