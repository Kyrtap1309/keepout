import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time
from builders import EARTH_ROTATION_RATE, EPOCH, LEO_R, LEO_V
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from keepout._types import FloatArray, StateVector
from keepout.errors import GeometryError
from keepout.frames import (
    FRAME_BIAS,
    inertial_to_rtn_covariance,
    rtn_rotation,
    rtn_to_inertial_covariance,
    to_gcrf,
    to_gcrf_all,
)

MAS = np.pi / (180.0 * 3600.0 * 1000.0)

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
    rotation = rtn_rotation(LEO_R, LEO_V)
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
    state = StateVector(EPOCH, LEO_R, LEO_V, "GCRF")
    assert to_gcrf(state) is state


def test_frame_bias_is_the_iau_2006_rotation() -> None:
    # IERS Conventions 2010, sec. 5.5.4: B = R1(-eta0) R2(xi0) R3(d_alpha0)
    # with d_alpha0 = -14.6 mas, xi0 = -16.617 mas, eta0 = -6.819 mas. To
    # first order B[2, 1] = eta0, B[0, 2] = -xi0, B[1, 0] = -d_alpha0.
    np.testing.assert_allclose(FRAME_BIAS @ FRAME_BIAS.T, np.eye(3), atol=1e-15)
    skew = FRAME_BIAS - np.eye(3)
    np.testing.assert_allclose(
        [skew[2, 1], skew[0, 2], skew[1, 0]],
        np.array([-6.819, 16.617, 14.6]) * MAS,
        atol=0.01 * MAS,
    )


def test_eme2000_is_rotated_by_the_frame_bias() -> None:
    state = StateVector(EPOCH, [7000e3, 1.0, 2.0], [3.0, 7500.0, 4.0], "EME2000")
    converted = to_gcrf(state)
    assert converted.frame == "GCRF"
    assert converted.epoch is state.epoch
    np.testing.assert_allclose(FRAME_BIAS @ converted.r, state.r, atol=1e-8)
    np.testing.assert_allclose(FRAME_BIAS @ converted.v, state.v, atol=1e-11)
    # Along x only the y and z rotations (xi0, d_alpha0) move the vector:
    # 22 mas, i.e. 0.75 m at 7000 km -- enough to matter against a hard body.
    shift = float(np.linalg.norm(converted.r - state.r))
    assert shift == pytest.approx(np.hypot(16.617, 14.6) * MAS * 7000e3, rel=1e-3)


def test_itrf_matches_vallado_example_3_15() -> None:
    # Vallado, Fundamentals of Astrodynamics and Applications (4th ed.),
    # Example 3-15. Residuals of ~0.25 m and ~0.5 mm/s come from astropy's
    # bundled Earth-orientation data differing from the book's.
    epoch = Time("2004-04-06T07:51:28.386009", scale="utc")
    itrf = StateVector(
        epoch,
        np.array([-1033.4793830, 7901.2952754, 6380.3565958]) * 1e3,
        np.array([-3.225636520, -2.872451450, 5.531924446]) * 1e3,
        "ITRF",
    )
    gcrf = to_gcrf(itrf)
    np.testing.assert_allclose(
        gcrf.r, np.array([5102.508958, 6123.011401, 6378.136928]) * 1e3, atol=1.0
    )
    np.testing.assert_allclose(
        gcrf.v,
        np.array([-4.743220157, 0.790536497, 5.533755272]) * 1e3,
        atol=2e-3,
    )


def test_batched_conversion_matches_one_at_a_time() -> None:
    states = [
        StateVector(EPOCH, [6778137.0, 0.0, 0.0], [0.0, 5900.0, 4400.0], "ITRF"),
        StateVector(EPOCH, LEO_R, LEO_V, "GCRF"),
        StateVector(EPOCH, [7000e3, 1.0, 2.0], [3.0, 7500.0, 4.0], "EME2000"),
        StateVector(EPOCH + 60 * u.s, [0.0, 7e6, 1e5], [-7e3, 0.0, 1e3], "ITRF"),
    ]
    batched = to_gcrf_all(states)
    assert batched[1] is states[1]
    for state, together in zip(states, batched, strict=True):
        alone = to_gcrf(state)
        assert together.frame == "GCRF"
        assert together.epoch is state.epoch
        np.testing.assert_allclose(together.r, alone.r, rtol=0, atol=1e-6)
        np.testing.assert_allclose(together.v, alone.v, rtol=0, atol=1e-9)


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
