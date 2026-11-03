import math

import astropy.units as u
import erfa
import numpy as np
import pytest
from astropy.time import Time
from builders import EARTH_ROTATION_RATE, EPOCH, LEO_R, LEO_V, approx, well_posed
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from keepout._types import FloatArray
from keepout.errors import GeometryError, UnsupportedFrame
from keepout.frames import (
    FRAME_BIAS,
    frame_to_gcrf,
    inertial_to_rtn_covariance,
    rtn_rotation,
    rtn_to_inertial_covariance,
    teme_to_gcrf,
    to_gcrf,
    to_gcrf_all,
)
from keepout.state import RefFrame, StateVector

MAS = np.pi / (180.0 * 3600.0 * 1000.0)

vectors = arrays(np.float64, (3,), elements=st.floats(min_value=-1e7, max_value=1e7))
square_roots = arrays(
    np.float64, (6, 6), elements=st.floats(min_value=-100.0, max_value=100.0)
)


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
    assert radial[0] == approx(r_norm, rel=1e-12)
    np.testing.assert_allclose(radial[1:], 0.0, atol=1e-9 * r_norm)
    in_rtn = rotation @ v
    assert abs(in_rtn[2]) <= 1e-9 * float(np.linalg.norm(v))
    # T points along the motion and N along the angular momentum, so flipping
    # both (still a proper rotation) is caught.
    assert in_rtn[1] > 0.0
    angular_momentum = np.cross(r, v)
    np.testing.assert_allclose(
        rotation[2],
        angular_momentum / np.linalg.norm(angular_momentum),
        atol=1e-9,
    )


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


# An inclined, eccentric state far from the inertial axes, so that the RTN
# rotation is not close to symmetric and transposing it moves every axis.
TILTED_R = np.array([4200e3, -3900e3, 4100e3])
TILTED_V = np.array([5200.0, 4600.0, -1300.0])


def rtn_axes(r: FloatArray, v: FloatArray) -> dict[str, FloatArray]:
    """The R, T, N unit vectors built from their definitions, not from the code."""
    angular_momentum = np.cross(r, v)
    in_track = np.cross(angular_momentum, r)
    return {
        "R": r / np.linalg.norm(r),
        "T": in_track / np.linalg.norm(in_track),
        "N": angular_momentum / np.linalg.norm(angular_momentum),
    }


AXES = rtn_axes(TILTED_R, TILTED_V)


@pytest.mark.parametrize(("axis", "index"), [("R", 0), ("T", 1), ("N", 2)])
def test_rtn_variance_along_one_axis_lies_along_that_axis_inertially(
    axis: str, index: int
) -> None:
    rtn = np.zeros((6, 6))
    rtn[index, index] = 100.0
    rtn[3 + index, 3 + index] = 1e-2
    inertial = rtn_to_inertial_covariance(rtn, TILTED_R, TILTED_V)
    direction = AXES[axis]
    np.testing.assert_allclose(
        inertial[:3, :3], 100.0 * np.outer(direction, direction), atol=1e-12
    )
    np.testing.assert_allclose(
        inertial[3:, 3:], 1e-2 * np.outer(direction, direction), atol=1e-16
    )
    np.testing.assert_allclose(inertial[:3, 3:], 0.0, atol=0.0)


@pytest.mark.parametrize(("axis", "index"), [("R", 0), ("T", 1), ("N", 2)])
def test_inertial_variance_along_an_rtn_axis_is_that_rtn_variance(
    axis: str, index: int
) -> None:
    direction = AXES[axis]
    inertial = np.zeros((6, 6))
    inertial[:3, :3] = 100.0 * np.outer(direction, direction)
    inertial[3:, 3:] = 1e-2 * np.outer(direction, direction)
    rtn = inertial_to_rtn_covariance(inertial, TILTED_R, TILTED_V)
    expected = np.zeros((6, 6))
    expected[index, index] = 100.0
    expected[3 + index, 3 + index] = 1e-2
    np.testing.assert_allclose(rtn, expected, atol=1e-12)


def test_positive_radial_in_track_correlation_stretches_along_r_plus_t() -> None:
    # sigma_R = sigma_T = 10 m with correlation +0.5: 150 m^2 along R + T,
    # 50 m^2 along R - T. A sign or orientation slip swaps the two.
    rtn = np.zeros((6, 6))
    rtn[:3, :3] = [[100.0, 50.0, 0.0], [50.0, 100.0, 0.0], [0.0, 0.0, 1.0]]
    position = rtn_to_inertial_covariance(rtn, TILTED_R, TILTED_V)[:3, :3]
    along = (AXES["R"] + AXES["T"]) / np.sqrt(2.0)
    across = (AXES["R"] - AXES["T"]) / np.sqrt(2.0)
    assert along @ position @ along == approx(150.0, rel=1e-12)
    assert across @ position @ across == approx(50.0, rel=1e-12)
    assert AXES["R"] @ position @ AXES["T"] == approx(50.0, rel=1e-12)


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
    assert shift == approx(np.hypot(16.617, 14.6) * MAS * 7000e3, rel=1e-3)


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


@pytest.mark.parametrize("frame", ["GCRF", "EME2000", "ITRF", "TEME"])
def test_arrays_in_any_frame_convert_as_their_state_vectors(frame: RefFrame) -> None:
    # Two epochs x three objects, as a propagator's (t, n, 3) grid.
    epochs = Time([EPOCH, EPOCH + 60 * u.s]).reshape(-1, 1)
    r = np.array([LEO_R, [7000e3, 1.0, 2.0], [0.0, 7e6, 1e5]])
    v = np.array([LEO_V, [3.0, 7500.0, 4.0], [-7e3, 0.0, 1e3]])
    gcrf_r, gcrf_v = frame_to_gcrf(frame, epochs, r, v)
    assert gcrf_r.shape == gcrf_v.shape == (2, 3, 3)
    for t, n in np.ndindex(2, 3):
        alone = to_gcrf(StateVector(epochs[t, 0], r[n], v[n], frame))
        np.testing.assert_allclose(gcrf_r[t, n], alone.r, rtol=0, atol=1e-6)
        np.testing.assert_allclose(gcrf_v[t, n], alone.v, rtol=0, atol=1e-9)
    if frame == "TEME":
        expected = teme_to_gcrf(epochs, r, v)
        np.testing.assert_array_equal(gcrf_r, expected[0])
        np.testing.assert_array_equal(gcrf_v, expected[1])


def test_array_conversion_needs_a_supported_frame() -> None:
    with pytest.raises(UnsupportedFrame, match="'ICRF'"):
        frame_to_gcrf("ICRF", EPOCH, LEO_R, LEO_V)  # type: ignore[arg-type]


def test_array_conversion_needs_matching_shapes() -> None:
    with pytest.raises(ValueError, match="do not broadcast"):
        frame_to_gcrf("GCRF", Time([EPOCH] * 2), np.zeros((3, 3)), np.zeros((3, 3)))


def angle_deg(a: FloatArray, b: FloatArray) -> float:
    cosine = float(a @ b) / float(np.linalg.norm(a) * np.linalg.norm(b))
    return math.degrees(math.acos(min(1.0, max(-1.0, cosine))))


def test_itrf_point_fixed_on_the_equator_moves_with_earth_rotation() -> None:
    radius = 6378137.0
    state = StateVector(EPOCH, [radius, 0.0, 0.0], [0.0, 0.0, 0.0], "ITRF")
    converted = to_gcrf(state)
    assert converted.frame == "GCRF"
    assert np.linalg.norm(converted.r) == pytest.approx(radius, abs=1e-3)
    assert np.linalg.norm(converted.v) == pytest.approx(
        EARTH_ROTATION_RATE * radius, abs=1e-3
    )
    # Independent of the conversion chain: the Greenwich meridian is at the
    # Earth rotation angle (UTC for UT1 here: < 0.01 deg), up to precession
    # and nutation since J2000 (< 0.3 deg in 2010). A transposed rotation
    # would put it at -ERA.
    era = erfa.era00(EPOCH.utc.jd1, EPOCH.utc.jd2)
    meridian = radius * np.array([math.cos(era), math.sin(era), 0.0])
    assert angle_deg(converted.r, meridian) < 0.3
    # Eastward: along z x r, perpendicular to r.
    east = np.cross([0.0, 0.0, 1.0], converted.r)
    assert angle_deg(converted.v, east) < 0.3
    assert abs(float(converted.r @ converted.v)) < 1e-6 * radius


def test_itrf_velocity_gains_the_earth_rotation_term() -> None:
    r = np.array([6778137.0, 0.0, 0.0])
    v = np.array([0.0, 5900.0, 4400.0])
    converted = to_gcrf(StateVector(EPOCH, r, v, "ITRF"))
    inertial = v + np.cross([0.0, 0.0, EARTH_ROTATION_RATE], r)
    assert np.linalg.norm(converted.r) == pytest.approx(np.linalg.norm(r), abs=1e-3)
    assert np.linalg.norm(converted.v) == pytest.approx(
        np.linalg.norm(inertial), abs=1e-2
    )
