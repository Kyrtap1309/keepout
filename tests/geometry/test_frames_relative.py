"""Relative positions in an object's RTN frame, many at once."""

import numpy as np
import pytest
from builders import LEO_R, LEO_V
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from keepout._types import FloatArray
from keepout.errors import GeometryError
from keepout.frames import relative_position_rtn, rtn_rotation

positions = arrays(np.float64, (3,), elements=st.floats(min_value=-5e7, max_value=5e7))
velocities = arrays(np.float64, (3,), elements=st.floats(min_value=-1e4, max_value=1e4))
offsets = arrays(np.float64, (3,), elements=st.floats(min_value=-1e5, max_value=1e5))


def well_posed(r: FloatArray, v: FloatArray) -> bool:
    r_norm = float(np.linalg.norm(r))
    v_norm = float(np.linalg.norm(v))
    h_norm = float(np.linalg.norm(np.cross(r, v)))
    return r_norm > 1.0 and v_norm > 1e-3 and h_norm > 1e-3 * r_norm * v_norm


def test_offsets_along_the_axes_of_a_circular_equatorial_orbit() -> None:
    # R is +x, T is +y (the motion), N is +z (the angular momentum).
    other = LEO_R + np.array([[10.0, 0.0, 0.0], [0.0, -20.0, 0.0], [0.0, 0.0, 30.0]])
    np.testing.assert_allclose(
        relative_position_rtn(LEO_R, LEO_V, other),
        [[10.0, 0.0, 0.0], [0.0, -20.0, 0.0], [0.0, 0.0, 30.0]],
        rtol=0,
        atol=1e-9,
    )


@given(positions, velocities, offsets)
def test_agrees_with_the_rtn_rotation(
    r: FloatArray, v: FloatArray, offset: FloatArray
) -> None:
    assume(well_posed(r, v))
    expected = rtn_rotation(r, v) @ offset
    result = relative_position_rtn(r, v, r + offset)
    # Rounding: r + offset - r loses the digits of r that offset lacks.
    tolerance = 1e-14 * (float(np.linalg.norm(r)) + float(np.linalg.norm(offset)))
    np.testing.assert_allclose(result, expected, rtol=0, atol=tolerance)
    # A rotation keeps the distance.
    assert float(np.linalg.norm(result)) == pytest.approx(
        float(np.linalg.norm(offset)), rel=0, abs=tolerance
    )


def test_broadcasts_over_pairs_and_times() -> None:
    rng = np.random.default_rng(7)
    r = LEO_R + rng.normal(0.0, 1e5, (4, 1, 3))  # 4 primaries
    v = LEO_V + rng.normal(0.0, 1e2, (4, 1, 3))
    other = LEO_R + rng.normal(0.0, 1e5, (5, 3))  # 5 secondaries
    result = relative_position_rtn(r, v, other)
    assert result.shape == (4, 5, 3)
    for i in range(4):
        for j in range(5):
            expected = rtn_rotation(r[i, 0], v[i, 0]) @ (other[j] - r[i, 0])
            np.testing.assert_allclose(result[i, j], expected, rtol=0, atol=1e-8)


def test_non_finite_rows_stay_non_finite_alone() -> None:
    r = np.array([LEO_R, [np.nan] * 3])
    v = np.array([LEO_V, [np.nan] * 3])
    other = np.array([LEO_R + 1.0, LEO_R + 1.0])
    result = relative_position_rtn(r, v, other)
    np.testing.assert_allclose(result[0], [1.0, 1.0, 1.0], rtol=0, atol=1e-9)
    assert np.all(np.isnan(result[1]))


@pytest.mark.parametrize(
    ("r", "v"),
    [
        ([0.0, 0.0, 0.0], [0.0, 7500.0, 0.0]),
        ([7000e3, 0.0, 0.0], [7500.0, 0.0, 0.0]),
        ([7000e3, 0.0, 0.0], [0.0, 0.0, 0.0]),
    ],
)
def test_degenerate_states_are_refused(r: list[float], v: list[float]) -> None:
    with pytest.raises(GeometryError, match="RTN frame is undefined"):
        relative_position_rtn(np.array([LEO_R, r]), np.array([LEO_V, v]), LEO_R)


def test_shapes_are_checked() -> None:
    with pytest.raises(ValueError, match="r must have 3 components"):
        relative_position_rtn([1.0, 2.0], LEO_V, LEO_R)
    with pytest.raises(ValueError, match="r_other must have 3 components"):
        relative_position_rtn(LEO_R, LEO_V, 1.0)
    with pytest.raises(ValueError, match=r"r \(2, 3\), v \(3, 3\) and r_other"):
        relative_position_rtn(np.zeros((2, 3)), np.zeros((3, 3)), LEO_R)
