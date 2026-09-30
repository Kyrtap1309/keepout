import astropy.units as u
import numpy as np
import pytest
from builders import (
    EPOCH,
    LEO_R,
    LEO_V,
    object_state,
    position_covariances,
    rotate,
    rotations,
)
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from keepout._types import FloatArray
from keepout.encounter import encounter_plane
from keepout.errors import GeometryError
from keepout.objects import ObjectState
from keepout.state import StateVector

offsets = arrays(np.float64, (3,), elements=st.floats(-1e4, 1e4))
relative_velocities = arrays(np.float64, (3,), elements=st.floats(-2e4, 2e4))


def test_head_on_encounter() -> None:
    primary = object_state(
        [7000e3, 0.0, 0.0], [0.0, 7000.0, 0.0], np.diag([4.0, 9.0, 16.0])
    )
    secondary = object_state([7000e3 + 100.0, 0.0, 0.0], [0.0, 7000.0, 14000.0])
    plane = encounter_plane(primary, secondary)
    # r = (100, 0, 0), v = (0, 0, 14000): x = (1, 0, 0), y = (0, 0, 1), z = x x y.
    np.testing.assert_allclose(plane.basis, [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    np.testing.assert_allclose(plane.miss_vector, [100.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(plane.covariance, [[5.0, 0.0], [0.0, 10.0]])
    np.testing.assert_array_equal(plane.relative_velocity, [0.0, 0.0, 14000.0])
    np.testing.assert_array_equal(plane.relative_position, [100.0, 0.0, 0.0])
    assert plane.relative_speed == 14000.0
    np.testing.assert_array_equal(plane.direction, [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(plane.combined_covariance, np.diag([5.0, 10.0, 17.0]))
    assert plane.miss_distance == pytest.approx(100.0)
    assert plane.closest_approach_offset == 0.0
    assert plane.tca is EPOCH


def test_along_track_offset_is_projected_out() -> None:
    primary = object_state([7000e3, 0.0, 0.0], [0.0, 7000.0, 0.0])
    secondary = object_state([7000e3 + 100.0, 0.0, 500.0], [0.0, 7000.0, 14000.0])
    plane = encounter_plane(primary, secondary)
    np.testing.assert_allclose(plane.miss_vector, [100.0, 0.0], atol=1e-9)
    assert plane.closest_approach_offset == pytest.approx(500.0)


def test_zero_relative_velocity_is_rejected() -> None:
    primary = object_state([7000e3, 0.0, 0.0], [0.0, 7500.0, 0.0])
    secondary = object_state([7000e3, 100.0, 0.0], [0.0, 7500.0, 0.0])
    with pytest.raises(GeometryError, match="relative velocity is zero"):
        encounter_plane(primary, secondary)


def test_mismatched_epochs_are_rejected() -> None:
    primary = object_state([7000e3, 0.0, 0.0], [0.0, 7500.0, 0.0])
    secondary = object_state(
        [7000e3, 100.0, 0.0], [0.0, 0.0, 7500.0], epoch=EPOCH + 1 * u.s
    )
    with pytest.raises(GeometryError, match="epochs differ"):
        encounter_plane(primary, secondary)


def test_relative_position_along_velocity_falls_back() -> None:
    primary = object_state([7000e3, 0.0, 0.0], [0.0, 7000.0, 0.0])
    secondary = object_state([7000e3, 0.0, 100.0], [0.0, 7000.0, 14000.0])
    plane = encounter_plane(primary, secondary)
    np.testing.assert_allclose(plane.miss_vector, [0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(plane.basis @ plane.basis.T, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(plane.basis @ [0.0, 0.0, 1.0], [0.0, 0.0], atol=1e-12)
    assert np.all(np.isfinite(plane.covariance))


@given(
    offsets,
    relative_velocities,
    rotations(),
    position_covariances(),
    position_covariances(),
)
def test_encounter_plane_is_invariant_under_rotation(
    offset: FloatArray,
    relative_velocity: FloatArray,
    rotation: FloatArray,
    first: FloatArray,
    second: FloatArray,
) -> None:
    speed = float(np.linalg.norm(relative_velocity))
    assume(speed > 1.0)
    along = relative_velocity / speed
    assume(float(np.linalg.norm(offset - (offset @ along) * along)) > 1.0)
    primary = object_state(LEO_R, LEO_V, first)
    secondary = object_state(LEO_R + offset, LEO_V + relative_velocity, second)
    plane = encounter_plane(primary, secondary)
    rotated = encounter_plane(rotate(primary, rotation), rotate(secondary, rotation))
    scale = float(np.max(np.abs(plane.covariance)))
    # Positions near 7e6 m carry ~1e-9 m of round-off into the miss vector.
    np.testing.assert_allclose(rotated.miss_vector, plane.miss_vector, atol=1e-6)
    np.testing.assert_allclose(rotated.covariance, plane.covariance, atol=1e-9 * scale)


@given(offsets, relative_velocities, position_covariances(), position_covariances())
def test_swapping_objects_preserves_the_geometry(
    offset: FloatArray,
    relative_velocity: FloatArray,
    first: FloatArray,
    second: FloatArray,
) -> None:
    assume(float(np.linalg.norm(relative_velocity)) > 1.0)
    primary = object_state(LEO_R, LEO_V, first)
    secondary = object_state(LEO_R + offset, LEO_V + relative_velocity, second)
    forward = encounter_plane(primary, secondary)
    backward = encounter_plane(secondary, primary)
    assert backward.miss_distance == pytest.approx(forward.miss_distance, abs=1e-9)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(backward.covariance),
        np.linalg.eigvalsh(forward.covariance),
        rtol=1e-9,
    )
    forward_mahalanobis = forward.miss_vector @ np.linalg.solve(
        forward.covariance, forward.miss_vector
    )
    backward_mahalanobis = backward.miss_vector @ np.linalg.solve(
        backward.covariance, backward.miss_vector
    )
    assert backward_mahalanobis == pytest.approx(
        forward_mahalanobis, rel=1e-9, abs=1e-12
    )


def test_unknown_covariance_contributes_nothing() -> None:
    known_object = object_state(LEO_R, LEO_V, np.diag([4.0, 9.0, 16.0]))
    offset = np.array([0.0, 0.0, 30.0])
    unknown = ObjectState(
        "SEC", StateVector(EPOCH, LEO_R + offset, -LEO_V, "GCRF"), None
    )
    plane = encounter_plane(known_object, unknown)
    assert plane.unknown_covariance == ("secondary",)
    np.testing.assert_array_equal(plane.combined_covariance, np.diag([4.0, 9.0, 16.0]))
    both = encounter_plane(ObjectState("PRI", known_object.state, None), unknown)
    assert both.unknown_covariance == ("primary", "secondary")
    np.testing.assert_array_equal(both.covariance, np.zeros((2, 2)))
