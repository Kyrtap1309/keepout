import numpy as np
import pytest
from pc_builders import EPOCH, LEO_R, LEO_V, head_on, object_state

from keepout.conjunction import Conjunction, combined_hard_body_radius
from keepout.errors import GeometryError
from keepout.objects import ObjectState


def with_radius(obj: ObjectState, radius: float | None) -> ObjectState:
    return ObjectState(obj.object_id, obj.state, obj.covariance, radius)


def test_conjunction_exposes_encounter_geometry() -> None:
    conjunction = head_on(miss=30.0, sigma_x=10.0, sigma_z=20.0, radius=5.0)
    assert conjunction.miss_distance == pytest.approx(30.0, rel=1e-12)
    assert conjunction.relative_speed == pytest.approx(2 * 7546.0, rel=1e-12)
    assert conjunction.tca == EPOCH
    np.testing.assert_allclose(
        conjunction.plane.covariance, np.diag([100.0, 400.0]), atol=1e-9
    )


@pytest.mark.parametrize("radius", [0.0, -1.0, float("nan"), float("inf")])
def test_radius_must_be_finite_and_positive(radius: float) -> None:
    with pytest.raises(ValueError, match="hard_body_radius"):
        head_on(miss=30.0, sigma_x=10.0, sigma_z=20.0, radius=radius)


def test_zero_relative_velocity_is_rejected() -> None:
    primary = object_state(LEO_R, LEO_V, np.eye(3))
    secondary = object_state(LEO_R + np.array([10.0, 0.0, 0.0]), LEO_V, np.eye(3))
    with pytest.raises(GeometryError, match="relative velocity is zero"):
        Conjunction(primary, secondary, 5.0)


def test_explicit_radius_wins() -> None:
    base = head_on(miss=30.0, sigma_x=10.0, sigma_z=20.0, radius=5.0)
    primary = with_radius(base.primary, 3.0)
    secondary = with_radius(base.secondary, 4.0)
    assert combined_hard_body_radius(primary, secondary, 10.0) == 10.0


def test_radius_is_the_sum_of_object_radii() -> None:
    base = head_on(miss=30.0, sigma_x=10.0, sigma_z=20.0, radius=5.0)
    primary = with_radius(base.primary, 3.0)
    secondary = with_radius(base.secondary, 4.0)
    assert combined_hard_body_radius(primary, secondary) == 7.0


def test_missing_radius_is_an_error_not_a_default() -> None:
    base = head_on(miss=30.0, sigma_x=10.0, sigma_z=20.0, radius=5.0)
    primary = with_radius(base.primary, 3.0)
    with pytest.raises(ValueError, match="no hard-body radius for SEC"):
        combined_hard_body_radius(primary, base.secondary)
