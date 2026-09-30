import dataclasses

import numpy as np
import pytest
from builders import EPOCH, LEO_R, LEO_V, head_on, object_state

from keepout.conjunction import Conjunction, combined_hard_body_radius
from keepout.covariance import make_covariance
from keepout.errors import GeometryError
from keepout.objects import ObjectState


def with_radius(obj: ObjectState, radius: float | None) -> ObjectState:
    return dataclasses.replace(obj, hard_body_radius=radius)


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


def test_zero_relative_velocity_has_no_encounter_plane() -> None:
    primary = object_state(LEO_R, LEO_V, np.eye(3))
    secondary = object_state(LEO_R + np.array([10.0, 0.0, 0.0]), LEO_V, np.eye(3))
    conjunction = Conjunction(primary, secondary, 5.0)
    assert conjunction.relative_speed == 0.0
    with pytest.raises(GeometryError, match="relative velocity is zero"):
        _ = conjunction.plane


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


def test_diagnostics_label_each_objects_covariance_repairs() -> None:
    base = head_on(miss=30.0, sigma_x=10.0, sigma_z=20.0, radius=5.0)
    assert base.diagnostics == ()
    negative = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, -1.0])
    secondary = ObjectState(
        "99", base.secondary.state, make_covariance(negative, "GCRF", EPOCH)
    )
    conjunction = Conjunction(base.primary, secondary, 5.0)
    assert conjunction.diagnostics == (
        "secondary (99): clipped 1 negative eigenvalue(s) to zero (most negative -1)",
    )


def test_unknown_covariance_is_reported() -> None:
    base = head_on(30.0, 10.0, 20.0, 5.0)
    secondary = ObjectState("99999", base.secondary.state, None)
    conjunction = Conjunction(base.primary, secondary, 5.0)
    assert conjunction.diagnostics == ("secondary (99999): covariance unknown",)
