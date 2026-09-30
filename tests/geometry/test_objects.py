import astropy.units as u
import numpy as np
import pytest
from builders import EPOCH, object_state

from keepout._types import StateVector
from keepout.covariance import Covariance
from keepout.errors import GeometryError
from keepout.objects import ObjectState

GCRF_STATE = StateVector(EPOCH, [7000e3, 0.0, 0.0], [0.0, 7500.0, 0.0], "GCRF")
GCRF_COVARIANCE = Covariance(np.eye(6), "GCRF", EPOCH)


def test_object_state_holds_its_parts() -> None:
    state = ObjectState("12345", GCRF_STATE, GCRF_COVARIANCE, 5.0)
    assert state.object_id == "12345"
    assert state.hard_body_radius == 5.0
    assert object_state([7000e3, 0.0, 0.0], [0.0, 7500.0, 0.0]).hard_body_radius is None


def test_object_state_needs_a_gcrf_state() -> None:
    state = StateVector(EPOCH, GCRF_STATE.r, GCRF_STATE.v, "EME2000")
    with pytest.raises(GeometryError, match="GCRF"):
        ObjectState("A", state, GCRF_COVARIANCE)


def test_object_state_needs_a_gcrf_covariance() -> None:
    with pytest.raises(GeometryError, match="GCRF"):
        ObjectState("A", GCRF_STATE, Covariance(np.eye(6), "RTN", EPOCH))


def test_object_state_needs_matching_epochs() -> None:
    covariance = Covariance(np.eye(6), "GCRF", EPOCH + 10 * u.s)
    with pytest.raises(GeometryError, match="epochs differ by 10 s"):
        ObjectState("A", GCRF_STATE, covariance)


@pytest.mark.parametrize("radius", [-1.0, float("nan"), float("inf")])
def test_object_state_rejects_bad_hard_body_radius(radius: float) -> None:
    with pytest.raises(ValueError, match="hard_body_radius"):
        ObjectState("A", GCRF_STATE, GCRF_COVARIANCE, radius)


def test_covariance_may_be_unknown() -> None:
    state = ObjectState("A", GCRF_STATE, None, 5.0)
    assert state.covariance is None
