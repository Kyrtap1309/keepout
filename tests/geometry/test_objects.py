from collections.abc import Sequence

import astropy.units as u
import numpy as np
import pytest
from builders import DIAGONAL_COVARIANCE, EPOCH, known, object_state

from keepout import frames
from keepout.covariance import Covariance
from keepout.errors import CovarianceError, GeometryError
from keepout.objects import ObjectState, ReportedObject, gcrf_object_states
from keepout.state import StateVector

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


def test_object_state_accepts_zero_hard_body_radius() -> None:
    # A point object: only the combined radius in Conjunction must be > 0.
    state = ObjectState("A", GCRF_STATE, GCRF_COVARIANCE, 0.0)
    assert state.hard_body_radius == 0.0


def test_covariance_may_be_unknown() -> None:
    state = ObjectState("A", GCRF_STATE, None, 5.0)
    assert state.covariance is None


ITRF_STATE = StateVector(EPOCH, [7000e3, 0.0, 0.0], [0.0, 7000.0, 1000.0], "ITRF")


def test_reported_objects_become_gcrf_object_states() -> None:
    first, second = gcrf_object_states(
        [
            ReportedObject("A", ITRF_STATE, DIAGONAL_COVARIANCE, 5.0),
            ReportedObject("B", GCRF_STATE, None),
        ]
    )
    assert (first.object_id, second.object_id) == ("A", "B")
    assert (first.hard_body_radius, second.hard_body_radius) == (5.0, None)
    assert first.state.frame == "GCRF"
    assert second.state is GCRF_STATE
    assert second.covariance is None
    # The RTN axes are those of the GCRF state.
    np.testing.assert_allclose(
        known(first.covariance).in_rtn(first.state).matrix,
        DIAGONAL_COVARIANCE,
        atol=1e-9,
    )


def test_itrf_states_are_converted_together(monkeypatch: pytest.MonkeyPatch) -> None:
    batches: list[int] = []
    convert = frames._itrf_to_gcrf

    def counting(states: Sequence[StateVector]) -> list[StateVector]:
        batches.append(len(states))
        return convert(states)

    monkeypatch.setattr(frames, "_itrf_to_gcrf", counting)
    gcrf_object_states([ReportedObject(name, ITRF_STATE, None) for name in ("A", "B")])
    assert batches == [2]


def test_reported_covariance_follows_the_policy() -> None:
    invalid = DIAGONAL_COVARIANCE.copy()
    invalid[0, 1] = invalid[1, 0] = 1e6
    reported = [ReportedObject("A", GCRF_STATE, invalid)]
    with pytest.raises(CovarianceError, match="not positive semi-definite"):
        gcrf_object_states(reported, policy="raise")
    (repaired,) = gcrf_object_states(reported)
    assert known(repaired.covariance).repairs
