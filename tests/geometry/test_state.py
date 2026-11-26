import numpy as np
import pytest
from builders import EPOCH

from keepout.errors import GeometryError, UnsupportedFrame
from keepout.state import REF_FRAMES, StateVector, as_ref_frame, relative_motion


def test_state_vector_copies_and_freezes_arrays() -> None:
    r = np.array([7000e3, 0.0, 0.0])
    state = StateVector(EPOCH, r, [0.0, 7500.0, 0.0], "GCRF")
    r[0] = 0.0
    assert state.r[0] == 7000e3
    assert state.v.dtype == np.float64
    with pytest.raises(ValueError, match="read-only"):
        state.r[0] = 1.0


GOOD = [1.0, 0.0, 0.0]


@pytest.mark.parametrize(
    ("r", "v", "message"),
    [
        ([1.0, 2.0], GOOD, r"r must have shape \(3,\)"),
        ([1.0, float("nan"), 0.0], GOOD, "r must be finite"),
        (GOOD, [1.0, 2.0], r"v must have shape \(3,\)"),
        (GOOD, [0.0, float("inf"), 0.0], "v must be finite"),
    ],
)
def test_state_vector_rejects_bad_vectors(
    r: list[float], v: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        StateVector(EPOCH, r, v, "GCRF")


def test_state_vector_rejects_unknown_frame() -> None:
    with pytest.raises(UnsupportedFrame, match="unsupported reference frame 'TOD'"):
        # The wrong literal is the point of the test.
        StateVector(EPOCH, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], "TOD")  # type: ignore[arg-type]


@pytest.mark.parametrize("frame", REF_FRAMES)
def test_as_ref_frame_accepts_supported_frames(frame: str) -> None:
    assert as_ref_frame(frame) == frame


@pytest.mark.parametrize("name", ["TOD", "teme", "gcrf", "", None])
def test_as_ref_frame_rejects_other_names(name: str | None) -> None:
    with pytest.raises(
        UnsupportedFrame, match=r"expected one of GCRF, EME2000, ITRF, TEME$"
    ):
        as_ref_frame(name)


def test_relative_motion_needs_states_in_one_frame() -> None:
    primary = StateVector(EPOCH, [7000e3, 0.0, 0.0], [0.0, 7500.0, 0.0], "GCRF")
    secondary = StateVector(EPOCH, [7000e3, 10.0, 0.0], [0.0, -7500.0, 0.0], "TEME")
    with pytest.raises(GeometryError, match="needs a GCRF state, got TEME"):
        relative_motion(primary, secondary)
