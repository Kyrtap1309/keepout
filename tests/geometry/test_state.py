import numpy as np
import pytest
from builders import EPOCH

from keepout.state import StateVector


def test_state_vector_copies_and_freezes_arrays() -> None:
    r = np.array([7000e3, 0.0, 0.0])
    state = StateVector(EPOCH, r, [0.0, 7500.0, 0.0], "GCRF")
    r[0] = 0.0
    assert state.r[0] == 7000e3
    assert state.v.dtype == np.float64
    with pytest.raises(ValueError, match="read-only"):
        state.r[0] = 1.0


@pytest.mark.parametrize(
    ("r", "message"),
    [([1.0, 2.0], r"shape \(3,\)"), ([1.0, float("nan"), 0.0], "finite")],
)
def test_state_vector_rejects_bad_position(r: list[float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        StateVector(EPOCH, r, [0.0, 0.0, 0.0], "GCRF")


def test_state_vector_rejects_unknown_frame() -> None:
    with pytest.raises(ValueError, match="unsupported frame 'TEME'"):
        # The wrong literal is the point of the test.
        StateVector(EPOCH, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], "TEME")  # type: ignore[arg-type]
