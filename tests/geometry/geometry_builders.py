"""Builders and Hypothesis strategies for geometry tests."""

import numpy as np
import numpy.typing as npt
from astropy.time import Time
from hypothesis import assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from keepout._types import FloatArray, StateVector
from keepout.covariance import Covariance
from keepout.objects import ObjectState

EPOCH = Time("2010-03-13T22:37:52.618", scale="utc")


def object_state(
    r: npt.ArrayLike,
    v: npt.ArrayLike,
    position_covariance: npt.ArrayLike | None = None,
    *,
    epoch: Time = EPOCH,
    object_id: str = "TEST",
) -> ObjectState:
    """A GCRF object state with the given position covariance [m^2]."""
    matrix = np.zeros((6, 6))
    matrix[:3, :3] = np.eye(3) if position_covariance is None else position_covariance
    matrix[3:, 3:] = 1e-4 * np.eye(3)
    return ObjectState(
        object_id,
        StateVector(epoch, r, v, "GCRF"),
        Covariance(matrix, "GCRF", epoch),
    )


@st.composite
def rotations(draw: st.DrawFn) -> FloatArray:
    """Uniform-ish proper rotation matrices from random quaternions."""
    quaternion = draw(arrays(np.float64, (4,), elements=st.floats(-1.0, 1.0)))
    norm = float(np.linalg.norm(quaternion))
    assume(norm > 0.1)
    w, x, y, z = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


@st.composite
def position_covariances(draw: st.DrawFn) -> FloatArray:
    """Random symmetric positive-definite 3x3 matrices [m^2]."""
    root: FloatArray = draw(
        arrays(np.float64, (3, 3), elements=st.floats(-100.0, 100.0))
    )
    return root @ root.T + np.eye(3)
