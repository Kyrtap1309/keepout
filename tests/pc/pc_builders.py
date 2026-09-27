"""Builders and Hypothesis strategies for Pc tests."""

import numpy as np
import numpy.typing as npt
from astropy.time import Time
from hypothesis import assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.spatial.transform import Rotation

from keepout._types import FloatArray, StateVector
from keepout.conjunction import Conjunction
from keepout.covariance import Covariance
from keepout.objects import ObjectState

EPOCH = Time("2010-03-13T22:37:52.618", scale="utc")
LEO_R = np.array([7000e3, 0.0, 0.0])
LEO_V = np.array([0.0, 7546.0, 0.0])


def object_state(
    r: npt.ArrayLike,
    v: npt.ArrayLike,
    position_covariance: npt.ArrayLike,
    object_id: str = "TEST",
) -> ObjectState:
    """A GCRF object state with the given 3x3 position covariance [m^2]."""
    matrix = np.zeros((6, 6))
    matrix[:3, :3] = position_covariance
    matrix[3:, 3:] = 1e-4 * np.eye(3)
    return ObjectState(
        object_id, StateVector(EPOCH, r, v, "GCRF"), Covariance(matrix, "GCRF", EPOCH)
    )


def head_on(miss: float, sigma_x: float, sigma_z: float, radius: float) -> Conjunction:
    """LEO head-on encounter with the miss along GCRF x.

    The secondary is ``miss`` metres radially above the primary and flies
    against it; the plane axes are GCRF x and -y-ish, the plane covariance is
    ``diag(sigma_x^2, sigma_z^2)`` (all of it on the primary).
    """
    primary = object_state(LEO_R, LEO_V, np.diag([sigma_x**2, 1.0, sigma_z**2]))
    secondary = object_state(
        LEO_R + np.array([miss, 0.0, 0.0]),
        -LEO_V,
        np.diag([1e-12, 1e-12, 1e-12]),
        "SEC",
    )
    return Conjunction(primary, secondary, radius)


def rotate(obj: ObjectState, rotation: FloatArray) -> ObjectState:
    """The same object with state and covariance rotated by ``rotation``."""
    block = np.zeros((6, 6))
    block[:3, :3] = block[3:, 3:] = rotation
    state = obj.state
    return ObjectState(
        obj.object_id,
        StateVector(state.epoch, rotation @ state.r, rotation @ state.v, "GCRF"),
        Covariance(block @ obj.covariance.matrix @ block.T, "GCRF", state.epoch),
        obj.hard_body_radius,
    )


@st.composite
def rotations(draw: st.DrawFn) -> FloatArray:
    """Proper rotation matrices from random quaternions."""
    quaternion = draw(arrays(np.float64, (4,), elements=st.floats(-1.0, 1.0)))
    assume(float(np.linalg.norm(quaternion)) > 0.1)
    matrix: FloatArray = Rotation.from_quat(quaternion, scalar_first=True).as_matrix()
    return matrix


@st.composite
def position_covariances(draw: st.DrawFn) -> FloatArray:
    """Random positive-definite 3x3 covariances, sigmas roughly 1 m to 1 km."""
    root: FloatArray = draw(
        arrays(np.float64, (3, 3), elements=st.floats(-300.0, 300.0))
    )
    return root @ root.T + np.eye(3)


@st.composite
def conjunctions(draw: st.DrawFn) -> Conjunction:
    """Random LEO conjunctions at TCA: miss up to 2 km, 50 m/s to 15 km/s."""
    direction = draw(arrays(np.float64, (3,), elements=st.floats(-1.0, 1.0)))
    assume(float(np.linalg.norm(direction)) > 0.1)
    speed = draw(st.floats(50.0, 15000.0))
    relative_velocity = speed * direction / np.linalg.norm(direction)
    offset = draw(arrays(np.float64, (3,), elements=st.floats(-2000.0, 2000.0)))
    # Put the secondary exactly at closest approach: no offset along v.
    unit = relative_velocity / speed
    offset = offset - float(offset @ unit) * unit
    primary = object_state(LEO_R, LEO_V, draw(position_covariances()), "PRI")
    secondary = object_state(
        LEO_R + offset, LEO_V + relative_velocity, draw(position_covariances()), "SEC"
    )
    radius = draw(st.floats(1.0, 50.0))
    return Conjunction(primary, secondary, radius)
