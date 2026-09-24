"""Reference-frame transformations.

* RTN (radial, transverse, normal) is built from an inertial state:
  ``R = r/|r|``, ``N = (r x v)/|r x v|``, ``T = N x R``.
* ITRF is converted to GCRF with astropy (ITRS -> GCRS), which includes the
  Earth-rotation term of the velocity.
* EME2000 is treated as GCRF: the two differ by the ~20 mas frame bias, a
  rotation common to both objects of a conjunction, which does not change Pc.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
import numpy.typing as npt
from astropy.coordinates import (
    GCRS,
    ITRS,
    CartesianDifferential,
    CartesianRepresentation,
)
from scipy.linalg import block_diag

from keepout._types import FloatArray, StateVector
from keepout.errors import GeometryError


def rtn_rotation(r: npt.ArrayLike, v: npt.ArrayLike) -> FloatArray:
    """Rotation matrix from an inertial frame to RTN.

    Parameters
    ----------
    r
        Inertial position [m].
    v
        Inertial velocity [m/s].

    Returns
    -------
    FloatArray
        3x3 matrix whose rows are the R, T and N unit vectors, so that
        ``x_rtn = M @ x_inertial``.

    Raises
    ------
    GeometryError
        If ``r`` is zero or parallel to ``v``.
    """
    position = np.asarray(r, dtype=np.float64)
    velocity = np.asarray(v, dtype=np.float64)
    angular_momentum = np.cross(position, velocity)
    r_norm = float(np.linalg.norm(position))
    h_norm = float(np.linalg.norm(angular_momentum))
    if r_norm == 0.0 or h_norm <= 1e-12 * r_norm * float(np.linalg.norm(velocity)):
        raise GeometryError(
            "RTN frame is undefined: position is zero or parallel to velocity"
        )
    radial = position / r_norm
    normal = angular_momentum / h_norm
    transverse = np.cross(normal, radial)
    return np.stack([radial, transverse, normal])


def _six_by_six(rotation: FloatArray) -> FloatArray:
    return np.asarray(block_diag(rotation, rotation), dtype=np.float64)


def rtn_to_inertial_covariance(
    covariance: npt.ArrayLike, r: npt.ArrayLike, v: npt.ArrayLike
) -> FloatArray:
    """Rotate a 6x6 position-velocity covariance from RTN to the inertial frame.

    Position and velocity blocks are rotated with the same matrix; the rotation
    rate of the RTN frame is neglected (standard conjunction-assessment
    practice; it affects only velocity terms, not Pc).

    Parameters
    ----------
    covariance
        6x6 covariance in RTN [m, m/s].
    r
        Inertial position defining the RTN axes [m].
    v
        Inertial velocity defining the RTN axes [m/s].
    """
    rotation = _six_by_six(rtn_rotation(r, v))
    return rotation.T @ np.asarray(covariance, dtype=np.float64) @ rotation


def inertial_to_rtn_covariance(
    covariance: npt.ArrayLike, r: npt.ArrayLike, v: npt.ArrayLike
) -> FloatArray:
    """Rotate a 6x6 position-velocity covariance from the inertial frame to RTN.

    Parameters
    ----------
    covariance
        6x6 inertial covariance [m, m/s].
    r
        Inertial position defining the RTN axes [m].
    v
        Inertial velocity defining the RTN axes [m/s].
    """
    rotation = _six_by_six(rtn_rotation(r, v))
    return rotation @ np.asarray(covariance, dtype=np.float64) @ rotation.T


def to_gcrf(state: StateVector) -> StateVector:
    """Express a state vector in GCRF.

    Parameters
    ----------
    state
        State in GCRF, EME2000 or ITRF.

    Returns
    -------
    StateVector
        ``state`` itself if already GCRF; otherwise a new GCRF state.
    """
    if state.frame == "GCRF":
        return state
    if state.frame == "EME2000":
        return StateVector(state.epoch, state.r, state.v, "GCRF")
    representation = CartesianRepresentation(
        state.r * u.m, differentials=CartesianDifferential(state.v * u.m / u.s)
    )
    gcrs = ITRS(representation, obstime=state.epoch).transform_to(
        GCRS(obstime=state.epoch)
    )
    cartesian = gcrs.cartesian
    return StateVector(
        state.epoch,
        cartesian.xyz.to_value(u.m),
        cartesian.differentials["s"].d_xyz.to_value(u.m / u.s),
        "GCRF",
    )
