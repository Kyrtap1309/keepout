"""Reference-frame transformations.

* RTN (radial, transverse, normal) is built from an inertial state:
  ``R = r/|r|``, ``N = (r x v)/|r x v|``, ``T = N x R``.
* ITRF is converted to GCRF with astropy (ITRS -> GCRS), which includes the
  Earth-rotation term of the velocity.
* EME2000 is rotated to GCRF by the IAU 2006 frame bias (about 23 mas, ~0.8 m
  at 7000 km), so objects given in different frames stay consistent.
"""

from __future__ import annotations

from collections.abc import Sequence

import astropy.units as u
import erfa
import numpy as np
import numpy.typing as npt
from astropy.time import Time

from keepout._types import FloatArray, StateVector
from keepout.errors import GeometryError

_J2000_JD = 2451545.0
# GCRS -> mean J2000 (EME2000) frame bias; constant, so any date gives it.
FRAME_BIAS: FloatArray = np.asarray(erfa.bp06(_J2000_JD, 0.0)[0], dtype=np.float64)
"""Rotation from GCRF to EME2000 (``x_eme2000 = FRAME_BIAS @ x_gcrf``)."""


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


def _rotate_covariance(covariance: npt.ArrayLike, rotation: FloatArray) -> FloatArray:
    """``M C M^T`` with ``M`` applying ``rotation`` to position and velocity."""
    six = np.zeros((6, 6))
    six[:3, :3] = six[3:, 3:] = rotation
    return six @ np.asarray(covariance, dtype=np.float64) @ six.T


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
    return _rotate_covariance(covariance, rtn_rotation(r, v).T)


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
    return _rotate_covariance(covariance, rtn_rotation(r, v))


def _itrf_to_gcrf(states: Sequence[StateVector]) -> list[StateVector]:
    """Convert ITRF states with one astropy transformation for all of them."""
    # Imported here: astropy.coordinates is slow to import and only ITRF needs it.
    from astropy.coordinates import (
        GCRS,
        ITRS,
        CartesianDifferential,
        CartesianRepresentation,
    )

    epochs = Time([state.epoch for state in states])
    representation = CartesianRepresentation(
        np.stack([state.r for state in states], axis=1) * u.m,
        differentials=CartesianDifferential(
            np.stack([state.v for state in states], axis=1) * u.m / u.s
        ),
    )
    cartesian = (
        ITRS(representation, obstime=epochs)
        .transform_to(GCRS(obstime=epochs))
        .cartesian
    )
    positions = cartesian.xyz.to_value(u.m)
    velocities = cartesian.differentials["s"].d_xyz.to_value(u.m / u.s)
    return [
        StateVector(state.epoch, positions[:, i], velocities[:, i], "GCRF")
        for i, state in enumerate(states)
    ]


def to_gcrf_all(states: Sequence[StateVector]) -> list[StateVector]:
    """Express several state vectors in GCRF, converting ITRF ones together.

    Parameters
    ----------
    states
        States in GCRF, EME2000 or ITRF.

    Returns
    -------
    list of StateVector
        GCRF states in the same order; GCRF inputs are returned as they are.
    """
    converted: list[StateVector] = []
    for state in states:
        if state.frame == "EME2000":
            state = StateVector(
                state.epoch, FRAME_BIAS.T @ state.r, FRAME_BIAS.T @ state.v, "GCRF"
            )
        converted.append(state)
    itrf = [i for i, state in enumerate(converted) if state.frame == "ITRF"]
    if itrf:
        for i, state in zip(
            itrf, _itrf_to_gcrf([converted[i] for i in itrf]), strict=True
        ):
            converted[i] = state
    return converted


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
    return to_gcrf_all([state])[0]
