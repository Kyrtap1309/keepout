"""Reference-frame transformations.

* RTN (radial, transverse, normal) is built from an inertial state:
  ``R = r/|r|``, ``N = (r x v)/|r x v|``, ``T = N x R``.
* ITRF is converted to GCRF with astropy (ITRS -> GCRS), which includes the
  Earth-rotation term of the velocity.
* EME2000 is rotated to GCRF by the IAU 2006 frame bias (about 23 mas, ~0.8 m
  at 7000 km), so objects given in different frames stay consistent.
* TEME, the frame of SGP4 output, is rotated to GCRF with ERFA (see
  :func:`teme_to_gcrf`); no Earth-orientation data is needed.

:func:`to_gcrf_all` converts state vectors in any of these frames, and
:func:`frame_to_gcrf` arrays of states in one frame (as a propagator gives).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import astropy.units as u
import erfa
import numpy as np
import numpy.typing as npt
from astropy.time import Time

from keepout._types import FloatArray, read_only
from keepout.errors import GeometryError
from keepout.state import RefFrame, StateVector, as_ref_frame
from keepout.units import DAY, MINUTE

__all__ = [
    "FRAME_BIAS",
    "PARALLEL_RTOL",
    "TEME_RATE_STEP",
    "frame_to_gcrf",
    "inertial_to_rtn_covariance",
    "relative_position_rtn",
    "rtn_rotation",
    "rtn_to_inertial_covariance",
    "teme_to_gcrf",
    "to_gcrf",
    "to_gcrf_all",
]

_J2000_JD: Final = 2451545.0
# GCRS -> mean J2000 (EME2000) frame bias; constant, so any date gives it.
FRAME_BIAS: Final[FloatArray] = np.array(erfa.bp06(_J2000_JD, 0.0)[0], dtype=np.float64)
"""Rotation from GCRF to EME2000 (``x_eme2000 = FRAME_BIAS @ x_gcrf``)."""
read_only(FRAME_BIAS)

PARALLEL_RTOL: Final = 1e-12
"""Relative size below which a vector's component perpendicular to another
is taken as zero: position and velocity are then parallel."""

TEME_RATE_STEP: Final = 10 * MINUTE
"""Half-width [s] of the central difference that gives the rate of the TEME
to GCRF rotation: long enough for rounding in GMST and ERA to stay small,
short against nutation (error below 1e-8 m/s at GEO)."""


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
    return _rtn_basis(np.asarray(r, dtype=np.float64), np.asarray(v, dtype=np.float64))


def _rtn_basis(position: FloatArray, velocity: FloatArray) -> FloatArray:
    """The RTN rotations of states, shape ``(..., 3, 3)``: rows R, T, N.

    A state with a non-finite component gives a non-finite rotation (NaN
    compares false), so rows of failed states pass through.

    Raises
    ------
    GeometryError
        If a finite position is zero or parallel to its velocity.
    """
    angular_momentum = np.cross(position, velocity)
    r_norm = np.linalg.norm(position, axis=-1, keepdims=True)
    h_norm = np.linalg.norm(angular_momentum, axis=-1, keepdims=True)
    speed = np.linalg.norm(velocity, axis=-1, keepdims=True)
    if np.any((r_norm == 0.0) | (h_norm <= PARALLEL_RTOL * r_norm * speed)):
        raise GeometryError(
            "RTN frame is undefined: position is zero or parallel to velocity"
        )
    radial = position / r_norm
    normal = angular_momentum / h_norm
    transverse = np.cross(normal, radial)
    basis: FloatArray = np.stack(
        np.broadcast_arrays(radial, transverse, normal), axis=-2
    )
    return basis


def relative_position_rtn(
    r: npt.ArrayLike, v: npt.ArrayLike, r_secondary: npt.ArrayLike
) -> FloatArray:
    """Positions of secondaries relative to a primary, in the primary's RTN.

    The vectorised form of ``rtn_rotation(r, v) @ (r_secondary - r)`` (same
    axes as :func:`rtn_rotation`), for many pairs and times at once: e.g.
    where a secondary is in the frame of the primary of a conjunction.

    Parameters
    ----------
    r
        Inertial positions of the primary [m], shape ``(..., 3)``.
    v
        Inertial velocities of the primary [m/s], shape ``(..., 3)``.
    r_secondary
        Inertial positions of the secondaries [m], shape ``(..., 3)``, in
        the same frame.

    Returns
    -------
    FloatArray
        Radial, transverse and normal components [m], shape ``(..., 3)``,
        where ``...`` is the broadcast of the three inputs' leading shapes.
        A row with a non-finite input is non-finite.

    Raises
    ------
    ValueError
        If an input does not have 3 components on its last axis, or the
        shapes do not broadcast.
    GeometryError
        If a finite position is zero or parallel to its velocity.
    """
    position = _components(r, "r")
    velocity = _components(v, "v")
    secondary = _components(r_secondary, "r_secondary")
    try:
        np.broadcast_shapes(position.shape, velocity.shape, secondary.shape)
    except ValueError:
        raise ValueError(
            f"r {position.shape}, v {velocity.shape} and r_secondary "
            f"{secondary.shape} do not broadcast"
        ) from None
    result: FloatArray = np.einsum(
        "...ij,...j->...i", _rtn_basis(position, velocity), secondary - position
    )
    return result


def _rotate_covariance(covariance: npt.ArrayLike, rotation: FloatArray) -> FloatArray:
    """``M C M^T`` with ``M`` applying ``rotation`` to position and velocity."""
    six = np.zeros((6, 6))
    six[:3, :3] = six[3:, 3:] = rotation
    return six @ np.asarray(covariance, dtype=np.float64) @ six.T


def rtn_to_inertial_covariance(
    covariance: npt.ArrayLike, r: npt.ArrayLike, v: npt.ArrayLike
) -> FloatArray:
    """A 6x6 position-velocity covariance rotated from RTN to the inertial frame.

    Position and velocity blocks are rotated with the same matrix; the rotation
    rate of the RTN frame is neglected (standard conjunction-assessment
    practice). That changes only the velocity terms: 2-D Pc, which uses the
    position block alone, is unaffected, while 3-D and Monte Carlo Pc use the
    velocity block and are validated against CARA with this convention.

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
    """A 6x6 position-velocity covariance rotated from the inertial frame to RTN.

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
    """ITRF states in GCRF, by one astropy transformation for all of them."""
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
    velocity = cartesian.differentials["s"]
    positions = cartesian.xyz.to_value(u.m)
    velocities = velocity.d_xyz.to_value(u.m / u.s)
    return [
        StateVector(state.epoch, positions[:, i], velocities[:, i], "GCRF")
        for i, state in enumerate(states)
    ]


def _teme_rotation(
    tt1: FloatArray, tt2: FloatArray, ut1: FloatArray, ut2: FloatArray
) -> FloatArray:
    """GCRF <- TEME rotation matrices, shape ``(..., 3, 3)``.

    ``r_gcrf = C^T R3(GMST82 - ERA) r_teme``, with ``C`` the IAU 2006/2000A
    GCRS -> CIRS matrix (TT) and ``R3`` ERFA's rotation of the axes about z.
    """
    theta = erfa.gmst82(ut1, ut2) - erfa.era00(ut1, ut2)
    cosine, sine = np.cos(theta), np.sin(theta)
    about_z = np.zeros((*np.shape(theta), 3, 3))
    about_z[..., 0, 0] = about_z[..., 1, 1] = cosine
    about_z[..., 0, 1] = sine
    about_z[..., 1, 0] = -sine
    about_z[..., 2, 2] = 1.0
    gcrs_to_cirs = erfa.c2i06a(tt1, tt2)
    rotation: FloatArray = np.swapaxes(gcrs_to_cirs, -1, -2) @ about_z
    return rotation


def _components(value: npt.ArrayLike, name: str) -> FloatArray:
    """``value`` as a float64 array whose last axis has length 3.

    Raises
    ------
    ValueError
        If ``value`` is a scalar or its last axis does not have length 3.
    """
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have 3 components on its last axis")
    return array


def _broadcast(
    epochs: Time, position: FloatArray, velocity: FloatArray
) -> tuple[int, ...]:
    """The broadcast shape of states at ``epochs``, without the 3 components.

    Raises
    ------
    ValueError
        If the shapes of ``epochs``, ``position`` and ``velocity`` (without
        its components) do not broadcast.
    """
    try:
        return np.broadcast_shapes(
            epochs.shape, position.shape[:-1], velocity.shape[:-1]
        )
    except ValueError:
        raise ValueError(
            f"epochs {epochs.shape}, r {position.shape} and v {velocity.shape} "
            "do not broadcast"
        ) from None


def teme_to_gcrf(
    epochs: Time, r: npt.ArrayLike, v: npt.ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """TEME positions and velocities in GCRF, many at once.

    TEME (true equator, mean equinox of date) is the frame of SGP4 output.
    Following Vallado et al. (2006), TEME is turned to the pseudo-Earth-fixed
    frame by GMST (IAU 1982) and then, through ITRF, to GCRF by the IAU
    2006/2000A model. Written as one rotation,
    ``r_gcrf = C^T R3(GMST82 - ERA) r_teme`` (``C``: GCRS -> CIRS matrix,
    ``ERA``: Earth rotation angle), polar motion cancels and UT1 enters only
    through ``GMST82 - ERA``, which changes by 7e-12 rad per second of
    UT1 - UTC: so UTC is used for UT1 (< 0.3 mm at GEO for |UT1 - UTC| <
    0.9 s) and no Earth-orientation data is read, for any epoch. The result
    agrees with astropy's TEME -> GCRS (which uses IERS data) to 1e-10 of
    the radius. The velocity is ``M v + (dM/dt) r``: the frame turns slowly
    (precession and nutation, up to 1.1e-11 rad/s, 5e-4 m/s at GEO), and its
    rate comes from a central difference over :data:`TEME_RATE_STEP`.

    Parameters
    ----------
    epochs
        Epoch of each state (UTC, TT, TAI, ...); a scalar for states at one
        epoch.
    r
        TEME positions [m], shape ``(..., 3)``.
    v
        TEME velocities [m/s], shape ``(..., 3)``.

    Returns
    -------
    tuple of FloatArray
        GCRF positions [m] and velocities [m/s], shape ``(..., 3)``, where
        ``...`` is the broadcast of ``epochs.shape``, ``r.shape[:-1]`` and
        ``v.shape[:-1]`` (e.g. ``(n,)`` epochs with ``(n, 3)`` states, or
        ``(t, 1)`` epochs with ``(n, 3)`` states for a ``(t, n, 3)`` grid).
        Non-finite inputs give non-finite outputs.

    Raises
    ------
    ValueError
        If ``r`` or ``v`` does not have 3 components on its last axis, or
        the shapes do not broadcast.

    References
    ----------
    D. A. Vallado, P. Crawford, R. Hujsak and T. S. Kelso, "Revisiting
    Spacetrack Report #3", AIAA 2006-6753, 2006.
    """
    position = _components(r, "r")
    velocity = _components(v, "v")
    _broadcast(epochs, position, velocity)
    tt, utc = epochs.tt, epochs.utc
    tt1, tt2 = np.asarray(tt.jd1), np.asarray(tt.jd2)
    ut1, ut2 = np.asarray(utc.jd1), np.asarray(utc.jd2)
    step = TEME_RATE_STEP / DAY
    rotation = _teme_rotation(tt1, tt2, ut1, ut2)
    later = _teme_rotation(tt1, tt2 + step, ut1, ut2 + step)
    earlier = _teme_rotation(tt1, tt2 - step, ut1, ut2 - step)
    rate = (later - earlier) / (2.0 * TEME_RATE_STEP)
    column_r, column_v = position[..., None], velocity[..., None]
    gcrf_r: FloatArray = (rotation @ column_r)[..., 0]
    gcrf_v: FloatArray = (rotation @ column_v + rate @ column_r)[..., 0]
    return gcrf_r, gcrf_v


def _teme_states_to_gcrf(states: Sequence[StateVector]) -> list[StateVector]:
    """TEME states in GCRF, by one :func:`teme_to_gcrf` call for all of them."""
    positions, velocities = teme_to_gcrf(
        Time([state.epoch for state in states]),
        np.stack([state.r for state in states]),
        np.stack([state.v for state in states]),
    )
    return [
        StateVector(state.epoch, positions[i], velocities[i], "GCRF")
        for i, state in enumerate(states)
    ]


def to_gcrf_all(states: Sequence[StateVector]) -> list[StateVector]:
    """Several state vectors in GCRF, ITRF and TEME ones converted together.

    Parameters
    ----------
    states
        States in GCRF, EME2000, ITRF or TEME.

    Returns
    -------
    list of StateVector
        GCRF states in the same order; GCRF inputs are returned as they are.
    """
    converted = [
        StateVector(state.epoch, FRAME_BIAS.T @ state.r, FRAME_BIAS.T @ state.v, "GCRF")
        if state.frame == "EME2000"
        else state
        for state in states
    ]
    # All states of a frame in one call: the conversions are vectorised.
    for frame, convert in (("ITRF", _itrf_to_gcrf), ("TEME", _teme_states_to_gcrf)):
        indices = [i for i, state in enumerate(converted) if state.frame == frame]
        if indices:
            for i, state in zip(
                indices, convert([converted[i] for i in indices]), strict=True
            ):
                converted[i] = state
    return converted


def frame_to_gcrf(
    frame: RefFrame, epochs: Time, r: npt.ArrayLike, v: npt.ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Positions and velocities in one frame, in GCRF, many at once.

    TEME states go through :func:`teme_to_gcrf` in one call; EME2000 states
    are rotated by :data:`FRAME_BIAS` in one call; ITRF states go as
    :class:`~keepout.state.StateVector` through :func:`to_gcrf_all`; GCRF
    states are copied.

    Parameters
    ----------
    frame
        Frame of ``r`` and ``v``.
    epochs
        Epoch of each state; a scalar for states at one epoch.
    r
        Positions [m], shape ``(..., 3)``.
    v
        Velocities [m/s], shape ``(..., 3)``.

    Returns
    -------
    tuple of FloatArray
        GCRF positions [m] and velocities [m/s], shape ``(..., 3)``, where
        ``...`` is the broadcast of ``epochs.shape``, ``r.shape[:-1]`` and
        ``v.shape[:-1]``.

    Raises
    ------
    ValueError
        If ``r`` or ``v`` does not have 3 components on its last axis, the
        shapes do not broadcast, or a state in EME2000 or ITRF is not finite.
    UnsupportedFrame
        If ``frame`` is not a supported reference frame.
    """
    frame = as_ref_frame(frame)
    if frame == "TEME":
        return teme_to_gcrf(epochs, r, v)
    position = _components(r, "r")
    velocity = _components(v, "v")
    shape = _broadcast(epochs, position, velocity)
    position = np.broadcast_to(position, (*shape, 3))
    velocity = np.broadcast_to(velocity, (*shape, 3))
    if frame == "GCRF":
        return position.copy(), velocity.copy()
    if frame == "EME2000":
        return _eme2000_to_gcrf(position, "r"), _eme2000_to_gcrf(velocity, "v")
    # Time broadcasts through numpy's __array_function__, unknown to its stubs.
    times = np.broadcast_to(epochs, shape).ravel()  # type: ignore[call-overload]
    states = to_gcrf_all(
        [
            StateVector(times[i], r_i, v_i, frame)
            for i, (r_i, v_i) in enumerate(
                zip(position.reshape(-1, 3), velocity.reshape(-1, 3), strict=True)
            )
        ]
    )
    gcrf_r = np.array([state.r for state in states]).reshape(*shape, 3)
    gcrf_v = np.array([state.v for state in states]).reshape(*shape, 3)
    return gcrf_r, gcrf_v


def _eme2000_to_gcrf(vectors: FloatArray, name: str) -> FloatArray:
    """EME2000 vectors ``(..., 3)`` in GCRF, ``FRAME_BIAS.T @ x`` for each.

    Raises
    ------
    ValueError
        If a component is not finite.
    """
    if not np.all(np.isfinite(vectors)):
        raise ValueError(f"{name} must be finite")
    # The same sum, term by term, as FRAME_BIAS.T @ x on each vector.
    gcrf: FloatArray = np.einsum("ji,...j->...i", FRAME_BIAS, vectors)
    return gcrf


def to_gcrf(state: StateVector) -> StateVector:
    """A state vector in GCRF.

    Parameters
    ----------
    state
        State in GCRF, EME2000, ITRF or TEME.

    Returns
    -------
    StateVector
        ``state`` itself if already GCRF; otherwise a new GCRF state.
    """
    return to_gcrf_all([state])[0]
