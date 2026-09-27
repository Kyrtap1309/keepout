"""Position-velocity covariance: validation, repair and frame changes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias, get_args

import numpy as np
import numpy.typing as npt
from astropy.time import Time

from keepout._types import FloatArray, StateVector
from keepout.errors import CovarianceError, GeometryError
from keepout.frames import inertial_to_rtn_covariance, rtn_to_inertial_covariance

CovarianceFrame: TypeAlias = Literal["RTN", "GCRF"]
"""Frames a :class:`Covariance` can be expressed in."""

COVARIANCE_FRAMES: tuple[CovarianceFrame, ...] = get_args(CovarianceFrame)

Policy: TypeAlias = Literal["raise", "repair"]
"""What to do with an invalid covariance: raise, or repair with a diagnostic."""

POLICIES: tuple[Policy, ...] = get_args(Policy)

SYMMETRY_RTOL = 1e-9
"""Allowed asymmetry, relative to the largest absolute entry."""

PSD_RTOL = 1e-10
"""Allowed negative eigenvalue, relative to the largest absolute entry."""


def _as_matrix(matrix: npt.ArrayLike) -> FloatArray:
    array = np.array(matrix, dtype=np.float64)
    if array.shape != (6, 6):
        raise CovarianceError(f"covariance must be 6x6, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise CovarianceError("covariance contains non-finite values")
    return array


def _inspect(array: FloatArray) -> tuple[float, float, FloatArray, FloatArray]:
    """Scale, asymmetry, and eigen-decomposition of the symmetric part."""
    scale = float(np.max(np.abs(array)))
    asymmetry = float(np.max(np.abs(array - array.T)))
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (array + array.T))
    return scale, asymmetry, eigenvalues, eigenvectors


def _problems(array: FloatArray) -> tuple[str, ...]:
    scale, asymmetry, eigenvalues, _ = _inspect(array)
    found: list[str] = []
    if asymmetry > SYMMETRY_RTOL * scale:
        found.append(f"not symmetric (max asymmetry {asymmetry:.3g})")
    if eigenvalues[0] < -PSD_RTOL * scale:
        found.append(
            f"not positive semi-definite (min eigenvalue {eigenvalues[0]:.3g})"
        )
    return tuple(found)


def problems(matrix: npt.ArrayLike) -> tuple[str, ...]:
    """Describe why a matrix is not a valid covariance.

    Parameters
    ----------
    matrix
        Candidate 6x6 covariance.

    Returns
    -------
    tuple of str
        Empty if the matrix is symmetric and positive semi-definite within
        :data:`SYMMETRY_RTOL` and :data:`PSD_RTOL`.

    Raises
    ------
    CovarianceError
        If the matrix is not 6x6 or contains non-finite values.
    """
    return _problems(_as_matrix(matrix))


def repair(matrix: npt.ArrayLike) -> tuple[FloatArray, tuple[str, ...]]:
    """Symmetrise a matrix and clip its negative eigenvalues to zero.

    Parameters
    ----------
    matrix
        6x6 matrix to repair.

    Returns
    -------
    tuple
        The repaired matrix and diagnostics describing what was changed (empty
        if the matrix was already valid).
    """
    scale, asymmetry, eigenvalues, eigenvectors = _inspect(_as_matrix(matrix))
    diagnostics: list[str] = []
    if asymmetry > SYMMETRY_RTOL * scale:
        diagnostics.append(f"symmetrised covariance (max asymmetry {asymmetry:.3g})")
    negative = eigenvalues < -PSD_RTOL * scale
    if np.any(negative):
        diagnostics.append(
            f"clipped {int(np.count_nonzero(negative))} negative eigenvalue(s) "
            f"to zero (most negative {float(eigenvalues[0]):.3g})"
        )
    fixed = (eigenvectors * np.clip(eigenvalues, 0.0, None)) @ eigenvectors.T
    return 0.5 * (fixed + fixed.T), tuple(diagnostics)


def _require_gcrf(state: StateVector) -> None:
    if state.frame != "GCRF":
        raise GeometryError(
            f"RTN axes must be built from a GCRF state, got {state.frame}; "
            "convert with keepout.frames.to_gcrf"
        )


def _frozen(array: FloatArray) -> FloatArray:
    symmetric = 0.5 * (array + array.T)
    symmetric.flags.writeable = False
    return symmetric


@dataclass(frozen=True, eq=False)
class Covariance:
    """A validated 6x6 position-velocity covariance in SI units (m, m/s).

    Parameters
    ----------
    matrix
        6x6 covariance; stored as a read-only symmetric copy.
    frame
        ``"RTN"`` or ``"GCRF"``.
    epoch
        Epoch the covariance refers to.
    repairs
        What :func:`make_covariance` changed to make the matrix valid; empty
        when it was valid as given. Kept through frame changes.

    Raises
    ------
    CovarianceError
        If the matrix is not 6x6, not finite, not symmetric or not positive
        semi-definite within tolerance.
    """

    matrix: FloatArray
    frame: CovarianceFrame
    epoch: Time
    repairs: tuple[str, ...] = field(default=(), kw_only=True)

    def __post_init__(self) -> None:
        if self.frame not in COVARIANCE_FRAMES:
            raise ValueError(f"unsupported covariance frame {self.frame!r}")
        array = _as_matrix(self.matrix)
        found = _problems(array)
        if found:
            raise CovarianceError("invalid covariance: " + "; ".join(found))
        object.__setattr__(self, "matrix", _frozen(array))
        object.__setattr__(self, "repairs", tuple(self.repairs))

    @classmethod
    def _trusted(
        cls,
        matrix: FloatArray,
        frame: CovarianceFrame,
        epoch: Time,
        repairs: tuple[str, ...] = (),
    ) -> Covariance:
        """Wrap a matrix already known to be valid, skipping the checks."""
        covariance = object.__new__(cls)
        for name, value in (
            ("matrix", _frozen(matrix)),
            ("frame", frame),
            ("epoch", epoch),
            ("repairs", repairs),
        ):
            object.__setattr__(covariance, name, value)
        return covariance

    @classmethod
    def from_position(
        cls, position: npt.ArrayLike, frame: CovarianceFrame, epoch: Time
    ) -> Covariance:
        """A covariance known only in position; the velocity block is zero.

        Parameters
        ----------
        position
            3x3 position covariance [m^2].
        frame
            ``"RTN"`` or ``"GCRF"``.
        epoch
            Epoch the covariance refers to.

        Raises
        ------
        CovarianceError
            If ``position`` is not 3x3 or not a valid covariance.
        """
        block = np.array(position, dtype=np.float64)
        if block.shape != (3, 3):
            raise CovarianceError(f"position covariance must be 3x3, got {block.shape}")
        matrix = np.zeros((6, 6))
        matrix[:3, :3] = block
        return cls(matrix, frame, epoch)

    @property
    def position(self) -> FloatArray:
        """The 3x3 position block [m^2]."""
        return self.matrix[:3, :3]

    def in_gcrf(self, state: StateVector) -> Covariance:
        """Return this covariance in GCRF.

        Parameters
        ----------
        state
            GCRF state of the object; defines the RTN axes.
        """
        _require_gcrf(state)
        if self.frame == "GCRF":
            return self
        matrix = rtn_to_inertial_covariance(self.matrix, state.r, state.v)
        return Covariance._trusted(matrix, "GCRF", self.epoch, self.repairs)

    def in_rtn(self, state: StateVector) -> Covariance:
        """Return this covariance in the object's RTN frame.

        Parameters
        ----------
        state
            GCRF state of the object; defines the RTN axes.
        """
        _require_gcrf(state)
        if self.frame == "RTN":
            return self
        matrix = inertial_to_rtn_covariance(self.matrix, state.r, state.v)
        return Covariance._trusted(matrix, "RTN", self.epoch, self.repairs)


def make_covariance(
    matrix: npt.ArrayLike,
    frame: CovarianceFrame,
    epoch: Time,
    *,
    policy: Policy = "repair",
) -> Covariance:
    """Build a :class:`Covariance`, repairing it if the policy allows.

    Parameters
    ----------
    matrix
        6x6 covariance [m, m/s].
    frame
        ``"RTN"`` or ``"GCRF"``.
    epoch
        Epoch the covariance refers to.
    policy
        ``"raise"`` raises :class:`~keepout.errors.CovarianceError` for an
        invalid matrix; ``"repair"`` (default, as in operational practice)
        symmetrises it and clips negative eigenvalues, recording what it did
        in :attr:`Covariance.repairs`.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown covariance policy {policy!r}")
    if frame not in COVARIANCE_FRAMES:
        raise ValueError(f"unsupported covariance frame {frame!r}")
    array = _as_matrix(matrix)
    if policy == "raise":
        return Covariance(array, frame, epoch)
    fixed, repairs = repair(array)
    return Covariance._trusted(fixed if repairs else array, frame, epoch, repairs)
