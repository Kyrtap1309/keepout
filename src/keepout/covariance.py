"""Position-velocity covariance: validation, repair and frame changes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt
from astropy.time import Time

from keepout._types import FloatArray, StateVector
from keepout.errors import CovarianceError, GeometryError
from keepout.frames import inertial_to_rtn_covariance, rtn_to_inertial_covariance

CovarianceFrame: TypeAlias = Literal["RTN", "GCRF"]
"""Frames a :class:`Covariance` can be expressed in."""

Policy: TypeAlias = Literal["raise", "repair"]
"""What to do with an invalid covariance: raise, or repair with a diagnostic."""

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


def _scale(matrix: FloatArray) -> float:
    return float(np.max(np.abs(matrix)))


def _asymmetry(matrix: FloatArray) -> float:
    return float(np.max(np.abs(matrix - matrix.T)))


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
    array = _as_matrix(matrix)
    scale = _scale(array)
    found: list[str] = []
    asymmetry = _asymmetry(array)
    if asymmetry > SYMMETRY_RTOL * scale:
        found.append(f"not symmetric (max asymmetry {asymmetry:.3g})")
    smallest = float(np.linalg.eigvalsh(0.5 * (array + array.T))[0])
    if smallest < -PSD_RTOL * scale:
        found.append(f"not positive semi-definite (min eigenvalue {smallest:.3g})")
    return tuple(found)


def repair(matrix: npt.ArrayLike) -> tuple[FloatArray, tuple[str, ...]]:
    """Symmetrise a matrix and clip its negative eigenvalues to zero.

    Parameters
    ----------
    matrix
        6x6 matrix to repair.

    Returns
    -------
    tuple
        The repaired matrix and diagnostics describing what was changed.
    """
    array = _as_matrix(matrix)
    scale = _scale(array)
    diagnostics: list[str] = []
    asymmetry = _asymmetry(array)
    if asymmetry > SYMMETRY_RTOL * scale:
        diagnostics.append(f"symmetrised covariance (max asymmetry {asymmetry:.3g})")
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (array + array.T))
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

    Raises
    ------
    CovarianceError
        If the matrix is not 6x6, not finite, not symmetric or not positive
        semi-definite within tolerance.
    """

    matrix: FloatArray
    frame: CovarianceFrame
    epoch: Time

    def __post_init__(self) -> None:
        if self.frame not in ("RTN", "GCRF"):
            raise ValueError(f"unsupported covariance frame {self.frame!r}")
        array = _as_matrix(self.matrix)
        found = problems(array)
        if found:
            raise CovarianceError("invalid covariance: " + "; ".join(found))
        array = 0.5 * (array + array.T)
        array.flags.writeable = False
        object.__setattr__(self, "matrix", array)

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
        return Covariance(matrix, "GCRF", self.epoch)

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
        return Covariance(matrix, "RTN", self.epoch)


def make_covariance(
    matrix: npt.ArrayLike,
    frame: CovarianceFrame,
    epoch: Time,
    *,
    policy: Policy = "repair",
) -> tuple[Covariance, tuple[str, ...]]:
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
        symmetrises it and clips negative eigenvalues.

    Returns
    -------
    tuple
        The covariance and diagnostics (empty when the matrix was valid).
    """
    if policy not in ("raise", "repair"):
        raise ValueError(f"unknown covariance policy {policy!r}")
    array = _as_matrix(matrix)
    if policy == "raise" or not problems(array):
        return Covariance(array, frame, epoch), ()
    fixed, diagnostics = repair(array)
    return Covariance(fixed, frame, epoch), diagnostics
