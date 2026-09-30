"""Position-velocity covariance: validation, repair and frame changes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias, get_args

import numpy as np
import numpy.typing as npt
from astropy.time import Time

from keepout._types import FloatArray, symmetric_part
from keepout.errors import CovarianceError, GeometryError
from keepout.frames import inertial_to_rtn_covariance, rtn_to_inertial_covariance
from keepout.state import StateVector

CovarianceFrame: TypeAlias = Literal["RTN", "GCRF"]
"""Frames a :class:`Covariance` can be expressed in."""

COVARIANCE_FRAMES: tuple[CovarianceFrame, ...] = get_args(CovarianceFrame)

Policy: TypeAlias = Literal["raise", "repair"]
"""What to do with an invalid covariance: raise, or repair with a diagnostic."""

POLICIES: tuple[Policy, ...] = get_args(Policy)

SYMMETRY_RTOL = 1e-9
"""Allowed asymmetry of ``C[i, j]``, relative to ``sqrt(|C[i, i] C[j, j]|)``."""

PSD_RTOL = 1e-10
"""Allowed negative eigenvalue of the correlation matrix ``D^-1 C D^-1``.

``D = diag(sqrt|C[i, i]|)`` (1 where a variance is 0 or subnormal), so each block is
judged at its own scale: a velocity variance of 1e-4 m^2/s^2 is checked as
strictly as a position variance of 1e8 m^2."""


RANK_RTOL = 1e-8
"""Position-block correlation eigenvalue below which :func:`repair` treats
the direction as unknown: velocity correlations along it are dropped rather
than divided by a vanishing variance."""

_SMALLEST_NORMAL = float(np.finfo(np.float64).tiny)


def _eigh(matrix: FloatArray) -> tuple[FloatArray, FloatArray]:
    """:func:`numpy.linalg.eigh` of a matrix with entries of any finite size.

    Eigenvalues scale with the matrix, so it is decomposed at unit scale;
    LAPACK could otherwise overflow on correlations far outside [-1, 1].
    """
    size = float(np.max(np.abs(matrix)))
    if size == 0.0:
        return np.linalg.eigh(matrix)
    values, vectors = np.linalg.eigh(matrix / size)
    return values * size, vectors


def _as_matrix(matrix: npt.ArrayLike) -> FloatArray:
    array = np.array(matrix, dtype=np.float64)
    if array.shape != (6, 6):
        raise CovarianceError(f"covariance must be 6x6, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise CovarianceError("covariance contains non-finite values")
    return array


def _inspect(array: FloatArray) -> tuple[FloatArray, float, FloatArray, FloatArray]:
    """Standard deviations ``D``, relative asymmetry, and the eigen-decomposition
    of the symmetric part of the correlation matrix ``D^-1 C D^-1``."""
    variances = np.abs(np.diag(array))
    # A subnormal variance is 0: its square root would make the ratios noise.
    deviations = np.where(variances >= _SMALLEST_NORMAL, np.sqrt(variances), 1.0)
    scaled = array / np.outer(deviations, deviations)
    asymmetry = float(np.max(np.abs(scaled - scaled.T)))
    eigenvalues, eigenvectors = _eigh(symmetric_part(scaled))
    return deviations, asymmetry, eigenvalues, eigenvectors


def _problems(array: FloatArray) -> tuple[str, ...]:
    _, asymmetry, eigenvalues, _ = _inspect(array)
    found: list[str] = []
    if asymmetry > SYMMETRY_RTOL:
        found.append(f"not symmetric (max relative asymmetry {asymmetry:.3g})")
    if eigenvalues[0] < -PSD_RTOL:
        found.append(
            "not positive semi-definite (min correlation-matrix eigenvalue "
            f"{eigenvalues[0]:.3g})"
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


def _clipped(block: FloatArray) -> tuple[FloatArray, int, float]:
    """A symmetric block with its negative eigenvalues set to 0.

    Returns the block, how many eigenvalues were below ``-PSD_RTOL`` and the
    most negative one.
    """
    values, vectors = _eigh(block)
    result: FloatArray = (vectors * np.clip(values, 0.0, None)) @ vectors.T
    return result, int(np.count_nonzero(values < -PSD_RTOL)), float(values[0])


def _repaired_correlation(scaled: FloatArray) -> tuple[FloatArray, list[str]]:
    """Make a symmetric correlation matrix positive semi-definite, minimally.

    The position block ``P`` is repaired only if it is itself invalid. The
    velocity block is then rebuilt from the conditional velocity covariance
    ``S = V - B P^+ B^T`` (the Schur complement, which is what the 3-D
    methods use), with the negative eigenvalues of ``S`` clipped. A valid
    position block, and with it 2-D Pc, is therefore never changed by a
    defect in the velocity terms.
    """
    position, cross, velocity = scaled[:3, :3], scaled[3:, :3], scaled[3:, 3:]
    notes: list[str] = []
    position, count, lowest = _clipped(position)
    if count:
        notes.append(
            f"clipped {count} negative position-block eigenvalue(s) to zero "
            f"(most negative {lowest:.3g} in correlation units)"
        )
    # B P^+ B^T over the directions the position block resolves; velocity
    # correlations along the others have nothing to condition on.
    values, vectors = _eigh(position)
    resolved = values > RANK_RTOL * max(float(values[-1]), 1.0)
    known, variances = vectors[:, resolved], values[resolved]
    cross = cross @ known @ known.T
    explained = (cross @ known / variances) @ (known.T @ cross.T)
    conditional, count, lowest = _clipped(symmetric_part(velocity - explained))
    if count:
        notes.append(
            f"clipped {count} negative eigenvalue(s) of the velocity covariance "
            f"conditional on position to zero (most negative {lowest:.3g} in "
            "correlation units)"
        )
    repaired = np.block(
        [[position, cross.T], [cross, symmetric_part(conditional + explained)]]
    )
    return repaired, notes


def _structured_repair(
    correlation: FloatArray, scale: FloatArray
) -> tuple[FloatArray, list[str]] | None:
    """:func:`_repaired_correlation` in covariance units, or None if it fails."""
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            repaired, notes = _repaired_correlation(correlation)
            fixed = symmetric_part(repaired * scale)
        valid = bool(np.all(np.isfinite(fixed))) and not _problems(fixed)
    except np.linalg.LinAlgError:
        return None
    return (fixed, notes) if valid else None


def repair(matrix: npt.ArrayLike) -> tuple[FloatArray, tuple[str, ...]]:
    """Symmetrise a matrix and make it positive semi-definite.

    Both are judged on the correlation matrix (see :data:`PSD_RTOL`), so a
    small block is repaired at its own scale. A negative eigenvalue is
    removed where it lies: in the position block if that block is invalid,
    otherwise in the velocity covariance conditional on position, leaving
    the position block (and 2-D Pc) unchanged.

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
    array = _as_matrix(matrix)
    deviations, asymmetry, eigenvalues, _ = _inspect(array)
    diagnostics: list[str] = []
    if asymmetry > SYMMETRY_RTOL:
        diagnostics.append(
            f"symmetrised covariance (max relative asymmetry {asymmetry:.3g})"
        )
    if eigenvalues[0] >= -PSD_RTOL:
        return symmetric_part(array), tuple(diagnostics)
    scale = np.outer(deviations, deviations)
    correlation = symmetric_part(array / scale)
    structured = _structured_repair(correlation, scale)
    notes: list[str] = []
    if structured is None:
        # Correlations far outside [-1, 1] defeat the conditioning; clipping
        # the whole correlation matrix always works.
        clipped, _, _ = _clipped(correlation)
        fixed = symmetric_part(clipped * scale)
    else:
        fixed, notes = structured
    # Each block's own eigenvalues may be within tolerance while the matrix
    # is not: the repair still ran, so it is still reported.
    diagnostics += notes or [
        f"made positive semi-definite (min correlation-matrix eigenvalue "
        f"{float(eigenvalues[0]):.3g})"
    ]
    return fixed, tuple(diagnostics)


def _require_gcrf(state: StateVector) -> None:
    if state.frame != "GCRF":
        raise GeometryError(
            f"RTN axes must be built from a GCRF state, got {state.frame}; "
            "convert with keepout.frames.to_gcrf"
        )


def _frozen(array: FloatArray) -> FloatArray:
    symmetric = symmetric_part(array)
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
