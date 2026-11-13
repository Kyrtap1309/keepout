"""Position-velocity covariance: validation, repair and frame changes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, TypeAlias, get_args

import numpy as np
import numpy.typing as npt

from keepout._linalg import from_eigen, symmetric_part
from keepout._types import FloatArray, read_only
from keepout._validation import require_choice
from keepout.errors import CovarianceError, UnsupportedFrame
from keepout.frames import inertial_to_rtn_covariance, rtn_to_inertial_covariance
from keepout.state import StateVector, require_frame

if TYPE_CHECKING:
    from collections.abc import Callable

    from astropy.time import Time

__all__ = [
    "COVARIANCE_FRAMES",
    "NEGLIGIBLE_VARIANCE",
    "POLICIES",
    "PSD_RTOL",
    "RANK_RTOL",
    "SYMMETRY_RTOL",
    "Covariance",
    "CovarianceFrame",
    "Policy",
    "make_covariance",
    "problems",
    "repair",
    "to_correlation",
]

CovarianceFrame: TypeAlias = Literal["RTN", "GCRF"]
"""Frames a :class:`Covariance` can be expressed in."""

COVARIANCE_FRAMES: Final[tuple[CovarianceFrame, ...]] = get_args(CovarianceFrame)
"""Every :data:`CovarianceFrame`: ``("RTN", "GCRF")``."""

Policy: TypeAlias = Literal["raise", "repair"]
"""What to do with an invalid covariance: raise, or repair with a diagnostic."""

POLICIES: Final[tuple[Policy, ...]] = get_args(Policy)
"""Every :data:`Policy`: ``("raise", "repair")``."""

_FRAME: Final = "covariance frame"
"""What a frame is, in the error for an unsupported one."""

SYMMETRY_RTOL: Final = 1e-9
"""Allowed asymmetry of ``C[i, j]``, relative to ``sqrt(|C[i, i] C[j, j]|)``."""

PSD_RTOL: Final = 1e-10
"""Allowed negative eigenvalue of the correlation matrix ``D^-1 C D^-1``.

``D = diag(sqrt|C[i, i]|)`` (1 where a variance is negligible, see
:data:`NEGLIGIBLE_VARIANCE`), so each block is judged at its own scale: a
velocity variance of 1e-4 m^2/s^2 is checked as strictly as a position
variance of 1e8 m^2."""

NEGLIGIBLE_VARIANCE: Final = 1e-200
"""Variance, relative to the largest absolute entry, below which it is taken
as 0 when forming the correlation matrix. Dividing by its square root would
otherwise overflow; real covariances span far less (a mm/s velocity next to
a 10 km position is 1e-14)."""


RANK_RTOL: Final = 1e-8
"""Position-block correlation eigenvalue below which :func:`repair` treats
the direction as unknown: velocity correlations along it are dropped rather
than divided by a vanishing variance."""

_SMALLEST_NORMAL: Final = float(np.finfo(np.float64).tiny)


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


class _Inspection(NamedTuple):
    """A matrix ``C`` judged at the scale of its standard deviations ``D``."""

    scale: FloatArray
    """``D D^T``: the outer product of the standard deviations."""
    correlation: FloatArray
    """Symmetric part of the correlation matrix ``D^-1 C D^-1``."""
    asymmetry: float
    """Relative asymmetry: the largest ``|R[i, j] - R[j, i]|``, ``R = D^-1 C D^-1``."""
    eigenvalues: FloatArray
    """Eigenvalues of ``correlation``, ascending."""


def to_correlation(matrix: FloatArray) -> tuple[FloatArray, FloatArray]:
    """A square matrix ``C`` scaled to correlation units ``D^-1 C D^-1``.

    ``D = diag(sqrt|C[i, i]|)``, with 1 where a variance is negligible (see
    :data:`NEGLIGIBLE_VARIANCE`), so that row and column are left as they are
    instead of being divided by a vanishing deviation.

    Parameters
    ----------
    matrix
        A square matrix, e.g. a 6x6 covariance [m, m/s].

    Returns
    -------
    tuple
        The deviations (diagonal of ``D``) and the scaled matrix.
    """
    variances = np.abs(np.diag(matrix))
    floor = max(_SMALLEST_NORMAL, NEGLIGIBLE_VARIANCE * float(np.max(np.abs(matrix))))
    deviations = np.where(variances >= floor, np.sqrt(variances), 1.0)
    scaled: FloatArray = matrix / np.outer(deviations, deviations)
    return deviations, scaled


def _inspect(array: FloatArray) -> _Inspection:
    deviations, scaled = to_correlation(array)
    scale = np.outer(deviations, deviations)
    asymmetry = float(np.max(np.abs(scaled - scaled.T)))
    correlation = symmetric_part(scaled)
    eigenvalues, _ = _eigh(correlation)
    return _Inspection(scale, correlation, asymmetry, eigenvalues)


def _problems(array: FloatArray) -> tuple[str, ...]:
    _, _, asymmetry, eigenvalues = _inspect(array)
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
    """Why a matrix is not a valid covariance, in words.

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


class _Clipped(NamedTuple):
    """A symmetric block's eigen-decomposition with negative eigenvalues set to 0."""

    values: FloatArray
    """Eigenvalues, ascending, clipped to be non-negative."""
    vectors: FloatArray
    """Unit eigenvectors (columns)."""
    negatives: int
    """How many eigenvalues were below ``-PSD_RTOL``."""
    lowest: float
    """The most negative eigenvalue before clipping (the smallest, if none is)."""

    @classmethod
    def of(cls, block: FloatArray) -> _Clipped:
        values, vectors = _eigh(block)
        negatives = int(np.count_nonzero(values < -PSD_RTOL))
        return cls(np.clip(values, 0.0, None), vectors, negatives, float(values[0]))

    @property
    def matrix(self) -> FloatArray:
        """The clipped block."""
        return from_eigen(self.values, self.vectors)


def _repaired_correlation(scaled: FloatArray) -> tuple[FloatArray, list[str]]:
    """A symmetric correlation matrix made positive semi-definite, minimally.

    The position block ``P`` is replaced by its clipped form if any of its
    eigenvalues is negative, but noted only when one is below ``-PSD_RTOL``.
    The velocity block is then rebuilt from the conditional velocity
    covariance ``S = V - B P^+ B^T`` (the Schur complement, which is what the
    3-D methods use), with the negative eigenvalues of ``S`` clipped. The
    position block, and with it 2-D Pc, is therefore never changed by a
    defect in the velocity terms.
    """
    position, cross, velocity = scaled[:3, :3], scaled[3:, :3], scaled[3:, 3:]
    notes: list[str] = []
    spectrum = _Clipped.of(position)
    if spectrum.negatives:
        notes.append(
            f"clipped {spectrum.negatives} negative position-block eigenvalue(s) to "
            f"zero (most negative {spectrum.lowest:.3g} in correlation units)"
        )
    if spectrum.lowest < 0.0:
        position = spectrum.matrix
    # B P^+ B^T over the directions the position block resolves; velocity
    # correlations along the others have nothing to condition on.
    values, vectors = spectrum.values, spectrum.vectors
    resolved = values > RANK_RTOL * max(float(values[-1]), 1.0)
    known, variances = vectors[:, resolved], values[resolved]
    resolved_cross = cross @ known @ known.T
    explained = (resolved_cross @ known / variances) @ (known.T @ resolved_cross.T)
    conditional = _Clipped.of(symmetric_part(velocity - explained))
    if conditional.negatives:
        notes.append(
            f"clipped {conditional.negatives} negative eigenvalue(s) of the velocity "
            "covariance conditional on position to zero (most negative "
            f"{conditional.lowest:.3g} in correlation units)"
        )
    repaired = np.block(
        [
            [position, resolved_cross.T],
            [resolved_cross, symmetric_part(conditional.matrix + explained)],
        ]
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
    """A matrix symmetrised and made positive semi-definite.

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
    scale, correlation, asymmetry, eigenvalues = _inspect(array)
    diagnostics: list[str] = []
    if asymmetry > SYMMETRY_RTOL:
        diagnostics.append(
            f"symmetrised covariance (max relative asymmetry {asymmetry:.3g})"
        )
    if eigenvalues[0] >= -PSD_RTOL:
        return symmetric_part(array), tuple(diagnostics)
    structured = _structured_repair(correlation, scale)
    notes: list[str] = []
    if structured is None:
        # Correlations far outside [-1, 1] defeat the conditioning; clipping
        # the whole correlation matrix always works.
        fixed = symmetric_part(_Clipped.of(correlation).matrix * scale)
    else:
        fixed, notes = structured
    # Each block's own eigenvalues may be within tolerance while the matrix
    # is not: the repair still ran, so it is still reported.
    diagnostics += notes or [
        (
            f"made positive semi-definite (min correlation-matrix eigenvalue "
            f"{float(eigenvalues[0]):.3g})"
        )
    ]
    return fixed, tuple(diagnostics)


def _frozen(array: FloatArray) -> FloatArray:
    symmetric = symmetric_part(array)
    read_only(symmetric)
    return symmetric


@dataclass(frozen=True, eq=False, slots=True)
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
    UnsupportedFrame
        If ``frame`` is not one of :data:`COVARIANCE_FRAMES`.
    """

    matrix: FloatArray
    frame: CovarianceFrame
    epoch: Time
    repairs: tuple[str, ...] = field(default=(), kw_only=True)

    def __post_init__(self) -> None:
        require_choice(self.frame, COVARIANCE_FRAMES, _FRAME, UnsupportedFrame)
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
        """A matrix and frame already known to be valid, wrapped without
        their checks."""
        covariance = object.__new__(cls)
        object.__setattr__(covariance, "matrix", _frozen(matrix))
        object.__setattr__(covariance, "frame", frame)
        object.__setattr__(covariance, "epoch", epoch)
        object.__setattr__(covariance, "repairs", repairs)
        return covariance

    def at(self, epoch: Time) -> Covariance:
        """The same matrix at another epoch, as a model holds it fixed.

        The matrix is not checked again.

        Parameters
        ----------
        epoch
            The new epoch.

        Returns
        -------
        Covariance
            The same matrix, frame and repairs, at ``epoch``.
        """
        moved = object.__new__(Covariance)
        for name in ("matrix", "frame", "repairs"):
            object.__setattr__(moved, name, getattr(self, name))
        object.__setattr__(moved, "epoch", epoch)
        return moved

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

        Returns
        -------
        Covariance
            The 6x6 covariance with a zero velocity block [m^2, m^2/s, m^2/s^2].

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

    @property
    def position_only(self) -> bool:
        """Whether the covariance is known only in position: every velocity
        term is zero, as from :meth:`from_position`."""
        return not np.any(self.matrix[3:])

    def in_gcrf(self, state: StateVector) -> Covariance:
        """The covariance in GCRF.

        Parameters
        ----------
        state
            GCRF state of the object; defines the RTN axes.

        Returns
        -------
        Covariance
            The covariance in GCRF [m^2, m^2/s, m^2/s^2], at the same epoch.
        """
        return self._in_frame("GCRF", state, rtn_to_inertial_covariance)

    def in_rtn(self, state: StateVector) -> Covariance:
        """The covariance in the object's RTN frame.

        Parameters
        ----------
        state
            GCRF state of the object; defines the RTN axes.

        Returns
        -------
        Covariance
            The covariance in the object's RTN frame [m^2, m^2/s, m^2/s^2], at the same
            epoch.
        """
        return self._in_frame("RTN", state, inertial_to_rtn_covariance)

    def _in_frame(
        self,
        target: CovarianceFrame,
        state: StateVector,
        rotate: Callable[[FloatArray, FloatArray, FloatArray], FloatArray],
    ) -> Covariance:
        require_frame(state, "GCRF", "RTN axes")
        if self.frame == target:
            return self
        matrix = rotate(self.matrix, state.r, state.v)
        return Covariance._trusted(matrix, target, self.epoch, self.repairs)


def make_covariance(
    matrix: npt.ArrayLike,
    frame: CovarianceFrame,
    epoch: Time,
    *,
    policy: Policy = "repair",
) -> Covariance:
    """A :class:`Covariance`, repaired if the policy allows.

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

    Returns
    -------
    Covariance
        The validated covariance; its repairs, if any, in ``repairs``.

    Raises
    ------
    CovarianceError
        If the matrix is not 6x6 or not finite, or, with ``policy="raise"``,
        not a valid covariance.
    UnsupportedFrame
        If ``frame`` is not one of :data:`COVARIANCE_FRAMES`.
    ValueError
        If ``policy`` is not one of :data:`POLICIES`.
    """
    require_choice(policy, POLICIES, "covariance policy")
    require_choice(frame, COVARIANCE_FRAMES, _FRAME, UnsupportedFrame)
    array = _as_matrix(matrix)
    if policy == "raise":
        return Covariance(array, frame, epoch)
    fixed, repairs = repair(array)
    return Covariance._trusted(fixed, frame, epoch, repairs)  # noqa: SLF001
