"""Vector and symmetric-matrix helpers shared by the package."""

from __future__ import annotations

from typing import Any, Final, TypeVar

import numpy as np
import numpy.typing as npt

from keepout._types import FloatArray, read_only

__all__ = [
    "NEGLIGIBLE_VARIANCE",
    "as_axis",
    "as_vector",
    "as_vectors",
    "correlation_parts",
    "from_eigen",
    "sigma_axes",
    "symmetric_part",
    "take",
    "to_correlation",
]

_Record = TypeVar("_Record", bound=tuple[Any, ...])

NEGLIGIBLE_VARIANCE: Final = 1e-200
"""Variance, relative to the largest absolute entry, below which it is taken
as 0 when forming the correlation matrix. Dividing by its square root would
otherwise overflow; real covariances span far less (a mm/s velocity next to
a 10 km position is 1e-14)."""

_SMALLEST_NORMAL: Final = float(np.finfo(np.float64).tiny)


def take(record: _Record, index: Any) -> _Record:  # noqa: ANN401 (any NumPy index)
    """The rows ``index`` of every array of ``record``, as the same named tuple.

    Parameters
    ----------
    record
        A named tuple of arrays indexed alike, e.g. one row per time.
    index
        Any NumPy index of their first axis: an int, a slice, integer or
        boolean arrays.
    """
    return type(record)(*(array[index] for array in record))


def as_vector(value: npt.ArrayLike, name: str) -> FloatArray:
    """A read-only, finite float64 copy of a 3-vector.

    Parameters
    ----------
    value
        Anything NumPy can turn into a shape ``(3,)`` array.
    name
        Name used in error messages.

    Raises
    ------
    ValueError
        If the shape is not ``(3,)`` or a component is not finite.
    """
    array = np.array(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    read_only(array)
    return array


def as_vectors(value: npt.ArrayLike, name: str) -> FloatArray:
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


def as_axis(values: npt.ArrayLike, name: str) -> FloatArray:
    """Finite numbers along one axis, as a float64 array of shape ``(N,)``,
    N >= 1.

    Raises
    ------
    ValueError
        If they are empty, not one-dimensional or not finite.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def symmetric_part(matrix: FloatArray) -> FloatArray:
    """``(M + M^T) / 2`` of a matrix or a stack of matrices ``(..., n, n)``."""
    result: FloatArray = 0.5 * (matrix + np.swapaxes(matrix, -1, -2))
    return result


def sigma_axes(covariance: FloatArray, sigmas: float = 1.0) -> FloatArray:
    """The axes of the ``sigmas``-sigma ellipse (ellipsoid) of a symmetric
    covariance, as rows: ``sigmas * sqrt(lambda)`` along each eigenvector
    (a negative eigenvalue taken as zero)."""
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axes: FloatArray = (
        sigmas
        * np.sqrt(np.clip(eigenvalues, 0.0, None))[:, np.newaxis]
        * eigenvectors.T
    )
    return axes


def from_eigen(values: FloatArray, vectors: FloatArray) -> FloatArray:
    """``V diag(values) V^T`` of symmetric matrices, batched over leading axes.

    Applies a function to a symmetric matrix through its eigen-decomposition:
    pass ``f(eigenvalues)`` with the eigenvectors (columns), as
    :func:`numpy.linalg.eigh` returns them, to get ``f(M)`` (e.g. ``1 /
    values`` for the inverse, ``sqrt`` for the symmetric square root).

    Parameters
    ----------
    values
        Mapped eigenvalues, shape ``(..., n)``.
    vectors
        Unit eigenvectors as columns, shape ``(..., n, n)``.
    """
    result: FloatArray = (vectors * values[..., None, :]) @ np.swapaxes(vectors, -1, -2)
    return result


def correlation_parts(matrix: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """:func:`to_correlation`'s deviations ``D``, the scale ``D D^T`` it
    divides by, and the scaled matrix."""
    variances = np.abs(np.diag(matrix))
    floor = max(_SMALLEST_NORMAL, NEGLIGIBLE_VARIANCE * float(np.max(np.abs(matrix))))
    deviations = np.where(variances >= floor, np.sqrt(variances), 1.0)
    scale = np.outer(deviations, deviations)
    scaled: FloatArray = matrix / scale
    return deviations, scale, scaled


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
    deviations, _, scaled = correlation_parts(matrix)
    return deviations, scaled
