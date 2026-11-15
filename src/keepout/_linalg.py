"""Vector and symmetric-matrix helpers shared by the package."""

from __future__ import annotations

from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

from keepout._types import FloatArray, read_only

__all__ = [
    "as_axis",
    "as_vector",
    "from_eigen",
    "sigma_axes",
    "symmetric_part",
    "take",
]

_Record = TypeVar("_Record", bound=tuple[Any, ...])


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
