"""Vector and symmetric-matrix helpers shared by the package."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from keepout._types import FloatArray, read_only

__all__ = ["as_vector", "from_eigen", "symmetric_part"]


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


def symmetric_part(matrix: FloatArray) -> FloatArray:
    """``(M + M^T) / 2`` of a matrix or a stack of matrices ``(..., n, n)``."""
    result: FloatArray = 0.5 * (matrix + np.swapaxes(matrix, -1, -2))
    return result


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
