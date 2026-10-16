"""Private helpers shared by the package: array aliases, argument checks and
symmetric-matrix algebra."""

from __future__ import annotations

import enum
import functools
import math
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]
"""A float64 NumPy array."""

ComplexArray: TypeAlias = npt.NDArray[np.complex128]
"""A complex128 NumPy array."""

BoolArray: TypeAlias = npt.NDArray[np.bool_]
"""A boolean NumPy mask."""

IndexArray: TypeAlias = npt.NDArray[np.intp]
"""A NumPy array of indices."""


@functools.total_ordering
class OrderedEnum(enum.Enum):
    """An enum whose members are ordered by value, only against each other.

    Unlike :class:`enum.IntEnum`, members are not ints: they never compare
    equal to a number and are always true in a boolean context.
    """

    def __lt__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return bool(self.value < other.value)


def read_only(*arrays: npt.NDArray[Any]) -> None:
    """Mark arrays read-only in place."""
    for array in arrays:
        array.flags.writeable = False


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


def as_vector(value: npt.ArrayLike, name: str) -> FloatArray:
    """Return a read-only, finite float64 copy of a 3-vector.

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


def require_positive(value: float, name: str) -> None:
    """Raise ``ValueError`` unless ``value`` is finite and positive.

    Raises
    ------
    ValueError
        If ``value`` is not finite or not greater than zero.
    """
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be finite and > 0, got {value}")


def require_non_negative(value: float, name: str) -> None:
    """Raise ``ValueError`` unless ``value`` is finite and non-negative.

    Raises
    ------
    ValueError
        If ``value`` is not finite or is negative.
    """
    if not (math.isfinite(value) and value >= 0.0):
        raise ValueError(f"{name} must be finite and >= 0, got {value}")
