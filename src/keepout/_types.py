"""Private helpers shared by the package: the array alias and argument checks."""

from __future__ import annotations

import math
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]
"""A float64 NumPy array."""


def symmetric_part(matrix: FloatArray) -> FloatArray:
    """``(M + M^T) / 2`` of a matrix or a stack of matrices ``(..., n, n)``."""
    result: FloatArray = 0.5 * (matrix + np.swapaxes(matrix, -1, -2))
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
    array.flags.writeable = False
    return array


def require_positive(value: float, name: str) -> None:
    """Raise ``ValueError`` unless ``value`` is finite and positive."""
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be finite and > 0, got {value}")
