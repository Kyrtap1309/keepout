"""Private helpers shared by the package: array aliases, argument checks and
symmetric-matrix algebra."""

from __future__ import annotations

import enum
import functools
import math
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    # typing.override from Python 3.12, for the type checkers only, so
    # typing_extensions stays out of the dependencies.
    from typing_extensions import override
else:
    _Function = TypeVar("_Function")

    def override(method: _Function) -> _Function:
        """Mark a method as overriding one of its base class (checked statically)."""
        return method


__all__ = [
    "BoolArray",
    "ComplexArray",
    "FloatArray",
    "IndexArray",
    "JsonValue",
    "OrderedEnum",
    "as_vector",
    "from_eigen",
    "is_probability",
    "override",
    "read_only",
    "require_at_least",
    "require_non_negative",
    "require_positive",
    "require_probability",
    "symmetric_part",
]

FloatArray: TypeAlias = npt.NDArray[np.float64]
"""A float64 NumPy array."""

ComplexArray: TypeAlias = npt.NDArray[np.complex128]
"""A complex128 NumPy array."""

BoolArray: TypeAlias = npt.NDArray[np.bool_]
"""A boolean NumPy mask."""

IndexArray: TypeAlias = npt.NDArray[np.intp]
"""A NumPy array of indices."""

JsonValue: TypeAlias = (
    "dict[str, JsonValue] | list[JsonValue] | str | int | float | bool | None"
)
"""A value :func:`json.loads` returns, to narrow with ``isinstance``."""


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


def is_probability(value: float) -> bool:
    """Whether ``value`` is a probability: finite and in [0, 1]."""
    return math.isfinite(value) and 0.0 <= value <= 1.0


def require_probability(value: float, name: str) -> None:
    """Raise ``ValueError`` unless ``value`` is a probability.

    Raises
    ------
    ValueError
        If ``value`` is not finite or not in [0, 1].
    """
    if not is_probability(value):
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def require_at_least(value: int, minimum: int, name: str) -> None:
    """Raise ``ValueError`` unless the count ``value`` is at least ``minimum``.

    Raises
    ------
    ValueError
        If ``value`` is below ``minimum``.
    """
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
