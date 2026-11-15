"""Type aliases shared by the package (NumPy arrays, JSON values), with
:func:`read_only` for those arrays, :class:`OrderedEnum` and
:func:`override`.

Argument checks are in :mod:`keepout._validation`, vector and matrix
helpers in :mod:`keepout._linalg`."""

from __future__ import annotations

import enum
import functools
from typing import TYPE_CHECKING, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    # typing.override from Python 3.12, for the type checkers only, so
    # typing_extensions stays out of the dependencies.
    from typing_extensions import override
else:
    _Function = TypeVar("_Function")

    def override(method: _Function) -> _Function:
        """The marker of a method overriding its base class's (checked statically)."""
        return method


__all__ = [
    "BoolArray",
    "ComplexArray",
    "FloatArray",
    "IndexArray",
    "JsonValue",
    "OrderedEnum",
    "override",
    "read_only",
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

    @property
    def key(self) -> str:
        """The member's stable name in data such as JSON: its name in lower
        case, ``prepare_maneuver``."""
        return self.name.lower()


def read_only(*arrays: npt.NDArray[np.generic]) -> None:
    """The ``arrays``, made read-only in place."""
    for array in arrays:
        array.flags.writeable = False
