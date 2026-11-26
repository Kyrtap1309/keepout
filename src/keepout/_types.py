"""Type aliases shared by the package (NumPy arrays, JSON values), with
:func:`read_only` and :func:`freeze_grid` for those arrays,
:class:`OrderedEnum` and :func:`~keepout._typing.override` (re-exported).

Argument checks are in :mod:`keepout._validation`, vector and matrix
helpers in :mod:`keepout._linalg`."""

from __future__ import annotations

import enum
import functools
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

from keepout._typing import override

__all__ = [
    "BoolArray",
    "ComplexArray",
    "FloatArray",
    "IndexArray",
    "InexactArray",
    "JsonValue",
    "OrderedEnum",
    "freeze_grid",
    "override",
    "read_only",
]

FloatArray: TypeAlias = npt.NDArray[np.float64]
"""A float64 NumPy array."""

ComplexArray: TypeAlias = npt.NDArray[np.complex128]
"""A complex128 NumPy array."""

InexactArray: TypeAlias = npt.NDArray[np.inexact[Any]]
"""A real or complex array: complex where it carries a complex-step
derivative (see :func:`keepout.equinoctial.complex_step`)."""

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


def freeze_grid(
    owner: object,
    axes: tuple[str, str],
    cells: tuple[str, ...],
    trailing: tuple[int, ...] = (),
) -> None:
    """The array fields of a frozen dataclass replaced by read-only float64
    copies: two axes of lengths ``(B, D)`` and the ``cells`` over them, of
    shape ``(B, D, *trailing)``.

    Raises
    ------
    ValueError
        If a cell array has another shape.
    """
    copies = {
        name: np.array(getattr(owner, name), dtype=np.float64)
        for name in (*axes, *cells)
    }
    first, second = (copies[name].shape for name in axes)
    shape = (*first, *second, *trailing)
    if any(copies[name].shape != shape for name in cells):
        got = " and ".join(str(copies[name].shape) for name in cells)
        raise ValueError(f"{' and '.join(cells)} must have shape {shape}, got {got}")
    read_only(*copies.values())
    for name, array in copies.items():
        object.__setattr__(owner, name, array)
