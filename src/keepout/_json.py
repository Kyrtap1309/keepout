"""Typed readers of parsed JSON (and TOML) values.

Each reader returns its value as the type it checks, or raises the error
its caller makes (``error``, :class:`TypeError` by default) with the
message ``"<where> must be <a type>, got <value>"``. A bool is not a
number, and ``null`` is of no type: a caller that takes ``null`` tests for
``None`` first.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeAlias, cast

__all__ = [
    "ErrorFactory",
    "boolean",
    "integer",
    "items",
    "number",
    "obj",
    "text",
    "texts",
]

ErrorFactory: TypeAlias = Callable[[str], Exception]
"""What makes the error of a value of the wrong type, from its message."""


def _message(value: object, where: str, expected: str) -> str:
    """The message of ``value``, at ``where``, not being ``expected``."""
    return f"{where} must be {expected}, got {value!r}"


def number(value: object, where: str, *, error: ErrorFactory = TypeError) -> float:
    """``value``, an int or a float (not a bool), as a float.

    Raises
    ------
    Exception
        ``error``'s, if it is not a number.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise error(_message(value, where, "a number"))
    return float(value)


def integer(value: object, where: str, *, error: ErrorFactory = TypeError) -> int:
    """``value``, an int (not a bool).

    Raises
    ------
    Exception
        ``error``'s, if it is not an integer.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise error(_message(value, where, "an integer"))
    return value


def boolean(value: object, where: str, *, error: ErrorFactory = TypeError) -> bool:
    """``value``, ``true`` or ``false``.

    Raises
    ------
    Exception
        ``error``'s, if it is not a bool.
    """
    if not isinstance(value, bool):
        raise error(_message(value, where, "true or false"))
    return value


def text(value: object, where: str, *, error: ErrorFactory = TypeError) -> str:
    """``value``, a string.

    Raises
    ------
    Exception
        ``error``'s, if it is not a string.
    """
    if not isinstance(value, str):
        raise error(_message(value, where, "a string"))
    return value


def obj(
    value: object, where: str, *, error: ErrorFactory = TypeError
) -> Mapping[str, object]:
    """``value``, an object (a dict, whose keys JSON and TOML make strings).

    Raises
    ------
    Exception
        ``error``'s, if it is not an object.
    """
    if not isinstance(value, dict):
        raise error(_message(value, where, "an object"))
    return cast("Mapping[str, object]", value)


def items(
    value: object, where: str, *, error: ErrorFactory = TypeError
) -> list[object]:
    """``value``, an array (a list).

    Raises
    ------
    Exception
        ``error``'s, if it is not an array.
    """
    if not isinstance(value, list):
        raise error(_message(value, where, "an array"))
    return cast("list[object]", value)


def texts(
    value: object, where: str, *, error: ErrorFactory = TypeError
) -> tuple[str, ...]:
    """``value``, an array of strings.

    Raises
    ------
    Exception
        ``error``'s, if it is not one.
    """
    listed = value if isinstance(value, list) else None
    found = tuple(
        item for item in cast("list[object]", listed or []) if isinstance(item, str)
    )
    if listed is None or len(found) != len(listed):
        raise error(_message(value, where, "a list of strings"))
    return found
