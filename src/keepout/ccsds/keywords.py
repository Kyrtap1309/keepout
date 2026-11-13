"""CCSDS keyword values shared by the message formats: typing and units.

A :class:`Keyword` says how one keyword of a CCSDS message (CDM, OMM) is
typed, converted to SI and grouped in NDM/XML; :func:`parse_value` and
:func:`format_value` convert its value between message text and Python, so
the message value is recovered exactly when it is written back. Epochs are
read and written by :mod:`keepout.ccsds.epochs`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias, assert_never

from astropy.time import Time

from keepout.ccsds.epochs import format_epoch, parse_epoch

__all__ = [
    "FieldValue",
    "Keyword",
    "ValueKind",
    "format_value",
    "message_number",
    "normalize_unit",
    "parse_value",
]

ValueKind: TypeAlias = Literal["str", "int", "float", "time"]
FieldValue: TypeAlias = str | int | float | Time


@dataclass(frozen=True, slots=True)
class Keyword:
    """How one CCSDS message keyword is typed, converted and placed.

    The keyword tables are :mod:`keepout.cdm.keywords` (CDM) and
    :mod:`keepout.omm.keywords` (OMM).

    Parameters
    ----------
    name
        Keyword as written in the message.
    kind
        Python type of the stored value.
    unit
        Canonical CCSDS unit, or None for unitless keywords.
    scale
        Factor converting the file value to SI.
    group
        NDM/XML element path inside the section element.
    """

    name: str
    kind: ValueKind
    unit: str | None = None
    scale: float = 1.0
    group: tuple[str, ...] = ()


_UNITLESS: Final = frozenset({"", "n/a"})
_INTEGER: Final = re.compile(r"^[+-]?[0-9]+$")
_REAL: Final = re.compile(r"^[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")


def normalize_unit(unit: str) -> str:
    """Canonical form for comparing units: lower case, no spaces, ``**`` or ``^``.

    Parameters
    ----------
    unit
        Unit text as written in a message.

    Returns
    -------
    str
        The unit in canonical form.
    """
    return unit.strip().lower().replace(" ", "").replace("**", "").replace("^", "")


def parse_value(keyword: Keyword, text: str, unit: str | None) -> FieldValue:
    """The Python value, in SI, of a value from a message.

    Parameters
    ----------
    keyword
        Table entry of the keyword.
    text
        Value text, without the unit.
    unit
        Unit given in the message, if any.

    Returns
    -------
    FieldValue
        The value, in SI units for a number.

    Raises
    ------
    ValueError
        If the unit does not match, or the text is not a valid value.
    """
    _check_unit(keyword, unit)
    if keyword.kind == "str":
        return text
    if not text:
        raise ValueError("empty value")
    # CCSDS numbers only: Python's int()/float() also take "1_0", "inf", ...
    match keyword.kind:
        case "time":
            return parse_epoch(text)
        case "int":
            if not _INTEGER.match(text):
                raise ValueError(f"invalid integer {text!r}")
            return int(text)
        case "float":
            if not _REAL.match(text):
                raise ValueError(f"invalid number {text!r}")
            number = float(text)
            if not math.isfinite(number):
                raise ValueError(f"number out of range {text!r}")
            return number * keyword.scale
        case _:
            assert_never(keyword.kind)


def _check_unit(keyword: Keyword, unit: str | None) -> None:
    """Check that ``unit``, if given, is the keyword's.

    Raises
    ------
    ValueError
        If ``unit`` is given for a unitless keyword, or does not match the
        keyword's unit.
    """
    if unit is None:
        return
    if keyword.unit is None:
        if normalize_unit(unit) not in _UNITLESS:
            raise ValueError(f"{keyword.name} takes no unit, got [{unit}]")
    elif normalize_unit(unit) != normalize_unit(keyword.unit):
        raise ValueError(f"unit [{unit}] does not match expected [{keyword.unit}]")


def format_value(keyword: Keyword, value: FieldValue) -> tuple[str, str | None]:
    """The message text and unit of a stored value.

    Parameters
    ----------
    keyword
        Table entry of the keyword.
    value
        Stored (SI) value.

    Returns
    -------
    tuple[str, str | None]
        The value's text and its unit, or None if the keyword has none.

    Raises
    ------
    TypeError
        If ``value`` is not of the keyword's kind.
    """
    match keyword.kind, value:
        case "time", Time():
            return format_epoch(value), None
        case "float", int() | float():
            return _shortest(float(value), keyword.scale).upper(), keyword.unit
        case "str" | "int", _:
            return str(value), None
    raise TypeError(f"{keyword.name} is a {keyword.kind} value, got {value!r}")


def _shortest(value: float, scale: float) -> str:
    """Shortest number text that :func:`parse_value` turns back into
    ``value``, or into a neighbouring double (see :func:`message_number`)."""
    return repr(message_number(value, scale))


_SIGNIFICANT: Final = tuple(f".{digits}g" for digits in range(1, 18))
"""Formats of 1 to 17 significant digits, enough for any double."""


def message_number(value: float, scale: float) -> float:
    """The number :func:`format_value` writes for a stored value, in message units.

    It is the shortest decimal that :func:`parse_value` scales back to
    ``value`` exactly, if there is one: ``value / scale`` itself may have
    spurious digits (6535.677249 km is 6535677.249 m, and dividing by 1000
    gives the neighbouring double 6535.677248999999); the shortest decimal is
    the one the message had. Reading the written text gives this number back.

    A value read from a message always comes back exactly. Other values
    (computed ones) may have no decimal that scales back to them; then
    ``value / scale`` is written, which scales back to within 1 ulp of
    ``value`` (a neighbouring double, for a few values in a hundred at
    scale 1000), unless ``value / scale`` is so small it is subnormal
    (below about 2.2e-308) and has fewer significant bits.

    Parameters
    ----------
    value
        Stored (SI) value.
    scale
        Factor converting the message value to SI (:attr:`Keyword.scale`).

    Returns
    -------
    float
        The value in the message's unit.
    """
    if scale == 1.0:
        # The shortest decimal that reads back as ``value`` is its repr.
        return value
    quotient = value / scale
    for spec in _SIGNIFICANT:
        candidate = float(format(quotient, spec))
        if candidate * scale == value:
            return candidate
    return quotient
