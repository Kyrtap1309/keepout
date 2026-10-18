"""Legacy two-line element sets (TLE) as OMM mean elements.

A TLE (Hoots and Roehrich, Spacetrack Report No. 3; CelesTrak's "NORAD
Two-Line Element Set Format") holds the SGP4 elements of an OMM in two
69-column lines, each ending in a modulo-10 checksum. A title line with the
object's name may come first (the "3LE" form; Space-Track writes it as
``0 NAME``). Catalog numbers above 99999 are written in the Alpha-5 scheme
(``A0000`` is 100000, ``Z9999`` is 339999; I and O are skipped); larger ones
have no TLE form, which is why OMM replaces it.

Each field becomes its OMM keyword's text, read like any other OMM, so
:meth:`~keepout.omm.model.Omm.satrec` gives the record
:meth:`sgp4.api.Satrec.twoline2rv` makes from the lines. Two-digit years
57-99 are 1957-1999 and 00-56 are 2000-2056. The epoch's day fraction has
eight digits, a grid of 864 microseconds, so it becomes whole microseconds
exactly. A TLE without a name line gets the ``OBJECT_NAME`` ``UNKNOWN``, and
a blank international designator the ``OBJECT_ID`` ``UNKNOWN``, as CCSDS
502.0-B-3 (table 4-2) asks; a blank classification or ephemeris type takes
the CCSDS default (:data:`~keepout.omm.keywords.DEFAULTS`).
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final, NamedTuple

from sgp4.alpha5 import from_alpha5
from sgp4.io import compute_checksum

from keepout.ccsds.message import Located
from keepout.errors import OmmParseError
from keepout.omm._io import load_file, parse_record
from keepout.omm.keywords import DEFAULTS
from keepout.omm.model import Omm

__all__ = ["UNKNOWN", "checksum", "load", "loads"]

UNKNOWN: Final = "UNKNOWN"
"""``OBJECT_NAME`` or ``OBJECT_ID`` of a TLE that does not give it."""

_LINES: Final = {
    "1": re.compile(
        r"1 (?P<NORAD_CAT_ID>.{5})(?P<CLASSIFICATION_TYPE>.) (?P<OBJECT_ID>.{8}) "
        r"(?P<EPOCH>.{14}) (?P<MEAN_MOTION_DOT>.{10}) (?P<MEAN_MOTION_DDOT>.{8}) "
        r"(?P<BSTAR>.{8}) (?P<EPHEMERIS_TYPE>.) (?P<ELEMENT_SET_NO>.{4})(?P<sum>.)"
    ),
    "2": re.compile(
        r"2 (?P<NORAD_CAT_ID>.{5}) (?P<INCLINATION>.{8}) (?P<RA_OF_ASC_NODE>.{8}) "
        r"(?P<ECCENTRICITY>.{7}) (?P<ARG_OF_PERICENTER>.{8}) (?P<MEAN_ANOMALY>.{8}) "
        r"(?P<MEAN_MOTION>.{11})(?P<REV_AT_EPOCH>.{5})(?P<sum>.)"
    ),
}
"""Each line's fixed columns, named by OMM keyword (``sum``: the checksum)."""

_LAYOUTS: Final = {
    "1": "1 NNNNNC NNNNNAAA NNNNN.NNNNNNNN +.NNNNNNNN +NNNNN-N +NNNNN-N N NNNNS",
    "2": "2 NNNNN NNN.NNNN NNN.NNNN NNNNNNN NNN.NNNN NNN.NNNN NN.NNNNNNNNNNNNNS",
}
"""Each line's columns for error messages (``S``: the checksum)."""

_PIVOT_YEAR: Final = 57
"""Two-digit years from here on are 19xx, those below 20xx."""

_MICROSECONDS_PER_DIGIT: Final = 864
"""Microseconds in 1e-8 day, the last digit of a TLE epoch."""

_NUMBER: Final = re.compile(r" *[0-9]+")
_ALPHA5: Final = re.compile(r"[A-HJ-NP-Z][0-9]{4}")
_DESIGNATOR: Final = re.compile(
    r"(?P<year>[0-9]{2})(?P<launch>[0-9]{3})(?P<piece>[A-Z]{1,3}) *"
)
_EPOCH: Final = re.compile(
    r"(?P<year>[0-9]{2})(?P<day>[ 0-9]{2}[0-9])\.(?P<fraction>[0-9]{8})"
)
_POINT_DECIMAL: Final = re.compile(r"(?P<sign>[-+ ])\.(?P<digits>[0-9]{8})")
_EXPONENTIAL: Final = re.compile(
    r"(?P<sign>[-+ ])(?P<digits>[0-9]{5})(?P<exponent>[-+][0-9])"
)
_DECIMAL: Final = re.compile(r" *[0-9]+\.[0-9]+")
_DIGITS: Final = re.compile(r"[0-9]{7}")


def checksum(line: str) -> int:
    """The TLE checksum of a line: its digits summed, each minus sign as 1, mod 10.

    Parameters
    ----------
    line
        The line; only its first 68 columns count.
    """
    return compute_checksum(line)


def _match(pattern: re.Pattern[str], text: str) -> re.Match[str]:
    match = pattern.fullmatch(text)
    if match is None:
        raise ValueError(f"invalid TLE field {text!r}")
    return match


def _catalog_number(text: str) -> str:
    if not _NUMBER.fullmatch(text):
        _match(_ALPHA5, text)
    return str(from_alpha5(text))


def _year(two_digits: str) -> int:
    year = int(two_digits)
    return year + (1900 if year >= _PIVOT_YEAR else 2000)


def _designator(text: str) -> str:
    if not text.strip():
        return UNKNOWN
    match = _match(_DESIGNATOR, text)
    return f"{_year(match['year'])}-{match['launch']}{match['piece']}"


def _epoch(text: str) -> str:
    match = _match(_EPOCH, text)
    year, day = _year(match["year"]), int(match["day"])
    epoch = datetime.datetime(year, 1, 1, tzinfo=datetime.UTC) + datetime.timedelta(
        days=day - 1,
        microseconds=int(match["fraction"]) * _MICROSECONDS_PER_DIGIT,
    )
    if day < 1 or epoch.year != year:
        raise ValueError(f"invalid TLE field {text!r}: no day {day} in {year}")
    return epoch.strftime("%Y-%m-%dT%H:%M:%S.%f")


def _point_decimal(text: str) -> str:
    match = _match(_POINT_DECIMAL, text)
    return f"{match['sign'].strip()}.{match['digits']}"


def _exponential(text: str) -> str:
    match = _match(_EXPONENTIAL, text)
    return f"{match['sign'].strip()}.{match['digits']}E{match['exponent']}"


def _number(text: str) -> str:
    return _match(_NUMBER, text).group().strip()


def _decimal(text: str) -> str:
    return _match(_DECIMAL, text).group().strip()


def _eccentricity(text: str) -> str:
    return "." + _match(_DIGITS, text).group()


def _classification(text: str) -> str:
    return text.strip() or DEFAULTS["CLASSIFICATION_TYPE"]


def _ephemeris_type(text: str) -> str:
    return text.strip() or DEFAULTS["EPHEMERIS_TYPE"]


_FIELDS: Final[Mapping[str, Mapping[str, Callable[[str], str]]]] = {
    "1": {
        "NORAD_CAT_ID": _catalog_number,
        "CLASSIFICATION_TYPE": _classification,
        "OBJECT_ID": _designator,
        "EPOCH": _epoch,
        "MEAN_MOTION_DOT": _point_decimal,
        "MEAN_MOTION_DDOT": _exponential,
        "BSTAR": _exponential,
        "EPHEMERIS_TYPE": _ephemeris_type,
        "ELEMENT_SET_NO": _number,
    },
    "2": {
        "INCLINATION": _decimal,
        "RA_OF_ASC_NODE": _decimal,
        "ECCENTRICITY": _eccentricity,
        "ARG_OF_PERICENTER": _decimal,
        "MEAN_ANOMALY": _decimal,
        "MEAN_MOTION": _decimal,
        "REV_AT_EPOCH": _number,
    },
}
"""The OMM keyword text of each TLE field, by line."""


def loads(text: str) -> tuple[Omm, ...]:
    """The OMMs of TLEs: two lines each, optionally after a name line.

    Blank lines are skipped and trailing blanks ignored. A name line may start
    with ``0`` and a blank (Space-Track's 3LE form), which is not part of the
    name.

    Parameters
    ----------
    text
        The element sets.

    Returns
    -------
    tuple[Omm, ...]
        The objects in file order (none for blank text).

    Raises
    ------
    OmmParseError
        If the lines are not (name,) line 1, line 2 in turn (a name may not
        start like line 2), a line does not have the TLE layout or its
        checksum is wrong, the two lines are of different objects, or a field
        is invalid. The error names the element set (``record``), line and
        OMM keyword.
    """
    lines = [
        (number, line.rstrip())
        for number, line in enumerate(text.splitlines(), start=1)
        if line.strip()
    ]
    omms: list[Omm] = []
    index = 0
    while index < len(lines):
        record = len(omms) + 1
        number, title = lines[index]
        name = UNKNOWN
        if title.startswith("2 "):
            raise OmmParseError("expected TLE line 1", record=record, line=number)
        if not title.startswith("1 "):
            name = title.removeprefix("0 ").strip()
            index += 1
        values: dict[str, Located] = {"OBJECT_NAME": Located(name, line=number)}
        first = _fields(lines, index, "1", record)
        second = _fields(lines, index + 1, "2", record)
        if first.number != second.number:
            raise OmmParseError(
                f"line 2 is of object {second.number!r}, line 1 of {first.number!r}",
                record=record,
                line=lines[index + 1][0],
                keyword="NORAD_CAT_ID",
            )
        for line in (first, second):
            values.update(line.values)
        omms.append(parse_record(values, record=record))
        index += 2
    return tuple(omms)


class _Line(NamedTuple):
    """One TLE line read: its catalog number field and its OMM values."""

    number: str
    values: dict[str, Located]


def _fields(lines: list[tuple[int, str]], index: int, digit: str, record: int) -> _Line:
    """TLE line ``digit`` at ``index``: layout and checksum checked, converted.

    Raises
    ------
    OmmParseError
        If the line is missing, is not line ``digit``, has another layout, a
        wrong checksum or an invalid field.
    """
    if index >= len(lines) or not lines[index][1].startswith(f"{digit} "):
        at = lines[index][0] if index < len(lines) else lines[-1][0] + 1
        raise OmmParseError(f"expected TLE line {digit}", record=record, line=at)
    number, line = lines[index]
    match = _LINES[digit].fullmatch(line)
    if match is None:
        raise OmmParseError(
            f"TLE line {digit} must have 69 columns laid out as "
            f"{_LAYOUTS[digit]!r}, got {line!r}",
            record=record,
            line=number,
        )
    if match["sum"] != str(checksum(line)):
        raise OmmParseError(
            f"TLE line {digit} checksum is {match['sum']!r}, "
            f"its columns give {checksum(line)}",
            record=record,
            line=number,
        )
    values: dict[str, Located] = {}
    for keyword, convert in _FIELDS[digit].items():
        try:
            values[keyword] = Located(convert(match[keyword]), line=number)
        except ValueError as exc:
            raise OmmParseError(
                str(exc), record=record, line=number, keyword=keyword
            ) from exc
    return _Line(match["NORAD_CAT_ID"], values)


def load(path: str | Path) -> tuple[Omm, ...]:
    """The OMMs in a TLE file (UTF-8, or gzip-compressed UTF-8); see :func:`loads`.

    Parameters
    ----------
    path
        File to read.
    """
    return load_file(path, loads)
