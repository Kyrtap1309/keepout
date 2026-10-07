"""Shared by the OMM readers: one object's keyword values into an :class:`Omm`."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar

from keepout.cdm.keywords import parse_value
from keepout.errors import OmmParseError
from keepout.omm.keywords import (
    ATTRIBUTES,
    FIXED_METADATA,
    KEYWORDS,
    SGP4_EPHEMERIS_TYPE,
)
from keepout.omm.model import InvalidElement, Omm

_KEYWORD_OF: dict[str, str] = {attribute: key for key, attribute in ATTRIBUTES.items()}

_T = TypeVar("_T")


def load_file(path: str | Path, loads: Callable[[str], _T]) -> _T:
    """Parse a UTF-8 file with ``loads``; a byte-order mark is ignored."""
    return loads(Path(path).read_text(encoding="utf-8-sig"))


def parse_record(
    values: Mapping[str, object],
    *,
    record: int | None = None,
    line: int | None = None,
) -> Omm:
    """The elements of one object from its keyword values (message text).

    Keywords other than those of :data:`~keepout.omm.keywords.KEYWORDS` and
    :data:`~keepout.omm.keywords.FIXED_METADATA` are ignored.

    Parameters
    ----------
    values
        Text value of each keyword, surrounding blanks ignored.
    record
        Position of the object in its file, for error messages.
    line
        Line of the object in its file, for error messages.

    Raises
    ------
    OmmParseError
        If a keyword is missing or has an invalid value, the metadata are not
        those of SGP4 elements, or the elements are invalid.
    """

    def error(message: str, keyword: str | None = None) -> OmmParseError:
        return OmmParseError(message, record=record, line=line, keyword=keyword)

    for keyword, expected in FIXED_METADATA.items():
        if keyword in values and _text(values[keyword]) != expected:
            raise error(f"must be {expected}, got {values[keyword]!r}", keyword)
    # Typed by each keyword's kind; Omm checks the types again.
    parsed: dict[str, Any] = {}
    for keyword, spec in KEYWORDS.items():
        if keyword not in values:
            raise error("missing keyword", keyword)
        text = _text(values[keyword])
        if text is None:
            raise error(f"must be text, got {values[keyword]!r}", keyword)
        try:
            parsed[keyword] = parse_value(spec, text, None)
        except ValueError as exc:
            raise error(str(exc), keyword) from exc
    if parsed["EPHEMERIS_TYPE"] != SGP4_EPHEMERIS_TYPE:
        raise error(
            f"ephemeris type {parsed['EPHEMERIS_TYPE']} is not SGP4 "
            f"({SGP4_EPHEMERIS_TYPE})",
            "EPHEMERIS_TYPE",
        )
    try:
        return Omm(**{ATTRIBUTES[key]: parsed[key] for key in ATTRIBUTES})
    except InvalidElement as exc:
        raise error(str(exc), _KEYWORD_OF[exc.attribute]) from exc


def _text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) else None
