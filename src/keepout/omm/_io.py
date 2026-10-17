"""Shared by the OMM readers: the family, files, and keyword values into an
:class:`Omm`."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, NamedTuple, TypeVar

from astropy.time import Time

from keepout.ccsds import files
from keepout.ccsds.keywords import format_epoch, parse_value
from keepout.ccsds.message import Family, Located
from keepout.errors import OmmParseError, OmmWriteError
from keepout.omm.keywords import (
    ATTRIBUTES,
    FIXED_METADATA,
    KEYWORDS,
    SGP4_EPHEMERIS_TYPE,
    VERSIONS,
)
from keepout.omm.model import InvalidElement, Omm

_KEYWORD_OF: dict[str, str] = {attribute: key for key, attribute in ATTRIBUTES.items()}

_T = TypeVar("_T")


OMM: Final = Family(
    "OMM", OmmParseError, OmmWriteError, keyword_width=len("MEAN_ELEMENT_THEORY")
)
"""The OMM message family."""


def load_file(path: str | Path, loads: Callable[[str], _T]) -> _T:
    """Parse a file with ``loads``; see :func:`keepout.ccsds.files.read_file`.

    Raises
    ------
    OmmParseError
        If the file is broken gzip data or not UTF-8 text.
    """
    return files.load_file(path, loads, OMM)


@dataclass(frozen=True, slots=True)
class _Errors:
    """Makes the parse errors of one object, placed at the keyword's value."""

    values: Mapping[str, object]
    record: int | None
    line: int | None
    path: str | None

    def __call__(self, message: str, keyword: str) -> OmmParseError:
        place = self.values.get(keyword)
        if not isinstance(place, Located):
            place = Located("")
        return OmmParseError(
            message,
            record=self.record,
            line=self.line if place.line is None else place.line,
            path=self.path if place.path is None else place.path,
            keyword=keyword,
        )


def parse_record(
    values: Mapping[str, object],
    *,
    record: int | None = None,
    line: int | None = None,
    path: str | None = None,
    epoch: Time | None = None,
) -> Omm:
    """The elements of one object from its keyword values (message text).

    Keywords other than those of :data:`~keepout.omm.keywords.KEYWORDS` and
    :data:`~keepout.omm.keywords.FIXED_METADATA` are ignored.

    Parameters
    ----------
    values
        Value of each keyword: its text, or a
        :class:`~keepout.ccsds.message.Located` with its unit and place.
        Surrounding blanks are ignored.
    record
        Position of the object in its file, for error messages.
    line
        Line of the object in its file, for error messages about a value
        without a line of its own.
    path
        Element path of the object, likewise.
    epoch
        The ``EPOCH`` value, if the caller has parsed its text already (with
        :func:`~keepout.ccsds.keywords.preparse_epochs`, as
        :func:`~keepout.ccsds.keywords.parse_epoch` parses it); used when
        the value has no unit.

    Raises
    ------
    OmmParseError
        If a keyword is missing or has an invalid value or unit, the metadata
        are not those of SGP4 elements, or the elements are invalid.
    """
    error = _Errors(values, record, line, path)

    def text_of(keyword: str) -> Located:
        match values[keyword]:
            case Located() as value:
                return value._replace(text=value.text.strip())
            case str() as value:
                return Located(value.strip())
            case value:
                raise error(f"must be text, got {value!r}", keyword)

    for keyword, accepted in FIXED_METADATA.items():
        if keyword in values and (stated := text_of(keyword).text) not in accepted:
            raise error(f"must be {' or '.join(accepted)}, got {stated!r}", keyword)
    # Typed by each keyword's kind; Omm checks the types again.
    parsed: dict[str, Any] = {}
    for keyword, spec in KEYWORDS.items():
        if keyword not in values:
            raise error("missing keyword", keyword)
        value = text_of(keyword)
        if keyword == "EPOCH" and epoch is not None and value.unit is None:
            parsed[keyword] = epoch
            continue
        try:
            parsed[keyword] = parse_value(spec, value.text, value.unit)
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


def check_version(
    version: str,
    *,
    record: int | None = None,
    line: int | None = None,
    path: str | None = None,
) -> None:
    """Check that a message's ``CCSDS_OMM_VERS`` is one keepout reads.

    Raises
    ------
    OmmParseError
        If it is not one of :data:`~keepout.omm.keywords.VERSIONS`.
    """
    if version not in VERSIONS:
        raise OmmParseError(
            f"unsupported OMM version {version!r}, expected {' or '.join(VERSIONS)}",
            record=record,
            line=line,
            path=path,
            keyword="CCSDS_OMM_VERS",
        )


class CcsdsValue(NamedTuple):
    """One value of an OMM as a CCSDS message writes it."""

    keyword: str
    text: str
    unit: str | None
    group: tuple[str, ...]
    """NDM/XML element path of the value inside the section element."""


def ccsds_values(omm: Omm) -> list[CcsdsValue]:
    """Each value of an OMM, in CCSDS order.

    The metadata (the fixed metadata after ``OBJECT_ID``), then the mean
    elements and the TLE parameters, in the order of the OMM schema.
    """
    texts = omm.keywords()
    values: list[CcsdsValue] = []
    for keyword, spec in KEYWORDS.items():
        values.append(CcsdsValue(keyword, texts[keyword], spec.unit, spec.group))
        if keyword == "OBJECT_ID":
            values += (
                CcsdsValue(fixed, accepted[0], None, spec.group)
                for fixed, accepted in FIXED_METADATA.items()
            )
    return values


def creation_date_text(creation_date: Time | None) -> str:
    """``CREATION_DATE`` text of a message: the given time, or now (whole seconds)."""
    if creation_date is None:
        creation_date = Time(Time.now(), precision=0)
    return format_epoch(creation_date)
