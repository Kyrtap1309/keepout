"""Shared by the OMM readers: the family, files, and keyword values into an
:class:`Omm`."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, NamedTuple

from astropy.time import Time

from keepout.ccsds.epochs import format_epoch
from keepout.ccsds.keywords import parse_value
from keepout.ccsds.message import Family, Located
from keepout.errors import OmmParseError, OmmWriteError
from keepout.omm.keywords import (
    ATTRIBUTE_KEYWORDS,
    ATTRIBUTES,
    FIXED_METADATA,
    KEYWORDS,
    SGP4_EPHEMERIS_TYPE,
    VERSION_KEYWORD,
    VERSIONS,
)
from keepout.omm.model import InvalidElement, Omm

OMM: Final = Family(
    "OMM",
    OmmParseError,
    OmmWriteError,
    keyword_width=len("MEAN_ELEMENT_THEORY"),
    unknown_is_text=True,
)
"""The OMM message family; keywords it does not keep are read as text."""


@dataclass(frozen=True, slots=True)
class _Errors:
    """The maker of one object's parse errors, placed at the keyword's value."""

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
        :func:`~keepout.ccsds.epochs.preparse_epochs`, as
        :func:`~keepout.ccsds.epochs.parse_epoch` parses it); used when
        the value has no unit.

    Raises
    ------
    OmmParseError
        If a keyword is missing or has an invalid value or unit, the metadata
        are not those of SGP4 elements, or the elements are invalid.
    """
    make_error = _Errors(values, record, line, path)

    def text_of(keyword: str) -> Located:
        match values[keyword]:
            case Located() as value:
                return value._replace(text=value.text.strip())
            case str() as value:
                return Located(value.strip())
            case value:
                raise make_error(f"must be text, got {value!r}", keyword)

    for keyword, accepted in FIXED_METADATA.items():
        if keyword in values and (stated := text_of(keyword).text) not in accepted:
            raise make_error(
                f"must be {' or '.join(accepted)}, got {stated!r}", keyword
            )
    # Typed by each keyword's kind; Omm checks the types again.
    parsed: dict[str, Any] = {}
    for keyword, spec in KEYWORDS.items():
        if keyword not in values:
            raise make_error("missing keyword", keyword)
        value = text_of(keyword)
        if keyword == "EPOCH" and epoch is not None and value.unit is None:
            parsed[keyword] = epoch
            continue
        try:
            parsed[keyword] = parse_value(spec, value.text, value.unit)
        except ValueError as exc:
            raise make_error(str(exc), keyword) from exc
    if parsed["EPHEMERIS_TYPE"] != SGP4_EPHEMERIS_TYPE:
        raise make_error(
            f"ephemeris type {parsed['EPHEMERIS_TYPE']} is not SGP4 "
            f"({SGP4_EPHEMERIS_TYPE})",
            "EPHEMERIS_TYPE",
        )
    try:
        return Omm(**{ATTRIBUTES[key]: parsed[key] for key in ATTRIBUTES})
    except InvalidElement as exc:
        raise make_error(str(exc), ATTRIBUTE_KEYWORDS[exc.attribute].name) from exc


def check_version(
    version: str,
    *,
    record: int | None = None,
    line: int | None = None,
    path: str | None = None,
) -> None:
    """A check that a message's ``CCSDS_OMM_VERS`` is one keepout reads.

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
            keyword=VERSION_KEYWORD,
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
