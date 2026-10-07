"""Shared by the OMM readers: files, and keyword values into an :class:`Omm`."""

from __future__ import annotations

import gzip
import zlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final, NamedTuple, TypeVar

from astropy.time import Time

from keepout.ccsds.keywords import format_epoch, parse_value
from keepout.errors import OmmParseError
from keepout.omm.keywords import (
    ATTRIBUTES,
    FIXED_METADATA,
    KEYWORDS,
    SGP4_EPHEMERIS_TYPE,
    VERSIONS,
)
from keepout.omm.model import InvalidElement, Omm

_KEYWORD_OF: dict[str, str] = {attribute: key for key, attribute in ATTRIBUTES.items()}

_GZIP_MAGIC: Final = b"\x1f\x8b"

_T = TypeVar("_T")


class Text(NamedTuple):
    """A keyword's value as written in a message, with its unit and place.

    Parameters
    ----------
    text
        The value, without its unit.
    unit
        The unit given with the value (KVN ``[unit]``, XML ``units``), if any.
    line
        Line of the value in a text format.
    path
        Element path of the value in an XML message.
    """

    text: str
    unit: str | None = None
    line: int | None = None
    path: str | None = None


def read_file(path: str | Path) -> str:
    """The text of a UTF-8 file, decompressed first if it is gzip data.

    gzip data is recognised by its magic bytes, whatever the file name. A
    byte-order mark is ignored.

    Raises
    ------
    OmmParseError
        If the file is broken gzip data or not UTF-8 text.
    """
    data = Path(path).read_bytes()
    if data.startswith(_GZIP_MAGIC):
        try:
            data = gzip.decompress(data)
        except (OSError, EOFError, zlib.error) as exc:
            raise OmmParseError(f"{path}: invalid gzip data: {exc}") from exc
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise OmmParseError(f"{path}: not UTF-8 text: {exc}") from exc


def load_file(path: str | Path, loads: Callable[[str], _T]) -> _T:
    """Parse a file with ``loads``; see :func:`read_file`."""
    return loads(read_file(path))


def parse_record(
    values: Mapping[str, object],
    *,
    record: int | None = None,
    line: int | None = None,
    path: str | None = None,
) -> Omm:
    """The elements of one object from its keyword values (message text).

    Keywords other than those of :data:`~keepout.omm.keywords.KEYWORDS` and
    :data:`~keepout.omm.keywords.FIXED_METADATA` are ignored.

    Parameters
    ----------
    values
        Value of each keyword: its text, or a :class:`Text` with its unit and
        place. Surrounding blanks are ignored.
    record
        Position of the object in its file, for error messages.
    line
        Line of the object in its file, for error messages about a value
        without a line of its own.
    path
        Element path of the object, likewise.

    Raises
    ------
    OmmParseError
        If a keyword is missing or has an invalid value or unit, the metadata
        are not those of SGP4 elements, or the elements are invalid.
    """

    def error(message: str, keyword: str) -> OmmParseError:
        place = values.get(keyword)
        if not isinstance(place, Text):
            place = Text("")
        return OmmParseError(
            message,
            record=record,
            line=line if place.line is None else place.line,
            path=path if place.path is None else place.path,
            keyword=keyword,
        )

    def text_of(keyword: str) -> Text:
        value = values[keyword]
        if isinstance(value, Text):
            return value._replace(text=value.text.strip())
        if isinstance(value, str):
            return Text(value.strip())
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
