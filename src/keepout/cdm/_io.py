"""Shared by the CDM readers and writers: the family, files, values and
parse errors."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

from keepout.ccsds import files
from keepout.ccsds.message import COMMENT, Family, Located
from keepout.cdm.keywords import SectionKind
from keepout.cdm.model import Cdm, Field, Section, parse_field
from keepout.errors import CdmParseError, CdmWriteError

CDM: Final = Family("CDM", CdmParseError, CdmWriteError, keyword_width=30)
"""The CDM message family."""


def load_file(path: str | Path, loads: Callable[[str], Cdm]) -> Cdm:
    """A message file parsed by ``loads``; see :func:`keepout.ccsds.files.read_file`.

    Raises
    ------
    CdmParseError
        If the file is broken gzip data or not UTF-8 text.
    """
    return files.load_file(path, loads, CDM)


def dump_file(cdm: Cdm, path: str | Path, dumps: Callable[[Cdm], str]) -> None:
    """A message formatted by ``dumps``, written as UTF-8."""
    Path(path).write_text(dumps(cdm), encoding="utf-8")


def field_of(kind: SectionKind, keyword: str, value: Located) -> Field:
    """The field of a section of ``kind`` that ``keyword``'s ``value`` gives.

    Raises
    ------
    CdmParseError
        At ``value``, if it is not a valid value of ``keyword``.
    """
    if keyword == COMMENT:
        return Field(COMMENT, value.text)
    try:
        return parse_field(kind, keyword, value.text, value.unit)
    except ValueError as exc:
        raise CdmParseError(
            str(exc), line=value.line, path=value.path, keyword=keyword
        ) from exc


def build_section(
    kind: SectionKind, fields: Sequence[Field], at: Located | None = None
) -> Section:
    """The section of ``fields``; their keywords are already known not to
    repeat (:class:`~keepout.ccsds.message.KeywordValues`).

    Raises
    ------
    CdmParseError
        At ``at`` if the section is invalid (no location if ``at`` is
        ``None``).
    """
    try:
        return Section(kind, tuple(fields))
    except ValueError as exc:
        where = at or Located("")
        raise CdmParseError(str(exc), line=where.line, path=where.path) from exc


def build_cdm(header: Section, relative: Section, segments: Sequence[Section]) -> Cdm:
    """The message of its sections.

    Raises
    ------
    CdmParseError
        If the sections do not make a valid CDM.
    """
    first, second = segments
    try:
        return Cdm(header, relative, (first, second))
    except ValueError as exc:
        raise CdmParseError(str(exc)) from exc
