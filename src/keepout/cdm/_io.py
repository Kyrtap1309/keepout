"""Shared by the CDM readers and writers: the family, files, values and
parse errors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from keepout.ccsds.message import COMMENT, Family, KeywordValues, Located
from keepout.cdm.keywords import SectionKind
from keepout.cdm.model import Cdm, Field, Section, parse_field
from keepout.errors import CdmParseError, CdmWriteError

CDM: Final = Family(
    "CDM", CdmParseError, CdmWriteError, keyword_width=30, unknown_is_text=False
)
"""The CDM message family."""


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


class SectionBuilder:
    """A section being read: its fields so far, each keyword checked not to
    repeat as it is added (:class:`~keepout.ccsds.message.KeywordValues`).

    Parameters
    ----------
    kind
        Kind of the section.
    at
        Where the section is in the message, for the error of an invalid
        section; no location if ``None``.
    """

    __slots__ = ("_fields", "_seen", "at", "kind")

    def __init__(self, kind: SectionKind, at: Located | None = None) -> None:
        self.kind: Final = kind
        self.at: Final = at
        self._fields: list[Field] = []
        self._seen = KeywordValues(CDM)

    def add(self, keyword: str, value: Located) -> None:
        """The field ``keyword``'s ``value`` gives, appended.

        Raises
        ------
        CdmParseError
            At ``value``, if ``keyword`` repeats or the value is invalid.
        """
        self._seen.add(keyword, value)
        self._fields.append(field_of(self.kind, keyword, value))

    def build(self) -> Section:
        """The section of the fields added.

        Raises
        ------
        CdmParseError
            At :attr:`at`, if the section is invalid.
        """
        try:
            return Section(self.kind, tuple(self._fields))
        except ValueError as exc:
            where = self.at or Located("")
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
