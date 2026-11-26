"""Shared by the CDM readers and writers: the family, files, values and
parse errors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from keepout.ccsds.message import COMMENT, Family, Located
from keepout.cdm.model import Cdm, DuplicateKeyword, Field, Section, parse_field
from keepout.errors import CdmParseError, CdmWriteError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from keepout.cdm.keywords import SectionKind

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
    """A section being read: its fields so far, with where each was read.

    :class:`~keepout.cdm.model.Section` checks the fields when built; a
    repeated keyword is reported at its second value.

    Parameters
    ----------
    kind
        Kind of the section.
    at
        Where the section is in the message, for the error of an invalid
        section; no location if ``None``.
    """

    __slots__ = ("_fields", "_places", "at", "kind")

    def __init__(self, kind: SectionKind, at: Located | None = None) -> None:
        self.kind: Final = kind
        self.at: Final = at
        self._fields: list[Field] = []
        self._places: list[Located] = []

    def add(self, keyword: str, value: Located) -> None:
        """The field ``keyword``'s ``value`` gives, appended.

        Raises
        ------
        CdmParseError
            At ``value``, if the value is invalid.
        """
        self._fields.append(field_of(self.kind, keyword, value))
        self._places.append(value)

    def build(self) -> Section:
        """The section of the fields added.

        Raises
        ------
        CdmParseError
            At the repeat of a keyword, or at :attr:`at` if the section is
            otherwise invalid.
        """
        try:
            return Section(self.kind, tuple(self._fields))
        except DuplicateKeyword as exc:
            where = [
                place
                for field, place in zip(self._fields, self._places, strict=True)
                if field.keyword == exc.keyword
            ][1]
            raise CdmParseError(
                "duplicate keyword",
                line=where.line,
                path=where.path,
                keyword=exc.keyword,
            ) from exc
        except ValueError as exc:
            where = self.at or Located("")
            raise CdmParseError(str(exc), line=where.line, path=where.path) from exc


def build_cdm(
    header: Section, relative: Section, object_sections: Sequence[Section]
) -> Cdm:
    """The message of its sections.

    Raises
    ------
    CdmParseError
        If the sections do not make a valid CDM.
    """
    first, second = object_sections
    try:
        return Cdm(header, relative, (first, second))
    except ValueError as exc:
        raise CdmParseError(str(exc)) from exc
