"""CDM Keyword = Value Notation (KVN) reader and writer."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from keepout.ccsds.kvn import assignment, comment, format_line, located_value
from keepout.ccsds.message import COMMENT, KeywordValues, Located
from keepout.cdm._io import (
    CDM,
    build_cdm,
    build_section,
    dump_file,
    field_of,
    load_file,
)
from keepout.cdm.keywords import KEYWORDS, VERSION_KEYWORD, SectionKind
from keepout.cdm.model import Cdm, Field, Section, format_field
from keepout.errors import CdmParseError

__all__ = ["dump", "dumps", "load", "loads"]

_LAYOUT: Final[tuple[SectionKind, ...]] = ("header", "relative", "segment", "segment")


@dataclass
class _Building:
    """A section being read: its fields, its keywords' values."""

    kind: SectionKind
    fields: list[Field] = dataclasses.field(default_factory=list)
    seen_keywords: KeywordValues = dataclasses.field(
        default_factory=lambda: KeywordValues(CDM)
    )

    def add(self, keyword: str, value: Located) -> None:
        self.seen_keywords.add(keyword, value)
        self.fields.append(field_of(self.kind, keyword, value))

    def build(self) -> Section:
        return build_section(self.kind, self.fields)


def _opened(kinds: list[SectionKind], keyword: str, line: int) -> SectionKind | None:
    """The section ``keyword`` opens after sections ``kinds``, if it opens one.

    ``CCSDS_CDM_VERS`` opens the header, the first relative-metadata keyword
    (``TCA``) the relative metadata and ``OBJECT`` an object section.

    Raises
    ------
    CdmParseError
        If the first keyword is not ``CCSDS_CDM_VERS``, an object section
        comes before the relative metadata, or a third object section opens.
    """
    if not kinds:
        if keyword != VERSION_KEYWORD:
            raise CdmParseError(
                f"{VERSION_KEYWORD} must be the first keyword",
                line=line,
                keyword=keyword,
            )
        return "header"
    if keyword != "OBJECT":
        if kinds[-1] == "header" and keyword in KEYWORDS["relative"]:
            return "relative"
        return None
    if kinds[-1] == "header":
        raise CdmParseError(
            "object section before relative metadata (TCA)",
            line=line,
            keyword=keyword,
        )
    if kinds.count("segment") == 2:
        raise CdmParseError("more than two object sections", line=line, keyword=keyword)
    return "segment"


def loads(text: str) -> Cdm:
    """The CDM in KVN ``text``.

    Comments go to the section of the keyword that follows them, so comments
    just before ``TCA`` or ``OBJECT`` open the new section.

    Parameters
    ----------
    text
        The message.

    Raises
    ------
    CdmParseError
        If the message is malformed; the error names the line and keyword.
    """
    building: list[_Building] = []
    pending: list[Located] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        remark = comment(line)
        if remark is not None:
            if not building:
                raise CdmParseError(
                    f"{VERSION_KEYWORD} must be the first keyword", line=number
                )
            pending.append(Located(remark, line=number))
            continue
        assigned = assignment(line)
        if assigned is None:
            raise CdmParseError(
                f"expected 'KEYWORD = value', got {line!r}", line=number
            )
        keyword, value = assigned
        opened = _opened([b.kind for b in building], keyword, number)
        if opened is not None:
            building.append(_Building(opened))
        current = building[-1]
        for held in pending:
            current.add(COMMENT, held)
        pending.clear()
        spec = KEYWORDS[current.kind].get(keyword)
        current.add(keyword, located_value(spec, value, number, unknown_is_text=False))
    for held in pending:
        building[-1].add(COMMENT, held)
    if tuple(b.kind for b in building) != _LAYOUT:
        raise CdmParseError(
            "a CDM needs a header, relative metadata (starting at TCA) and "
            "OBJECT1 and OBJECT2 sections"
        )
    header, relative, *segments = (b.build() for b in building)
    return build_cdm(header, relative, segments)


def dumps(cdm: Cdm) -> str:
    """A CDM as KVN text, with CCSDS units.

    A multi-line comment is written as one ``COMMENT`` line per line.

    Parameters
    ----------
    cdm
        The message.

    Raises
    ------
    CdmWriteError
        If a value other than a comment contains a line break (KVN has no
        escaping).
    """
    lines: list[str] = []
    for section in (cdm.header, cdm.relative, *cdm.segments):
        for field in section.fields:
            if field.keyword == COMMENT:
                lines.extend(
                    f"COMMENT {text}".rstrip()
                    for text in str(field.value).splitlines() or [""]
                )
                continue
            text, unit = format_field(section.kind, field)
            lines.append(format_line(field.keyword, text, unit, CDM))
    return "\n".join(lines) + "\n"


def load(path: str | Path) -> Cdm:
    """The CDM in a KVN file (UTF-8, gzip-compressed or not; a
    byte-order mark is ignored).

    Parameters
    ----------
    path
        File to read.
    """
    return load_file(path, loads)


def dump(cdm: Cdm, path: str | Path) -> None:
    """A CDM written to a KVN file (UTF-8).

    Parameters
    ----------
    cdm
        The message.
    path
        File to write.
    """
    dump_file(cdm, path, dumps)
