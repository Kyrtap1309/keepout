"""CDM Keyword = Value Notation (KVN) reader and writer."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from keepout.ccsds.kvn import format_line, located_value, read_lines, require_started
from keepout.ccsds.message import COMMENT, Located
from keepout.cdm._io import CDM, SectionBuilder, build_cdm
from keepout.cdm.keywords import KEYWORDS, VERSION_KEYWORD, SectionKind
from keepout.cdm.model import Cdm, format_field
from keepout.errors import CdmParseError

__all__ = ["dump", "dumps", "load", "loads"]

_LAYOUT: Final[tuple[SectionKind, ...]] = ("header", "relative", "segment", "segment")


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
    building: list[SectionBuilder] = []
    pending: list[Located] = []
    for line in read_lines(text, CDM):
        if line.keyword is None:
            require_started(line, CDM, VERSION_KEYWORD, started=bool(building))
            pending.append(Located(line.text, line=line.number))
            continue
        opened = _opened([b.kind for b in building], line.keyword, line.number)
        if opened is not None:
            building.append(SectionBuilder(opened))
        current = building[-1]
        for held in pending:
            current.add(COMMENT, held)
        pending.clear()
        spec = KEYWORDS[current.kind].get(line.keyword)
        current.add(line.keyword, located_value(spec, line.text, line.number, CDM))
    for held in pending:
        building[-1].add(COMMENT, held)
    if tuple(b.kind for b in building) != _LAYOUT:
        raise CdmParseError(
            "a CDM needs a header, relative metadata (starting at TCA) and "
            "OBJECT1 and OBJECT2 sections"
        )
    header, relative, *object_sections = (b.build() for b in building)
    return build_cdm(header, relative, object_sections)


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
    for section in (cdm.header, cdm.relative, *cdm.object_sections):
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
    return CDM.load(path, loads)


def dump(cdm: Cdm, path: str | Path) -> None:
    """A CDM written to a KVN file (UTF-8).

    Parameters
    ----------
    cdm
        The message.
    path
        File to write.
    """
    Path(path).write_text(dumps(cdm), encoding="utf-8")
