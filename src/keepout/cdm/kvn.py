"""CDM Keyword = Value Notation (KVN) reader and writer."""

from __future__ import annotations

import re
from pathlib import Path

from keepout.cdm._io import read_text, write_text
from keepout.cdm.keywords import KEYWORDS, SectionKind
from keepout.cdm.model import (
    COMMENT,
    Cdm,
    DuplicateKeyword,
    Field,
    Section,
    format_field,
    parse_field,
)
from keepout.errors import CdmParseError, CdmWriteError

_ASSIGNMENT = re.compile(r"^(?P<keyword>[A-Z][A-Z0-9_]*)\s*=\s*(?P<value>.*)$")
_UNIT = re.compile(r"^(?P<value>.*?)\s*\[(?P<unit>[^\[\]]*)\]$")
_COMMENT = re.compile(r"^COMMENT(?:\s+(?P<text>.*))?$")
_KEYWORD_WIDTH = 30
_LAYOUT: list[SectionKind] = ["header", "relative", "segment", "segment"]


def _split_unit(text: str) -> tuple[str, str | None]:
    match = _UNIT.match(text)
    if match is None:
        return text, None
    return match.group("value").strip(), match.group("unit").strip()


def _field(kind: SectionKind, keyword: str, text: str, line: int) -> Field:
    spec = KEYWORDS[kind].get(keyword)
    value, unit = (text, None) if spec and spec.kind == "str" else _split_unit(text)
    try:
        return parse_field(kind, keyword, value, unit)
    except ValueError as exc:
        raise CdmParseError(str(exc), line=line, keyword=keyword) from exc


def _opens(kinds: list[SectionKind], keyword: str) -> SectionKind | None:
    """The section a keyword starts, if it starts one."""
    if keyword == "OBJECT":
        return "segment"
    if kinds[-1] == "header" and keyword in KEYWORDS["relative"]:
        return "relative"
    return None


def _section(kind: SectionKind, fields: list[Field], lines: list[int]) -> Section:
    try:
        return Section(kind, tuple(fields))
    except DuplicateKeyword as exc:
        raise CdmParseError(
            "duplicate keyword", line=lines[exc.index], keyword=exc.keyword
        ) from exc
    except ValueError as exc:
        raise CdmParseError(str(exc)) from exc


def loads(text: str) -> Cdm:
    """Parse a CDM in KVN.

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
    kinds: list[SectionKind] = []
    sections: list[list[Field]] = []
    lines: list[list[int]] = []
    pending: list[tuple[Field, int]] = []

    def append(field: Field, number: int) -> None:
        sections[-1].append(field)
        lines[-1].append(number)

    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        comment = _COMMENT.match(line)
        if comment is not None:
            if not sections:
                raise CdmParseError(
                    "CCSDS_CDM_VERS must be the first keyword", line=number
                )
            pending.append(
                (Field(COMMENT, (comment.group("text") or "").strip()), number)
            )
            continue
        assignment = _ASSIGNMENT.match(line)
        if assignment is None:
            raise CdmParseError(
                f"expected 'KEYWORD = value', got {line!r}", line=number
            )
        keyword = assignment.group("keyword")
        if not sections:
            if keyword != "CCSDS_CDM_VERS":
                raise CdmParseError(
                    "CCSDS_CDM_VERS must be the first keyword",
                    line=number,
                    keyword=keyword,
                )
            opened: SectionKind | None = "header"
        else:
            opened = _opens(kinds, keyword)
        if opened == "segment":
            if kinds[-1] == "header":
                raise CdmParseError(
                    "object section before relative metadata (TCA)",
                    line=number,
                    keyword=keyword,
                )
            if kinds.count("segment") == 2:
                raise CdmParseError(
                    "more than two object sections", line=number, keyword=keyword
                )
        if opened is not None:
            kinds.append(opened)
            sections.append([])
            lines.append([])
        for field, at in pending:
            append(field, at)
        pending.clear()
        value = assignment.group("value").strip()
        append(_field(kinds[-1], keyword, value, number), number)
    for field, at in pending:
        append(field, at)
    if kinds != _LAYOUT:
        raise CdmParseError(
            "a CDM needs a header, relative metadata (starting at TCA) and "
            "OBJECT1 and OBJECT2 sections"
        )
    header, relative, first, second = (
        _section(kind, fields, numbers)
        for kind, fields, numbers in zip(kinds, sections, lines, strict=True)
    )
    try:
        return Cdm(header, relative, (first, second))
    except ValueError as exc:
        raise CdmParseError(str(exc)) from exc


def dumps(cdm: Cdm) -> str:
    """Write a CDM as KVN, with CCSDS units.

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
            if len(text.splitlines()) > 1:
                raise CdmWriteError(
                    f"{field.keyword}: KVN cannot hold a multi-line value {text!r}"
                )
            line = f"{field.keyword:<{_KEYWORD_WIDTH}} = {text}"
            lines.append(f"{line} [{unit}]" if unit is not None else line.rstrip())
    return "\n".join(lines) + "\n"


def load(path: str | Path) -> Cdm:
    """Read a KVN CDM file (UTF-8; a byte-order mark is ignored).

    Parameters
    ----------
    path
        File to read.
    """
    return loads(read_text(path))


def dump(cdm: Cdm, path: str | Path) -> None:
    """Write a CDM to a KVN file (UTF-8).

    Parameters
    ----------
    cdm
        The message.
    path
        File to write.
    """
    write_text(path, dumps(cdm))
