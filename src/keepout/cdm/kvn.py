"""CDM Keyword = Value Notation (KVN) reader and writer."""

from __future__ import annotations

import re
from pathlib import Path

from keepout.cdm.keywords import KEYWORDS, SectionKind
from keepout.cdm.model import COMMENT, Cdm, Field, Section, format_field, parse_field
from keepout.errors import CdmParseError

_ASSIGNMENT = re.compile(r"^(?P<keyword>[A-Z0-9_]+)\s*=\s*(?P<value>.*)$")
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


def loads(text: str) -> Cdm:
    """Parse a CDM in KVN.

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
    seen: set[str] = set()

    def start(kind: SectionKind) -> None:
        kinds.append(kind)
        sections.append([])
        seen.clear()

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
            sections[-1].append(Field(COMMENT, (comment.group("text") or "").strip()))
            continue
        assignment = _ASSIGNMENT.match(line)
        if assignment is None:
            raise CdmParseError(
                f"expected 'KEYWORD = value', got {line!r}", line=number
            )
        keyword = assignment.group("keyword")
        value = assignment.group("value").strip()
        if not sections:
            if keyword != "CCSDS_CDM_VERS":
                raise CdmParseError(
                    "CCSDS_CDM_VERS must be the first keyword",
                    line=number,
                    keyword=keyword,
                )
            start("header")
        elif keyword == "OBJECT":
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
            start("segment")
        elif kinds[-1] == "header" and keyword in KEYWORDS["relative"]:
            start("relative")
        if keyword in seen:
            raise CdmParseError("duplicate keyword", line=number, keyword=keyword)
        seen.add(keyword)
        sections[-1].append(_field(kinds[-1], keyword, value, number))
    if kinds != _LAYOUT:
        raise CdmParseError(
            "a CDM needs a header, relative metadata (starting at TCA) and "
            "OBJECT1 and OBJECT2 sections"
        )
    try:
        return Cdm(
            Section("header", tuple(sections[0])),
            Section("relative", tuple(sections[1])),
            (
                Section("segment", tuple(sections[2])),
                Section("segment", tuple(sections[3])),
            ),
        )
    except ValueError as exc:
        raise CdmParseError(str(exc)) from exc


def dumps(cdm: Cdm) -> str:
    """Write a CDM as KVN, with CCSDS units.

    Parameters
    ----------
    cdm
        The message.
    """
    lines: list[str] = []
    for section in (cdm.header, cdm.relative, *cdm.segments):
        for field in section.fields:
            if field.keyword == COMMENT:
                lines.append(f"COMMENT {field.value}".rstrip())
                continue
            text, unit = format_field(section.kind, field)
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
    return loads(Path(path).read_text(encoding="utf-8-sig"))


def dump(cdm: Cdm, path: str | Path) -> None:
    """Write a CDM to a KVN file (UTF-8).

    Parameters
    ----------
    cdm
        The message.
    path
        File to write.
    """
    Path(path).write_text(dumps(cdm), encoding="utf-8")
