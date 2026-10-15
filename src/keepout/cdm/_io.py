"""Shared by the CDM readers and writers: file encoding and parse errors."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NamedTuple

from keepout.ccsds import files
from keepout.cdm.keywords import SectionKind
from keepout.cdm.model import Cdm, DuplicateKeyword, Field, Section
from keepout.errors import CdmParseError


def load_file(path: str | Path, loads: Callable[[str], Cdm]) -> Cdm:
    """Parse a message file with ``loads``; see :func:`keepout.ccsds.files.read_file`.

    Raises
    ------
    CdmParseError
        If the file is broken gzip data or not UTF-8 text.
    """
    return files.load_file(path, loads, CdmParseError)


def dump_file(cdm: Cdm, path: str | Path, dumps: Callable[[Cdm], str]) -> None:
    """Write a message formatted by ``dumps`` as UTF-8."""
    Path(path).write_text(dumps(cdm), encoding="utf-8")


class Location(NamedTuple):
    """Where something is in a message: a KVN line or an XML element path."""

    line: int | None = None
    path: str | None = None


def build_section(
    kind: SectionKind,
    fields: Sequence[Field],
    locations: Sequence[Location],
    at: Location | None = None,
) -> Section:
    """The section of ``fields``, read at ``locations`` (one per field).

    Raises
    ------
    CdmParseError
        Where the section repeats a keyword, or at ``at`` if it is otherwise
        invalid (no location if ``at`` is ``None``).
    """
    try:
        return Section(kind, tuple(fields))
    except DuplicateKeyword as exc:
        where = locations[exc.index]
        raise CdmParseError(
            "duplicate keyword", line=where.line, path=where.path, keyword=exc.keyword
        ) from exc
    except ValueError as exc:
        where = at or Location()
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
