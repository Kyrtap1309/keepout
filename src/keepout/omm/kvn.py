"""CCSDS OMM Keyword = Value Notation (KVN) reader and writer.

An OMM in KVN (CCSDS 502.0-B-3, section 4 and 7) is a header that starts
with ``CCSDS_OMM_VERS``, metadata and data, one ``KEYWORD = value [unit]`` per
line. A file may hold several OMMs one after the other, as CelesTrak's KVN
does: each ``CCSDS_OMM_VERS`` line starts the next. Versions 2.0 (CCSDS
502.0-B-2) and 3.0 (502.0-B-3) are read; version 3.0 is written.

Only the SGP4 elements of :data:`keepout.omm.keywords.KEYWORDS` are kept. The
header, comments, ``GM``, spacecraft parameters, covariance and user-defined
parameters are read past; the metadata of
:data:`~keepout.omm.keywords.FIXED_METADATA` must be those of SGP4 elements.
``EPHEMERIS_TYPE`` and ``CLASSIFICATION_TYPE`` may be omitted
(:data:`~keepout.omm.keywords.DEFAULTS`).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Final

from astropy.time import Time

from keepout.errors import OmmParseError
from keepout.omm._io import (
    Text,
    ccsds_values,
    check_version,
    creation_date_text,
    load_file,
    parse_record,
)
from keepout.omm.keywords import DEFAULTS, KEYWORDS, VERSION
from keepout.omm.model import Omm

__all__ = ["dumps", "load", "loads"]

VERSION_KEYWORD: Final = "CCSDS_OMM_VERS"
_ASSIGNMENT: Final = re.compile(r"^(?P<keyword>[A-Z][A-Z0-9_]*)\s*=\s*(?P<value>.*)$")
_UNIT: Final = re.compile(r"^(?P<value>.*?)\s*\[(?P<unit>[^\[\]]*)\]$")
_COMMENT: Final = re.compile(r"^COMMENT(?:\s|$)")
_KEYWORD_WIDTH: Final = len("MEAN_ELEMENT_THEORY")


def _value(keyword: str, text: str, line: int) -> Text:
    spec = KEYWORDS.get(keyword)
    if spec is None or spec.kind == "str":
        # Text values keep brackets: "GOES 9 [P]" is a name, not a unit.
        return Text(text, line=line)
    match = _UNIT.match(text)
    if match is None:
        return Text(text, line=line)
    return Text(match.group("value"), match.group("unit").strip(), line)


def loads(text: str) -> tuple[Omm, ...]:
    """Parse one or more OMMs in KVN.

    Parameters
    ----------
    text
        The messages, each starting with its ``CCSDS_OMM_VERS`` line.

    Returns
    -------
    tuple[Omm, ...]
        The elements of each message, in order (none for blank text).

    Raises
    ------
    OmmParseError
        If a line is not ``KEYWORD = value`` or a comment, a message does not
        start with ``CCSDS_OMM_VERS`` or has an unsupported version, repeats a
        keyword, or its elements are missing or invalid. The error names the
        message (``record``), line and keyword.
    """
    # Each message's first line and values, keyed by keyword.
    messages: list[tuple[int, dict[str, Text]]] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        current = len(messages) or None  # the message being read, 1-based
        assignment = _ASSIGNMENT.match(line)
        keyword = assignment["keyword"] if assignment else None
        if keyword == VERSION_KEYWORD and assignment:
            check_version(
                assignment["value"].strip(), record=len(messages) + 1, line=number
            )
            messages.append((number, {}))
            continue
        if current is None:
            raise OmmParseError(
                f"{VERSION_KEYWORD} must be the first keyword",
                line=number,
                keyword=keyword,
            )
        if _COMMENT.match(line):
            continue
        if assignment is None or keyword is None:
            raise OmmParseError(
                f"expected 'KEYWORD = value', got {line!r}", record=current, line=number
            )
        values = messages[-1][1]
        if keyword in values:
            raise OmmParseError(
                "duplicate keyword", record=current, line=number, keyword=keyword
            )
        values[keyword] = _value(keyword, assignment["value"].strip(), number)
    return tuple(
        parse_record({**DEFAULTS, **values}, record=record, line=start)
        for record, (start, values) in enumerate(messages, start=1)
    )


def load(path: str | Path) -> tuple[Omm, ...]:
    """Read a KVN OMM file (UTF-8, or gzip-compressed UTF-8); see :func:`loads`.

    Parameters
    ----------
    path
        File to read.
    """
    return load_file(path, loads)


def dumps(
    omms: Iterable[Omm],
    *,
    originator: str = "UNKNOWN",
    creation_date: Time | None = None,
) -> str:
    """Write OMMs in KVN (version 3.0), one message per object, with units.

    Messages are separated by a blank line. Reading the text gives the same
    elements back.

    Parameters
    ----------
    omms
        The objects, written in order.
    originator
        ``ORIGINATOR`` of every message: the creating agency or operator.
    creation_date
        ``CREATION_DATE`` of every message; the current time (whole seconds)
        if not given.

    Raises
    ------
    ValueError
        If a value contains a line break, which KVN cannot hold.
    """
    created = creation_date_text(creation_date)
    messages: list[str] = []
    for omm in omms:
        lines = [
            _line(VERSION_KEYWORD, VERSION, None),
            _line("CREATION_DATE", created, None),
            _line("ORIGINATOR", originator, None),
        ]
        group: tuple[str, ...] = ()
        for keyword, value, unit, value_group in ccsds_values(omm):
            if value_group != group:
                lines.append("")
                group = value_group
            lines.append(_line(keyword, value, unit))
        messages.append("\n".join(lines) + "\n")
    return "\n".join(messages)


def _line(keyword: str, value: str, unit: str | None) -> str:
    if len(f"<{value}>".splitlines()) > 1:
        raise ValueError(f"{keyword}: KVN cannot hold a line break in {value!r}")
    line = f"{keyword:<{_KEYWORD_WIDTH}} = {value}"
    return f"{line} [{unit}]" if unit is not None else line.rstrip()
