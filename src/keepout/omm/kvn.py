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

from collections.abc import Iterable
from pathlib import Path

from astropy.time import Time

from keepout.ccsds.kvn import assignment, comment, format_line, located_value
from keepout.ccsds.message import KeywordValues
from keepout.errors import OmmParseError
from keepout.omm._io import (
    OMM,
    ccsds_values,
    check_version,
    creation_date_text,
    load_file,
    parse_record,
)
from keepout.omm.keywords import DEFAULTS, KEYWORDS, VERSION, VERSION_KEYWORD
from keepout.omm.model import Omm

__all__ = ["dumps", "load", "loads"]


def loads(text: str) -> tuple[Omm, ...]:
    """The one or more OMMs in KVN ``text``.

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
    messages: list[tuple[int, KeywordValues]] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        assigned = assignment(line)
        keyword = assigned[0] if assigned else None
        if assigned and keyword == VERSION_KEYWORD:
            record = len(messages) + 1
            check_version(assigned[1], record=record, line=number)
            messages.append((number, KeywordValues(OMM, record=record)))
            continue
        if not messages:
            raise OmmParseError(
                f"{VERSION_KEYWORD} must be the first keyword",
                line=number,
                keyword=keyword,
            )
        if comment(line) is not None:
            continue
        if assigned is None:
            raise OmmParseError(
                f"expected 'KEYWORD = value', got {line!r}",
                record=len(messages),  # the message being read, 1-based
                line=number,
            )
        keyword, value = assigned
        spec = KEYWORDS.get(keyword)
        messages[-1][1].add(
            keyword, located_value(spec, value, number, unknown_is_text=True)
        )
    return tuple(
        parse_record({**DEFAULTS, **found.values}, record=record, line=start)
        for record, (start, found) in enumerate(messages, start=1)
    )


def load(path: str | Path) -> tuple[Omm, ...]:
    """The OMMs in a KVN file (UTF-8, or gzip-compressed UTF-8); see :func:`loads`.

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
    """OMMs as KVN (version 3.0), one message per object, with units.

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
    OmmWriteError
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
    return format_line(keyword, value, unit, OMM)
