"""CCSDS Keyword = Value Notation (KVN) lines: reading and writing.

A KVN message is one ``KEYWORD = value [unit]`` per line, with ``COMMENT``
lines and blank lines between them (CCSDS 502.0-B-3 section 7, 508.0-B-1
section 4). The message readers and writers (:mod:`keepout.cdm.kvn`,
:mod:`keepout.omm.kvn`) give the lines their meaning.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final, NamedTuple

from keepout.ccsds.message import Located

if TYPE_CHECKING:
    from collections.abc import Iterator

    from keepout.ccsds.keywords import Keyword
    from keepout.ccsds.message import Family

__all__ = [
    "KEYWORD",
    "KvnLine",
    "assignment",
    "comment",
    "format_line",
    "located_value",
    "read_lines",
    "require_started",
    "split_unit",
]

KEYWORD: Final = re.compile(r"[A-Z][A-Z0-9_]*")
"""What a KVN keyword looks like (also a valid XML element name)."""

_ASSIGNMENT: Final = re.compile(
    rf"^(?P<keyword>{KEYWORD.pattern})\s*=\s*(?P<value>.*)$"
)
_UNIT: Final = re.compile(r"^(?P<value>.*?)\s*\[(?P<unit>[^\[\]]*)\]$")
_COMMENT: Final = re.compile(r"^COMMENT(?:\s+(?P<text>.*))?$")
_LINE_BREAK: Final = re.compile("[\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029]")
"""A character :meth:`str.splitlines` breaks a line at."""


def assignment(line: str) -> tuple[str, str] | None:
    """Keyword and value (with its unit, if any) of a ``KEYWORD = value`` line.

    Parameters
    ----------
    line
        One line of a message, without surrounding blanks.

    Returns
    -------
    tuple[str, str] | None
        The keyword and the value without surrounding blanks, or None if the
        line is not an assignment.
    """
    match = _ASSIGNMENT.match(line)
    if match is None:
        return None
    return match["keyword"], match["value"].strip()


def comment(line: str) -> str | None:
    """Text of a ``COMMENT`` line, or None if the line is not a comment.

    Parameters
    ----------
    line
        One line of a message, without surrounding blanks.

    Returns
    -------
    str or None
        The comment's text, or None.
    """
    match = _COMMENT.match(line)
    if match is None:
        return None
    return (match["text"] or "").strip()


def split_unit(text: str) -> tuple[str, str | None]:
    """A value and its trailing ``[unit]``, if it has one.

    Text values must not be split: ``GOES 9 [P]`` is a name, not a unit.

    Parameters
    ----------
    text
        The value of an assignment.

    Returns
    -------
    tuple[str, str | None]
        The value's text and its unit, or None if it has none.
    """
    match = _UNIT.match(text)
    if match is None:
        return text, None
    return match["value"].strip(), match["unit"].strip()


class KvnLine(NamedTuple):
    """A line of a KVN message that is not blank."""

    number: int
    """1-based line number."""
    keyword: str | None
    """The keyword of an assignment; None for a ``COMMENT`` line."""
    text: str
    """The value of an assignment (with its unit, if any), or the comment text."""


def read_lines(text: str, family: Family) -> Iterator[KvnLine]:
    """The comments and assignments of KVN ``text``, in order; blank lines
    are skipped.

    Parameters
    ----------
    text
        The message text.
    family
        The message family, whose parse error a malformed line raises.

    Yields
    ------
    KvnLine
        Each comment and assignment, in order.

    Raises
    ------
    CcsdsParseError
        The family's, for a line that is neither a comment nor
        ``KEYWORD = value``.
    """
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        remark = comment(line)
        if remark is not None:
            yield KvnLine(number, None, remark)
            continue
        assigned = assignment(line)
        if assigned is None:
            raise family.parse_error(
                f"expected 'KEYWORD = value', got {line!r}", line=number
            )
        yield KvnLine(number, *assigned)


def require_started(
    line: KvnLine, family: Family, version_keyword: str, *, started: bool
) -> None:
    """A check that ``line`` comes after a message's first keyword,
    ``version_keyword``.

    Parameters
    ----------
    line
        The line read.
    family
        The message family, whose parse error a misplaced line raises.
    version_keyword
        The keyword that starts each message of the family.
    started
        Whether a message has started before ``line``.

    Raises
    ------
    CcsdsParseError
        The family's, at ``line``, if no message has started.
    """
    if not started:
        raise family.parse_error(
            f"{version_keyword} must be the first keyword",
            line=line.number,
            keyword=line.keyword,
        )


def located_value(
    spec: Keyword | None, text: str, line: int, family: Family
) -> Located:
    """The value of an assignment, split from its unit unless it is text.

    Text values keep brackets: ``GOES 9 [P]`` is a name, not a unit.

    Parameters
    ----------
    spec
        Table entry of the keyword, or None if the table has none.
    text
        The value of the assignment (see :func:`assignment`).
    line
        Line number of the assignment.
    family
        The message family; a keyword without a table entry has a text
        value if its :attr:`~keepout.ccsds.message.Family.unknown_is_text`.

    Returns
    -------
    Located
        The value's text and unit, with where it was read.
    """
    if (spec is None and family.unknown_is_text) or (
        spec is not None and spec.kind == "str"
    ):
        return Located(text, None, line)
    value, unit = split_unit(text)
    return Located(value, unit, line)


def format_line(keyword: str, value: str, unit: str | None, family: Family) -> str:
    """A ``KEYWORD = value [unit]`` line, the keyword padded to the family's
    :attr:`~keepout.ccsds.message.Family.keyword_width`.

    Parameters
    ----------
    keyword
        The keyword.
    value
        Its value as message text.
    unit
        Its unit, or None to write none.
    family
        The message family.

    Returns
    -------
    str
        The KVN line, without a line ending.

    Raises
    ------
    KeepoutError
        The family's write error if the keyword is not one KVN can hold
        (:data:`KEYWORD`), or the value contains a line break (KVN has no
        escaping).
    """
    if not KEYWORD.fullmatch(keyword):
        raise family.write_error(f"KVN cannot hold the keyword {keyword!r}")
    if _LINE_BREAK.search(value):
        raise family.write_error(
            f"{keyword}: KVN cannot hold a line break in {value!r}"
        )
    line = f"{keyword:<{family.keyword_width}} = {value}"
    return f"{line} [{unit}]" if unit is not None else line.rstrip()
