"""CCSDS Keyword = Value Notation (KVN) lines: reading and writing.

A KVN message is one ``KEYWORD = value [unit]`` per line, with ``COMMENT``
lines and blank lines between them (CCSDS 502.0-B-3 section 7, 508.0-B-1
section 4). The message readers and writers (:mod:`keepout.cdm.kvn`,
:mod:`keepout.omm.kvn`) give the lines their meaning.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

from keepout.ccsds.message import Located

if TYPE_CHECKING:
    from keepout.ccsds.keywords import Keyword
    from keepout.ccsds.message import Family

__all__ = ["assignment", "comment", "format_line", "located_value", "split_unit"]

_ASSIGNMENT: Final = re.compile(r"^(?P<keyword>[A-Z][A-Z0-9_]*)\s*=\s*(?P<value>.*)$")
_UNIT: Final = re.compile(r"^(?P<value>.*?)\s*\[(?P<unit>[^\[\]]*)\]$")
_COMMENT: Final = re.compile(r"^COMMENT(?:\s+(?P<text>.*))?$")


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
    """
    match = _UNIT.match(text)
    if match is None:
        return text, None
    return match["value"].strip(), match["unit"].strip()


def located_value(
    spec: Keyword | None, text: str, line: int, *, unknown_is_text: bool
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
    unknown_is_text
        Whether a keyword without a table entry has a text value.
    """
    if (spec is None and unknown_is_text) or (spec is not None and spec.kind == "str"):
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

    Raises
    ------
    KeepoutError
        The family's write error if the value contains a line break (KVN
        has no escaping).
    """
    # The brackets make a leading or trailing line break count as well.
    if len(f"<{value}>".splitlines()) > 1:
        raise family.write_error(
            f"{keyword}: KVN cannot hold a line break in {value!r}"
        )
    line = f"{keyword:<{family.keyword_width}} = {value}"
    return f"{line} [{unit}]" if unit is not None else line.rstrip()
