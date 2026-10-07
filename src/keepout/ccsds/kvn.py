"""CCSDS Keyword = Value Notation (KVN) lines: reading and writing.

A KVN message is one ``KEYWORD = value [unit]`` per line, with ``COMMENT``
lines and blank lines between them (CCSDS 502.0-B-3 section 7, 508.0-B-1
section 4). The message readers and writers (:mod:`keepout.cdm.kvn`,
:mod:`keepout.omm.kvn`) give the lines their meaning.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Final

__all__ = ["assignment", "comment", "format_line", "split_unit"]

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


def format_line(
    keyword: str,
    value: str,
    unit: str | None,
    *,
    width: int,
    error: Callable[[str], Exception],
) -> str:
    """A ``KEYWORD = value [unit]`` line, the keyword padded to ``width``.

    Parameters
    ----------
    keyword
        The keyword.
    value
        Its value as message text.
    unit
        Its unit, or None to write none.
    width
        Width the keyword is padded to, aligning the ``=`` of a message.
    error
        Makes the exception raised for a value KVN cannot hold, from its
        message: the writer error of the message family.

    Raises
    ------
    Exception
        ``error(...)`` if the value contains a line break (KVN has no
        escaping).
    """
    # The brackets make a leading or trailing line break count as well.
    if len(f"<{value}>".splitlines()) > 1:
        raise error(f"{keyword}: KVN cannot hold a line break in {value!r}")
    line = f"{keyword:<{width}} = {value}"
    return f"{line} [{unit}]" if unit is not None else line.rstrip()
