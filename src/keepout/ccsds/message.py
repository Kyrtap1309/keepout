"""What the readers of every CCSDS message family share: the family's
errors, values with their place in the message, and keyword uniqueness.

A message family (CDM, OMM) is described once by a :class:`Family`, which
the helpers of :mod:`keepout.ccsds.files`, :mod:`keepout.ccsds.kvn` and
:mod:`keepout.ccsds.xml` take to raise the family's errors. A reader gives
each keyword value as a :class:`Located` (its text, unit, line or element
path) and collects a message part in :class:`KeywordValues`, which refuses
a repeated keyword the same way in every family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, NamedTuple, TypeVar

from keepout.ccsds.files import read_file

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from keepout.errors import CcsdsParseError, KeepoutError

__all__ = [
    "COMMENT",
    "UNKNOWN",
    "USER_DEFINED",
    "USER_DEFINED_PREFIX",
    "Family",
    "KeywordValues",
    "Located",
]

COMMENT: Final = "COMMENT"
"""The keyword of a comment; it may repeat."""

UNKNOWN: Final = "UNKNOWN"
"""The value CCSDS gives a text keyword that is not known, such as the
``OBJECT_NAME`` or ``OBJECT_ID`` an element set lacks, or an
``ORIGINATOR`` not stated."""

USER_DEFINED: Final = "USER_DEFINED"
"""The NDM/XML element of a user-defined parameter, named by its
``parameter`` attribute."""

USER_DEFINED_PREFIX: Final = "USER_DEFINED_"
"""Start of the keyword of a user-defined parameter: ``USER_DEFINED_NAME``
in KVN, ``<USER_DEFINED parameter="NAME">`` in XML."""

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class Family:
    """A CCSDS message family, as the shared readers and writers need it.

    Parameters
    ----------
    name
        The message type, for errors: ``CDM``, ``OMM``.
    parse_error
        Raised for a message that cannot be read.
    write_error
        Raised for a value a format cannot hold.
    keyword_width
        Width KVN keywords are padded to, aligning the ``=`` of a message.
    unknown_is_text
        Whether a KVN keyword the family's table does not know has a text
        value (kept whole, brackets included) rather than a ``value [unit]``.
    """

    name: str
    parse_error: type[CcsdsParseError]
    write_error: type[KeepoutError]
    keyword_width: int
    unknown_is_text: bool

    def load(self, path: str | Path, loads: Callable[[str], _T]) -> _T:
        """The message file ``path`` parsed by ``loads``; see
        :func:`keepout.ccsds.files.read_file`.

        Raises
        ------
        CcsdsParseError
            The family's, if the file is broken gzip data or not UTF-8 text.
        """
        return loads(read_file(path, self))


class Located(NamedTuple):
    """A keyword's value as written in a message, with its unit and place."""

    text: str
    """The value, without its unit."""
    unit: str | None = None
    """The unit given with the value (KVN ``[unit]``, XML ``units``), if any."""
    line: int | None = None
    """Line of the value in a text format."""
    path: str | None = None
    """Element path of the value in an XML message."""


class KeywordValues:
    """The values of one part of a message by keyword, as a reader finds
    them; a keyword other than ``COMMENT`` may come only once.

    Parameters
    ----------
    family
        The message family, whose parse error a repeat raises.
    record
        Position of the message in its document, for errors.
    """

    __slots__ = ("_values", "family", "record")

    def __init__(self, family: Family, *, record: int | None = None) -> None:
        self.family = family
        self.record = record
        self._values: dict[str, Located] = {}

    @property
    def values(self) -> dict[str, Located]:
        """Each keyword's value, comments left out, in the order found."""
        return dict(self._values)

    def add(self, keyword: str, value: Located) -> None:
        """Record ``value`` under ``keyword``; comments are not recorded.

        Raises
        ------
        CcsdsParseError
            The family's ``duplicate keyword`` error, at ``value``, if
            ``keyword`` already has a value.
        """
        if keyword == COMMENT:
            return
        if keyword in self._values:
            raise self.family.parse_error(
                "duplicate keyword",
                record=self.record,
                line=value.line,
                path=value.path,
                keyword=keyword,
            )
        self._values[keyword] = value
