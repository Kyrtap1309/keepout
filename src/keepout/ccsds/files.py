"""Message files: UTF-8 text, gzip-compressed or not."""

from __future__ import annotations

import gzip
import zlib
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from keepout.ccsds.message import Family

__all__ = ["BOM", "decode_file", "read_file", "strip_bom"]

_GZIP_MAGIC: Final = b"\x1f\x8b"

BOM: Final = "\ufeff"
"""The Unicode byte-order mark, which some files start with."""


def strip_bom(text: str) -> str:
    """``text`` without its leading byte-order mark, if it has one.

    Parameters
    ----------
    text
        Decoded file text.

    Returns
    -------
    str
        ``text`` without a leading byte-order mark.
    """
    return text.removeprefix(BOM)


def read_file(path: str | Path, family: Family) -> str:
    """The text of a UTF-8 file, decompressed first if it is gzip data.

    gzip data is recognised by its magic bytes, whatever the file name. A
    byte-order mark is ignored.

    Parameters
    ----------
    path
        File to read.
    family
        The message family, whose parse error a broken file raises.

    Returns
    -------
    str
        The file's text, without a byte-order mark.

    Raises
    ------
    CcsdsParseError
        The family's, if the file is broken gzip data or not UTF-8 text.
    """
    return decode_file(Path(path).read_bytes(), path, family)


def decode_file(data: bytes, path: str | Path, family: Family) -> str:
    """The text of a file's bytes, as :func:`read_file` reads them.

    Parameters
    ----------
    data
        The file's bytes.
    path
        The file, for the error message.
    family
        The message family, whose parse error broken data raises.

    Returns
    -------
    str
        The decoded text.

    Raises
    ------
    CcsdsParseError
        The family's, if the data is broken gzip data or not UTF-8 text.
    """
    if data.startswith(_GZIP_MAGIC):
        try:
            data = gzip.decompress(data)
        except (OSError, EOFError, zlib.error) as exc:
            raise family.parse_error(f"{path}: invalid gzip data: {exc}") from exc
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise family.parse_error(f"{path}: not UTF-8 text: {exc}") from exc
