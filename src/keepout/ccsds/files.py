"""Message files: UTF-8 text, gzip-compressed or not."""

from __future__ import annotations

import gzip
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Final, TypeVar

from keepout.ccsds.message import Family

__all__ = ["load_file", "read_file"]

_GZIP_MAGIC: Final = b"\x1f\x8b"

_T = TypeVar("_T")


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

    Raises
    ------
    CcsdsParseError
        The family's, if the file is broken gzip data or not UTF-8 text.
    """
    data = Path(path).read_bytes()
    if data.startswith(_GZIP_MAGIC):
        try:
            data = gzip.decompress(data)
        except (OSError, EOFError, zlib.error) as exc:
            raise family.parse_error(f"{path}: invalid gzip data: {exc}") from exc
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise family.parse_error(f"{path}: not UTF-8 text: {exc}") from exc


def load_file(path: str | Path, loads: Callable[[str], _T], family: Family) -> _T:
    """A file parsed by ``loads``; see :func:`read_file`."""
    return loads(read_file(path, family))
