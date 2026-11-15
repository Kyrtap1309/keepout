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

_GZIP_WBITS: Final = 16 + zlib.MAX_WBITS
"""zlib's window bits for gzip data (header and trailer)."""

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


def _gunzip(data: bytes, path: str | Path, family: Family, max_bytes: int) -> bytes:
    """Gzip ``data`` decompressed, refused once it grows past ``max_bytes``
    (nothing more is inflated).

    Raises
    ------
    CcsdsParseError
        The family's, if the data is broken gzip data or decompresses to more
        than ``max_bytes``.
    """
    out = bytearray()
    rest = data
    while rest:
        if not rest.startswith(_GZIP_MAGIC):
            raise family.parse_error(f"{path}: invalid gzip data: trailing bytes")
        unpacker = zlib.decompressobj(_GZIP_WBITS)
        try:
            out += unpacker.decompress(rest, max_bytes + 1 - len(out))
        except zlib.error as exc:
            raise family.parse_error(f"{path}: invalid gzip data: {exc}") from exc
        if len(out) > max_bytes:
            raise family.parse_error(
                f"{path}: larger than {max_bytes} bytes once decompressed"
            )
        if not unpacker.eof:
            raise family.parse_error(f"{path}: invalid gzip data: truncated")
        rest = unpacker.unused_data
    return bytes(out)


def decode_file(
    data: bytes, path: str | Path, family: Family, *, max_bytes: int | None = None
) -> str:
    """The text of a file's bytes, as :func:`read_file` reads them.

    Parameters
    ----------
    data
        The file's bytes.
    path
        The file, for the error message.
    family
        The message family, whose parse error broken data raises.
    max_bytes
        Most bytes the file, and its data once decompressed if it is gzip
        data, may hold; ``None`` for no limit. With a limit, gzip data is
        inflated no further than it.

    Returns
    -------
    str
        The decoded text.

    Raises
    ------
    CcsdsParseError
        The family's, if the data is broken gzip data, not UTF-8 text, or
        larger than ``max_bytes`` (before or after decompression).
    """
    if max_bytes is not None and len(data) > max_bytes:
        raise family.parse_error(f"{path}: larger than {max_bytes} bytes")
    if data.startswith(_GZIP_MAGIC):
        if max_bytes is not None:
            data = _gunzip(data, path, family, max_bytes)
        else:
            try:
                data = gzip.decompress(data)
            except (OSError, EOFError, zlib.error) as exc:
                raise family.parse_error(f"{path}: invalid gzip data: {exc}") from exc
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise family.parse_error(f"{path}: not UTF-8 text: {exc}") from exc
