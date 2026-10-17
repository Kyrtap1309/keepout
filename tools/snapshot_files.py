"""Shared by the snapshot tools: saved answers checked against their digest,
JSON records written atomically, and committed files that stay byte for byte
the same while their data does."""

from __future__ import annotations

import gzip
import hashlib
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Protocol, TypedDict

from keepout.sources.cache import gunzip


class Digested(Protocol):
    """A saved file listed in a raw index with its SHA-256 digest."""

    @property
    def file(self) -> str:
        """Its file name in the raw directory."""
        ...

    @property
    def sha256(self) -> str:
        """SHA-256 hex digest of its bytes."""
        ...


class FileDigest(TypedDict):
    """Length and SHA-256 hex digest of a committed file as stored."""

    bytes: int
    sha256: str


def read_body(raw: Path, saved: Digested, index: str) -> bytes:
    """A saved file of ``raw``, checked against its digest in ``index``.

    Raises
    ------
    ValueError
        If the file's digest is not the one listed.
    """
    data = (raw / saved.file).read_bytes()
    if hashlib.sha256(data).hexdigest() != saved.sha256:
        raise ValueError(f"{raw / saved.file} does not match its digest in {index}")
    return data


def file_digests(files: Mapping[str, bytes]) -> dict[str, FileDigest]:
    """Each file's length and digest, by name, as a manifest lists them."""
    return {
        name: {"bytes": len(stored), "sha256": hashlib.sha256(stored).hexdigest()}
        for name, stored in files.items()
    }


def stored_files(
    data: Path, contents: Mapping[str, bytes], gzipped: Collection[str]
) -> dict[str, bytes]:
    """The bytes each committed file should hold, by name.

    The files named in ``gzipped`` are stored compressed; one already in
    ``data`` with the right contents is kept as it is, so a different zlib
    build does not change a file whose data did not.
    """
    files: dict[str, bytes] = {}
    for name, content in contents.items():
        path = data / name
        if name not in gzipped:
            files[name] = content
        elif path.exists() and gunzip(path.read_bytes()) == content:
            files[name] = path.read_bytes()
        else:
            files[name] = gzip.compress(content, compresslevel=9, mtime=0)
    return files
