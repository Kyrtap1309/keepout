"""Downloaded reference data: cache location, verified atomic writes, HTTP.

Shared by :mod:`keepout.sources.cara` and :mod:`keepout.sources.kelvins`.
Every cached file has a pinned SHA-256 digest; a file is written under a
unique temporary name and renamed into place only once its digest matches,
so the cache never uses a partial or unverified file. A hard kill (SIGKILL,
power loss) mid-write can leave the temporary ``.part`` file behind; it is
never read.
"""

from __future__ import annotations

import hashlib
import http.client
import io
import os
import stat
import tempfile
import threading
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Final, Protocol, TypeAlias

import platformdirs

from keepout.errors import ChecksumMismatch, SourceUnavailable

__all__ = [
    "CHUNK_SIZE",
    "DOWNLOAD_TIMEOUT",
    "Sink",
    "cache_dir",
    "has_digest",
    "http_download",
    "http_stream",
    "store",
]

DOWNLOAD_TIMEOUT: Final = 60.0
"""Seconds :func:`http_download` and :func:`http_stream` wait for the server."""

CHUNK_SIZE: Final = 1 << 20
"""Bytes :func:`http_stream` reads at a time."""


class Sink(Protocol):
    """Where streamed bytes go: anything with a binary ``write``."""

    def write(self, data: bytes, /) -> object:
        """Take the next chunk."""


def cache_dir() -> Path:
    """Root cache directory for downloaded reference data.

    ``$KEEPOUT_CACHE_DIR`` if set, else the platform's user cache directory
    for keepout (on Linux ``$XDG_CACHE_HOME/keepout`` or
    ``~/.cache/keepout``; on macOS ``~/Library/Caches/keepout``). The
    directory is not created here.
    """
    override = os.environ.get("KEEPOUT_CACHE_DIR")
    if override:
        return Path(override)
    return platformdirs.user_cache_path("keepout")


def http_download(url: str) -> bytes:
    """Download ``url`` and return the body.

    Raises
    ------
    SourceUnavailable
        On any network or HTTP error.
    """
    buffer = io.BytesIO()
    http_stream(url, buffer)
    return buffer.getvalue()


def http_stream(url: str, sink: Sink) -> None:
    """Download ``url`` into ``sink`` in chunks of :data:`CHUNK_SIZE` bytes.

    For files too large to hold in memory at once.

    Raises
    ------
    SourceUnavailable
        On any network or HTTP error, also one in the middle of the body.
    """
    try:
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as response:
            while chunk := response.read(CHUNK_SIZE):
                sink.write(chunk)
    except (OSError, http.client.HTTPException) as exc:
        raise SourceUnavailable(f"cannot download {url}: {exc}") from exc


_Signature: TypeAlias = tuple[int, int, int]
"""Inode, size and modification time [ns] of a cached file."""

_verified: Final[dict[Path, tuple[str, _Signature]]] = {}
"""Cached files whose digest this process has checked (or written), with the
signature they had then; see :func:`has_digest`."""
_verified_lock = threading.Lock()


def _signature(path: Path) -> _Signature | None:
    """Signature of the regular file at ``path``; ``None`` if there is none."""
    try:
        status = path.stat()
    except OSError:
        return None
    if not stat.S_ISREG(status.st_mode):
        return None
    return status.st_ino, status.st_size, status.st_mtime_ns


def _remember(path: Path, sha256: str, signature: _Signature | None) -> None:
    """Record that ``path``, with ``signature``, has digest ``sha256``."""
    if signature is not None:
        with _verified_lock:
            _verified[path] = (sha256, signature)


def has_digest(path: Path, sha256: str) -> bool:
    """Whether the file at ``path`` exists and has SHA-256 digest ``sha256``.

    A file already checked by this process (or written by :func:`store`) is
    not hashed again while its inode, size and modification time are
    unchanged.
    """
    signature = _signature(path)
    if signature is None:
        return False
    with _verified_lock:
        if _verified.get(path) == (sha256, signature):
            return True
    with path.open("rb") as cached:
        if hashlib.file_digest(cached, "sha256").hexdigest() != sha256:
            return False
    _remember(path, sha256, signature)
    return True


class _Hashing:
    """A sink that hashes what it writes to ``stream``."""

    def __init__(self, stream: Sink) -> None:
        self.stream = stream
        self.digest = hashlib.sha256()

    def write(self, data: bytes, /) -> object:
        self.digest.update(data)
        return self.stream.write(data)


def store(
    target: Path, write: Callable[[Sink], object], sha256: str, source: str
) -> None:
    """Write ``target`` atomically, keeping it only if its digest is ``sha256``.

    Parameters
    ----------
    target
        The file to create or replace; parent directories are created.
    write
        Writes the contents to the sink it is given.
    sha256
        Expected SHA-256 hex digest of the contents.
    source
        Where the contents come from, for the error message.

    Raises
    ------
    ChecksumMismatch
        If the contents do not have digest ``sha256``; ``target`` is then
        left as it was.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    # A unique temporary name, so concurrent writers never share a partial file.
    descriptor, name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".part"
    )
    partial = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            hashing = _Hashing(stream)
            write(hashing)
        actual = hashing.digest.hexdigest()
        if actual != sha256:
            raise ChecksumMismatch(f"{source}: expected sha256 {sha256}, got {actual}")
        partial.replace(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    _remember(target, sha256, _signature(target))
