"""The cache of downloaded data: its location and verified atomic writes.

Every cached file has a pinned SHA-256 digest; a file is written under a
unique temporary name and renamed into place only once its digest matches,
so the cache never uses a partial or unverified file (:func:`store`). Bytes
already in memory are written the same way, without a digest to match
(:func:`store_bytes`). A hard kill (SIGKILL, power loss) mid-write can leave
the temporary ``.part`` file behind; it is never read. A digest this process
has checked is not computed again while the file is unchanged
(:func:`has_digest`).
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypeAlias

import platformdirs

from keepout._environment import CACHE_DIR_VARIABLE
from keepout.errors import ChecksumMismatch

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from keepout.sources._http import Sink

__all__ = [
    "cache_dir",
    "forget_verified",
    "has_digest",
    "store",
    "store_bytes",
    "store_json",
]


def cache_dir(given: Path | None = None) -> Path:
    """Root cache directory for downloaded reference data.

    ``given`` if not ``None``, else ``$KEEPOUT_CACHE_DIR`` if set, else the
    platform's user cache directory for keepout (on Linux
    ``$XDG_CACHE_HOME/keepout`` or ``~/.cache/keepout``; on macOS
    ``~/Library/Caches/keepout``). The directory is not created here.
    """
    if given is not None:
        return given
    override = os.environ.get(CACHE_DIR_VARIABLE)
    if override:
        return Path(override)
    return platformdirs.user_cache_path("keepout")


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


def forget_verified() -> None:
    """Forget this process's record of the cached files it has verified (or
    written).

    The next :func:`has_digest` of each file hashes it again, even if its
    inode, size and modification time are unchanged: for a file changed
    in a way its signature does not show.
    """
    with _verified_lock:
        _verified.clear()


def sha256_of(path: Path) -> str:
    """The SHA-256 (hex) of the file at ``path``.

    Raises
    ------
    OSError
        If the file cannot be read.
    """
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def has_digest(path: Path, sha256: str) -> bool:
    """Whether the file at ``path`` exists and has SHA-256 digest ``sha256``.

    A file already checked by this process (or written by :func:`store` or
    :func:`store_bytes`) is not hashed again while its inode, size and
    modification time are unchanged, until :func:`forget_verified`.
    """
    signature = _signature(path)
    if signature is None:
        return False
    with _verified_lock:
        if _verified.get(path) == (sha256, signature):
            return True
    if sha256_of(path) != sha256:
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

    def check(actual: str) -> None:
        if actual != sha256:
            raise ChecksumMismatch(f"{source}: expected sha256 {sha256}, got {actual}")

    _write(target, write, check)


def store_bytes(target: Path, data: bytes) -> str:
    """Write ``data`` to ``target`` atomically, as :func:`store` does.

    For contents already in memory, so there is no digest to check them
    against; the one computed while writing is returned.

    Parameters
    ----------
    target
        The file to create or replace; parent directories are created.
    data
        Its contents.

    Returns
    -------
    str
        The SHA-256 hex digest of ``data``.
    """
    return _write(target, lambda sink: sink.write(data))


def store_json(target: Path, record: Mapping[str, object]) -> str:
    """Write ``record`` to ``target`` atomically, as indented JSON text.

    Returns
    -------
    str
        The SHA-256 hex digest of the text written.
    """
    return store_bytes(target, (json.dumps(record, indent=2) + "\n").encode())


def _write(
    target: Path,
    write: Callable[[Sink], object],
    check: Callable[[str], None] | None = None,
) -> str:
    """The atomic write of ``target``; ``check`` may refuse the contents' digest.

    Returns the SHA-256 hex digest of what was written. If ``write`` or
    ``check`` raises, ``target`` is left as it was.
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
        if check is not None:
            check(actual)
        partial.replace(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    _remember(target, actual, _signature(target))
    return actual
