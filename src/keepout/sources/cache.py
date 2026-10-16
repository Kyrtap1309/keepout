"""Downloaded reference data: cache location, verified atomic writes, HTTP.

Shared by :mod:`keepout.sources.cara`, :mod:`keepout.sources.kelvins`,
:mod:`keepout.sources.celestrak` and :mod:`keepout.sources.spacetrack`.
Every cached file has a pinned SHA-256 digest; a file is written under a
unique temporary name and renamed into place only once its digest matches,
so the cache never uses a partial or unverified file (:func:`store`). Bytes
already in memory are written the same way, without a digest to match
(:func:`store_bytes`). A hard kill (SIGKILL,
power loss) mid-write can leave the temporary ``.part`` file behind; it is
never read.

A download kept for reuse (CelesTrak, Space-Track) has a ``.meta.json``
file next to it recording its URL, digest and times (:class:`CopyMeta`);
:func:`read_copy` gives the copy only if both files are intact.
"""

from __future__ import annotations

import dataclasses
import hashlib
import http.client
import io
import json
import math
import os
import stat
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus
from importlib.metadata import version
from pathlib import Path
from typing import Final, NamedTuple, Protocol, TypeAlias, TypeGuard

import platformdirs

from keepout.errors import ChecksumMismatch, HttpStatusError, SourceUnavailable

__all__ = [
    "CHUNK_SIZE",
    "DOWNLOAD_TIMEOUT",
    "HOME_PAGE",
    "USER_AGENT",
    "CachedCopy",
    "CopyMeta",
    "Sink",
    "cache_dir",
    "has_digest",
    "http_download",
    "http_stream",
    "meta_path",
    "read_copy",
    "store",
    "store_bytes",
    "write_meta",
]

DOWNLOAD_TIMEOUT: Final = 60.0
"""Seconds :func:`http_download` and :func:`http_stream` wait for the server."""

CHUNK_SIZE: Final = 1 << 20
"""Bytes :func:`http_stream` reads at a time."""

HOME_PAGE: Final = "https://github.com/Kyrtap1309/keepout"
"""keepout's home page, named in :data:`USER_AGENT`."""

USER_AGENT: Final = f"keepout/{version('keepout')} (+{HOME_PAGE})"
"""The ``User-Agent`` of every request: keepout, its version and where to
find it, for servers that ask clients to identify themselves (CelesTrak)."""


class Sink(Protocol):
    """Where streamed bytes go: anything with a binary ``write``."""

    def write(self, data: bytes, /) -> object:
        """Take the next chunk."""


def cache_dir(given: Path | None = None) -> Path:
    """Root cache directory for downloaded reference data.

    ``given`` if not ``None``, else ``$KEEPOUT_CACHE_DIR`` if set, else the
    platform's user cache directory for keepout (on Linux
    ``$XDG_CACHE_HOME/keepout`` or ``~/.cache/keepout``; on macOS
    ``~/Library/Caches/keepout``). The directory is not created here.
    """
    if given is not None:
        return given
    override = os.environ.get("KEEPOUT_CACHE_DIR")
    if override:
        return Path(override)
    return platformdirs.user_cache_path("keepout")


def http_download(url: str) -> bytes:
    """Download ``url`` and return the body, as :func:`http_stream` does.

    Parameters
    ----------
    url
        What to download.

    Raises
    ------
    HttpStatusError
        If the server answers with a status other than 200 OK.
    SourceUnavailable
        On any other network or HTTP error.
    """
    buffer = io.BytesIO()
    http_stream(url, buffer)
    return buffer.getvalue()


def http_stream(url: str, sink: Sink) -> None:
    """Download ``url`` into ``sink`` in chunks of :data:`CHUNK_SIZE` bytes.

    For files too large to hold in memory at once. The request names keepout
    in the ``User-Agent`` (:data:`USER_AGENT`). Redirects are followed; any
    final status but 200 OK is an error, the success codes included, and
    nothing is retried.

    Parameters
    ----------
    url
        What to download.
    sink
        Where the body goes.

    Raises
    ------
    HttpStatusError
        If the server answers with a status other than 200 OK.
    SourceUnavailable
        On any other network or HTTP error, also one in the middle of the body.
    ValueError
        If ``url`` is not an ``http`` or ``https`` URL.
    """
    if urllib.parse.urlsplit(url).scheme not in {"http", "https"}:
        raise ValueError(f"not an HTTP URL: {url!r}")
    # The scheme is checked above, so urllib cannot open a local file.
    request = urllib.request.Request(  # noqa: S310
        url, headers={"User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(  # noqa: S310
            request, timeout=DOWNLOAD_TIMEOUT
        ) as response:
            if response.status != HTTPStatus.OK:
                raise HttpStatusError(
                    f"cannot download {url}: HTTP {response.status} {response.reason}",
                    url=url,
                    status=response.status,
                )
            while chunk := response.read(CHUNK_SIZE):
                sink.write(chunk)
    except urllib.error.HTTPError as exc:
        raise HttpStatusError(
            f"cannot download {url}: {exc}", url=url, status=exc.code
        ) from exc
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

    A file already checked by this process (or written by :func:`store` or
    :func:`store_bytes`) is not hashed again while its inode, size and
    modification time are unchanged.
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


def _write(
    target: Path,
    write: Callable[[Sink], object],
    check: Callable[[str], None] | None = None,
) -> str:
    """Write ``target`` atomically; ``check`` may refuse the contents' digest.

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


_META_FORMAT: Final = 1
"""Version of the ``.meta.json`` layout; another version is not read."""


@dataclass(frozen=True, slots=True)
class CopyMeta:
    """What the ``.meta.json`` file of a cached download records.

    Parameters
    ----------
    url
        Where the copy was downloaded from.
    sha256
        SHA-256 hex digest of the copy.
    retrieved
        Unix time of the download [s].
    checked
        Unix time the source last confirmed the copy current [s]: the
        download, or later (CelesTrak's HTTP 403 for unchanged data).
    """

    url: str
    sha256: str
    retrieved: float
    checked: float


class CachedCopy(NamedTuple):
    """A cached download and its metadata, as :func:`read_copy` finds them."""

    data: bytes
    meta: CopyMeta


def meta_path(target: Path) -> Path:
    """The ``.meta.json`` file of the cached download at ``target``."""
    return target.with_name(f"{target.name}.meta.json")


def read_copy(target: Path, url: str) -> CachedCopy | None:
    """The cached download of ``url`` at ``target``, if it is complete and intact.

    ``None`` if either file is missing or unreadable, the metadata is not
    of this layout or names another URL, or the digest does not match.
    """
    try:
        record = json.loads(meta_path(target).read_bytes())
        data = target.read_bytes()
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or record.get("format") != _META_FORMAT:
        return None
    url_, digest = record.get("url"), record.get("sha256")
    retrieved, checked = record.get("retrieved"), record.get("checked")
    if (
        url_ != url
        or not isinstance(digest, str)
        or not _is_time(retrieved)
        or not _is_time(checked)
        or hashlib.sha256(data).hexdigest() != digest
    ):
        return None
    meta = CopyMeta(url=url, sha256=digest, retrieved=retrieved, checked=checked)
    return CachedCopy(data, meta)


def _is_time(value: object) -> TypeGuard[float]:
    """Whether ``value`` is a Unix time as :func:`write_meta` writes one."""
    return isinstance(value, float) and math.isfinite(value)


def write_meta(target: Path, meta: CopyMeta) -> None:
    """Write the ``.meta.json`` file of the download at ``target`` atomically.

    Times must be floats (``time.time()``); :func:`read_copy` refuses others.
    """
    record = {"format": _META_FORMAT, **dataclasses.asdict(meta)}
    store_bytes(meta_path(target), (json.dumps(record, indent=2) + "\n").encode())
