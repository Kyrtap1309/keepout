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
:func:`read_copy` gives the copy only if both files are intact, and
:func:`cached_fetch` holds the policy around it: use a fresh copy that can
be read, else download, read and save (gzipped if asked).
"""

from __future__ import annotations

import contextlib
import dataclasses
import gzip
import hashlib
import http.client
import io
import json
import math
import os
import stat
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections.abc import Callable, Collection, Generator, Mapping, Sequence
from dataclasses import dataclass
from http import HTTPStatus
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple, Protocol, TypeAlias, TypeVar

import platformdirs

from keepout._types import JsonValue
from keepout.errors import (
    ChecksumMismatch,
    HttpStatusError,
    SourceFormatError,
    SourceUnavailable,
)

if TYPE_CHECKING:
    from typing_extensions import TypeIs

__all__ = [
    "CHUNK_SIZE",
    "DOWNLOAD_TIMEOUT",
    "HOME_PAGE",
    "RETRY_DELAYS",
    "USER_AGENT",
    "CachedCopy",
    "CopyMeta",
    "Sink",
    "Unchanged",
    "cache_dir",
    "cached_fetch",
    "forget_verified",
    "gunzip",
    "has_digest",
    "http_download",
    "http_request",
    "http_stream",
    "meta_path",
    "network_errors",
    "read_copy",
    "require_max_age",
    "retrying",
    "save_copy",
    "store",
    "store_bytes",
    "store_json",
    "write_meta",
]

DOWNLOAD_TIMEOUT: Final = 60.0
"""Seconds :func:`http_download` and :func:`http_stream` wait for the server."""

_T = TypeVar("_T")

RETRY_DELAYS: Final = (5.0, 30.0)
"""Seconds waited before each new attempt of a download that failed
transiently (:func:`retrying`): two more attempts, over about half a minute."""

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


def http_download(url: str, *, retry_delays: Sequence[float] = RETRY_DELAYS) -> bytes:
    """Download ``url`` and return the body, as :func:`http_stream` does.

    A download that fails on the network or with a server error is tried
    again (:func:`retrying`).

    Parameters
    ----------
    url
        What to download.
    retry_delays
        Seconds waited before each new attempt.

    Raises
    ------
    HttpStatusError
        If the server answers with a status other than 200 OK.
    SourceUnavailable
        On any other network or HTTP error.
    """

    def attempt() -> bytes:
        buffer = io.BytesIO()
        http_stream(url, buffer)
        return buffer.getvalue()

    return retrying(attempt, delays=retry_delays)


def retrying(
    attempt: Callable[[], _T],
    *,
    delays: Sequence[float] = RETRY_DELAYS,
    sleep: Callable[[float], object] | None = None,
) -> _T:
    """``attempt()``, tried again after each of ``delays`` while it fails
    transiently.

    A failure is transient if it is a
    :class:`~keepout.errors.SourceUnavailable` other than an HTTP status (a
    network error), or an HTTP 429 or 5xx status. ``attempt`` must start
    afresh each time: a stream already written to cannot be retried. The
    waits are ``sleep``'s, :func:`time.sleep` if ``None``.

    Raises
    ------
    SourceUnavailable
        The first failure that is not transient, or the last one.
    """
    for delay in delays:
        try:
            return attempt()
        except SourceUnavailable as exc:
            if not _transient(exc):
                raise
        (time.sleep if sleep is None else sleep)(delay)
    return attempt()


def _transient(error: SourceUnavailable) -> bool:
    """Whether a download that failed with ``error`` may succeed if tried again."""
    if not isinstance(error, HttpStatusError):
        return type(error) is SourceUnavailable
    return error.status == HTTPStatus.TOO_MANY_REQUESTS or error.status >= 500


def http_request(
    url: str,
    *,
    schemes: Collection[str] = ("http", "https"),
    data: bytes | None = None,
    method: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> urllib.request.Request:
    """A request for ``url`` that names keepout in its ``User-Agent``
    (:data:`USER_AGENT`), with ``headers`` added.

    Raises
    ------
    ValueError
        If ``url``'s scheme is not one of ``schemes``: urllib must never
        open a local file.
    """
    if urllib.parse.urlsplit(url).scheme not in schemes:
        named = " or ".join(scheme.upper() for scheme in schemes)
        raise ValueError(f"not an {named} URL: {url!r}")
    # The scheme is checked above, so urllib cannot open a local file.
    return urllib.request.Request(  # noqa: S310
        url,
        data=data,
        headers={"User-Agent": USER_AGENT, **(headers or {})},
        method=method,
    )


@contextlib.contextmanager
def network_errors(message: str) -> Generator[None]:
    """Turn a network or protocol error inside into
    :class:`~keepout.errors.SourceUnavailable`, ``<message>: <error>``.

    Raises
    ------
    SourceUnavailable
        On an :class:`OSError` or :class:`http.client.HTTPException`.
    """
    try:
        yield
    except (OSError, http.client.HTTPException) as exc:
        raise SourceUnavailable(f"{message}: {exc}") from exc


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
    request = http_request(url)
    with network_errors(f"cannot download {url}"):
        try:
            with urllib.request.urlopen(  # noqa: S310
                request, timeout=DOWNLOAD_TIMEOUT
            ) as response:
                if response.status != HTTPStatus.OK:
                    raise HttpStatusError(
                        f"cannot download {url}: "
                        f"HTTP {response.status} {response.reason}",
                        url=url,
                        status=response.status,
                    )
                while chunk := response.read(CHUNK_SIZE):
                    sink.write(chunk)
        except urllib.error.HTTPError as exc:
            raise HttpStatusError(
                f"cannot download {url}: {exc}", url=url, status=exc.code
            ) from exc


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
    """Forget which cached files this process has verified (or written).

    The next :func:`has_digest` of each file hashes it again, even if its
    inode, size and modification time are unchanged: for a file changed
    in a way its signature does not show.
    """
    with _verified_lock:
        _verified.clear()


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


def store_json(target: Path, record: Mapping[str, object]) -> str:
    """Write ``record`` to ``target`` as indented JSON text, atomically.

    Returns
    -------
    str
        The SHA-256 hex digest of the text written.
    """
    return store_bytes(target, (json.dumps(record, indent=2) + "\n").encode())


def gunzip(data: bytes) -> bytes | None:
    """``data`` decompressed, or ``None`` if it is not intact gzip data."""
    try:
        return gzip.decompress(data)
    except (OSError, EOFError, zlib.error):
        return None


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
        record: JsonValue = json.loads(meta_path(target).read_bytes())
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


def _is_time(value: object) -> TypeIs[float]:
    """Whether ``value`` is a Unix time as :func:`write_meta` writes one."""
    return isinstance(value, float) and math.isfinite(value)


def save_copy(target: Path, url: str, data: bytes, now: float) -> CopyMeta:
    """Store a download at ``target`` with its metadata, both atomically.

    Parameters
    ----------
    target
        The cached copy.
    url
        Where ``data`` was downloaded from.
    data
        The body.
    now
        When it was received, Unix time [s]: its retrieval and check time.

    Returns
    -------
    CopyMeta
        The metadata written beside it.
    """
    digest = store_bytes(target, data)
    meta = CopyMeta(url=url, sha256=digest, retrieved=now, checked=now)
    write_meta(target, meta)
    return meta


def write_meta(target: Path, meta: CopyMeta) -> None:
    """Write the ``.meta.json`` file of the download at ``target`` atomically.

    Times must be floats (``time.time()``); :func:`read_copy` refuses others.
    """
    record = {"format": _META_FORMAT, **dataclasses.asdict(meta)}
    store_json(meta_path(target), record)


class Unchanged(SourceUnavailable):
    """The source answered that the copy cached is current (CelesTrak's HTTP
    403 for unchanged data); raised by the ``download`` of
    :func:`cached_fetch`.

    :func:`cached_fetch` then uses the cached copy, however old, and records
    the check; it raises this error itself only if there is no readable copy,
    so the message says what the answer means without one.
    """


def require_max_age(max_age: float, interval: float, reason: str) -> None:
    """Refuse a ``max_age`` below a source's least ``interval`` between two
    downloads.

    Raises
    ------
    ValueError
        ``max_age must be at least <interval> s (<reason>), got <max_age>``,
        if ``max_age`` is below ``interval`` (or NaN).
    """
    if not max_age >= interval:
        raise ValueError(
            f"max_age must be at least {interval:g} s ({reason}), got {max_age}"
        )


def cached_fetch(
    target: Path,
    url: str,
    read: Callable[[bytes], _T],
    download: Callable[[], bytes],
    *,
    max_age: float,
    interval: float,
    reason: str,
    now: Callable[[], float],
    compressed: bool = False,
) -> tuple[_T, CopyMeta]:
    """``url`` read from the copy cached at ``target``, else downloaded,
    read and cached.

    A copy last checked current less than ``max_age`` ago (by
    :attr:`CopyMeta.checked`, which is its download time unless the source
    confirmed it later) is used if ``read`` can read it. Otherwise
    ``download`` is called once; if it raises :class:`Unchanged`, the
    cached copy is used however old and its check time is renewed. A
    download is read before it is cached, so one ``read`` refuses is not
    kept.

    Parameters
    ----------
    target
        The cached copy, with its ``.meta.json`` file beside it.
    url
        Where the copy comes from, recorded in its metadata.
    read
        Reads a body; raises :class:`~keepout.errors.SourceFormatError` if
        it cannot.
    download
        Downloads ``url`` and returns the body; called only if no fresh
        readable copy is cached.
    max_age
        Seconds a copy is used without downloading again.
    interval
        The least ``max_age`` allowed: the source's update interval or
        rate limit.
    reason
        Why ``interval`` is the least, for the error message.
    now
        Gives the current Unix time [s]: called before the cache is read
        (the age of the copy, the time of a check) and again once a
        download has arrived (its retrieval time).
    compressed
        Whether the copy is kept gzipped (the body is compressed before it
        is stored and decompressed before it is read).

    Returns
    -------
    tuple[T, CopyMeta]
        What ``read`` made of the body, and the metadata of its copy.

    Raises
    ------
    ValueError
        If ``max_age`` is below ``interval``.
    Unchanged
        If ``download`` raised it and there is no readable cached copy.
    SourceUnavailable, SourceFormatError
        What ``download`` or ``read`` raised for a download.
    """
    require_max_age(max_age, interval, reason)
    start = now()
    cached = read_copy(target, url)
    if cached is not None and 0.0 <= start - cached.meta.checked < max_age:
        value = _read_cached(read, cached, compressed=compressed)
        if value is not None:
            return value[0], cached.meta
    try:
        data = download()
    except Unchanged:
        if cached is None:
            raise
        value = _read_cached(read, cached, compressed=compressed)
        if value is None:
            raise
        meta = dataclasses.replace(cached.meta, checked=start)
        write_meta(target, meta)
        return value[0], meta
    result = read(data)
    # The full Space-Track catalog is 36 MB of JSON, 5 MB gzipped.
    stored = gzip.compress(data, compresslevel=6, mtime=0) if compressed else data
    return result, save_copy(target, url, stored, now())


def _read_cached(
    read: Callable[[bytes], _T], cached: CachedCopy, *, compressed: bool
) -> tuple[_T] | None:
    """``(read(body),)`` of a cached copy; ``None`` if it cannot be read."""
    data = gunzip(cached.data) if compressed else cached.data
    if data is None:
        return None
    try:
        return (read(data),)
    except SourceFormatError:
        return None
