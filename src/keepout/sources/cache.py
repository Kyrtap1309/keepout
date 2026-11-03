"""Downloads kept for reuse: a copy with its metadata, and the policy of
when to download again.

A download kept for reuse (CelesTrak, Space-Track) has a ``.meta.json``
file next to it recording its URL, digest and times (:class:`CopyMeta`).
:func:`read_copy` gives the copy only if both files are intact.
:func:`cached_fetch` holds the policy around it. It uses a fresh copy that
can be read; otherwise it downloads, reads and saves a new one (gzipped if
asked). Both files are written atomically under keepout's cache directory.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import gzip
import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple, TypeVar

from keepout._types import JsonValue
from keepout.errors import SourceFormatError, SourceUnavailable
from keepout.sources._http import gunzip
from keepout.sources._store import store_bytes, store_json

if TYPE_CHECKING:
    from typing_extensions import TypeIs

__all__ = [
    "CachedCopy",
    "CopyMeta",
    "Unchanged",
    "cached_fetch",
    "meta_path",
    "read_copy",
    "require_max_age",
    "save_copy",
    "write_meta",
]

_T = TypeVar("_T")

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
    """Raise :class:`ValueError` if ``max_age`` is below a source's least
    ``interval`` between two downloads.

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


class _Unreadable(enum.Enum):
    """The marker of a cached copy that does not decompress, or that ``read``
    refused (see :func:`cached_fetch`)."""

    UNREADABLE = enum.auto()


_UNREADABLE: Final = _Unreadable.UNREADABLE


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
    if cached is None:
        return _read_and_save(target, url, read, download(), now, compressed=compressed)
    # The cached copy is read at most once: gunzip and parse can take seconds.
    read_cached = functools.cache(
        functools.partial(_read_cached, read, cached, compressed=compressed)
    )
    if 0.0 <= start - cached.meta.checked < max_age:
        value = read_cached()
        if value is not _UNREADABLE:
            return value, cached.meta
    try:
        data = download()
    except Unchanged:
        value = read_cached()
        if value is _UNREADABLE:
            raise
        meta = dataclasses.replace(cached.meta, checked=start)
        write_meta(target, meta)
        return value, meta
    return _read_and_save(target, url, read, data, now, compressed=compressed)


def _read_cached(
    read: Callable[[bytes], _T], cached: CachedCopy, *, compressed: bool
) -> _T | _Unreadable:
    """``read(body)`` of a cached copy; :data:`_UNREADABLE` if it cannot."""
    data = gunzip(cached.data) if compressed else cached.data
    if data is None:
        return _UNREADABLE
    try:
        return read(data)
    except SourceFormatError:
        return _UNREADABLE


def _read_and_save(
    target: Path,
    url: str,
    read: Callable[[bytes], _T],
    data: bytes,
    now: Callable[[], float],
    *,
    compressed: bool,
) -> tuple[_T, CopyMeta]:
    """What ``read`` makes of a download, and the metadata of its copy, saved
    (gzipped if ``compressed``) only once it has been read."""
    result = read(data)
    # The full Space-Track catalog is 36 MB of JSON, 5 MB gzipped.
    stored = gzip.compress(data, compresslevel=6, mtime=0) if compressed else data
    return result, save_copy(target, url, stored, now())
