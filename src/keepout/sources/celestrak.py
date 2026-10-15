"""CelesTrak: public GP element sets and SOCRATES close-approach reports.

CelesTrak (https://celestrak.org) republishes the general perturbations (GP)
element sets of the 18th Space Defense Squadron from Space-Track.org in
groups; there is no single query for every object. The groups of
:data:`PUBLIC_GROUPS` together form the public catalog, about 20 000
objects: not the whole catalog of about 30 000, as most rocket bodies and
debris are in no group (the full catalog is Space-Track's ``gp`` query).
Element sets are fetched as CSV (one OMM keyword per column), a third of the
size of the JSON form, and read by :func:`keepout.omm.celestrak.loads_csv`.
A SOCRATES report is kept as text and read by
:meth:`SocratesReport.approaches` (see :mod:`keepout.sources.socrates`).

The client follows CelesTrak's usage policy
(https://celestrak.org/usage-policy.php):

- GP data change about every two hours, and each URL should be downloaded
  at most once per change. A download is cached and reused for at least
  :data:`TTL` (2 h) without asking CelesTrak again.
- CelesTrak answers a repeat download of unchanged data with HTTP 403. The
  cached copy is then current: it is returned, and the time of the check is
  recorded, so the next request again waits :data:`TTL`.
- Any other status than 200 OK, and any network error, raises at once;
  nothing is retried (clients that ignore errors are blocked).
- Requests name keepout, its version and its home page in the
  ``User-Agent`` (:data:`~keepout.sources.cache.USER_AGENT`).

Cached copies live under ``<cache>/celestrak/`` (``gp/<group>.csv`` and
``socrates/sort-minRange.csv``), each with a ``.meta.json`` file holding its
URL, SHA-256 digest and the times it was downloaded and last checked. Both
files are written atomically; a copy whose metadata cannot be read, or whose
digest does not match, is not used.

The data is basic space situational awareness data of USSPACECOM (18th
Space Defense Squadron), from Space-Track.org via CelesTrak; cite it as
such when you republish it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Final, TypeAlias, TypeGuard, TypeVar

from astropy.time import Time

from keepout.errors import (
    HttpStatusError,
    OmmParseError,
    SourceFormatError,
    SourceUnavailable,
)
from keepout.omm.celestrak import loads_csv
from keepout.omm.model import Omm, deduplicate
from keepout.sources.cache import cache_dir, http_download, store_bytes
from keepout.sources.socrates import COLUMNS, SocratesApproach, read_socrates

__all__ = [
    "GP_URL",
    "PUBLIC_GROUPS",
    "SOCRATES_URL",
    "TTL",
    "Catalog",
    "Downloader",
    "GpGroup",
    "Retrieval",
    "SocratesReport",
    "fetch_catalog",
    "fetch_group",
    "fetch_socrates",
    "group_url",
]

GP_URL: Final = "https://celestrak.org/NORAD/elements/gp.php"
"""CelesTrak's GP query; :func:`group_url` builds a group's URL from it."""

SOCRATES_URL: Final = "https://celestrak.org/socrates/sort-minRange.csv"
"""The latest SOCRATES Plus report, sorted by miss distance (CSV)."""

TTL: Final = 7200.0
"""Seconds a download is used without asking CelesTrak again: CelesTrak's
update interval of two hours. Also the least ``max_age`` allowed."""

PUBLIC_GROUPS: Final = (
    "active",
    "last-30-days",
    "analyst",
    "fengyun-1c-debris",
    "cosmos-2251-debris",
    "iridium-33-debris",
)
"""The GP groups that together form CelesTrak's public catalog: active
satellites, everything launched in the last 30 days (new rocket bodies and
debris too), analyst objects (tracked, not yet identified) and the debris of
the three largest fragmentations still in orbit. They overlap (a new
satellite is in ``active`` and ``last-30-days``); :func:`fetch_catalog`
keeps one element set per object. Supplemental GP data (operator
ephemerides) is not GP data of the 18th SDS and is left out."""

_GROUP_NAME: Final = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
"""A CelesTrak group name, lower case: ``active``, ``last-30-days``."""

_META_FORMAT: Final = 1
"""Version of the ``.meta.json`` layout; another version is not read."""

_SOCRATES_COLUMN: Final = COLUMNS[0]
"""A column of the SOCRATES CSV header, to recognise the report."""

Downloader: TypeAlias = Callable[[str], bytes]
"""Downloads a URL and returns the body, raising
:class:`~keepout.errors.HttpStatusError` for an HTTP status other than 200
and :class:`~keepout.errors.SourceUnavailable` for other failures, like
:func:`~keepout.sources.cache.http_download`."""

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class Retrieval:
    """Where and when a cached CelesTrak file came from.

    Parameters
    ----------
    url
        The URL it was downloaded from.
    path
        The cached copy.
    retrieved
        When CelesTrak sent this copy (UTC).
    checked
        When CelesTrak last confirmed it current: the download, or a later
        HTTP 403 for unchanged data (UTC).
    """

    url: str
    path: Path
    retrieved: Time
    checked: Time


@dataclass(frozen=True, slots=True)
class GpGroup:
    """The element sets of one CelesTrak GP group.

    Parameters
    ----------
    name
        The group, e.g. ``active``.
    omms
        Its element sets, in CelesTrak's order.
    retrieval
        Where and when they were downloaded.
    """

    name: str
    omms: tuple[Omm, ...]
    retrieval: Retrieval


@dataclass(frozen=True, slots=True)
class Catalog:
    """Element sets of several GP groups, one per object.

    Parameters
    ----------
    omms
        The newest element set of each object, by catalog number.
    duplicates
        Element sets of objects that were in more than one group, left out
        because another set of the object is newer (see
        :func:`keepout.omm.deduplicate`); exact copies are not listed.
    groups
        The groups, in the order they were fetched.
    """

    omms: tuple[Omm, ...]
    duplicates: tuple[Omm, ...]
    groups: tuple[GpGroup, ...]


@dataclass(frozen=True, slots=True)
class SocratesReport:
    """A SOCRATES Plus report as CelesTrak publishes it (CSV text).

    Parameters
    ----------
    text
        The CSV, one row per close approach.
    retrieval
        Where and when it was downloaded.
    """

    text: str
    retrieval: Retrieval

    def approaches(self) -> tuple[SocratesApproach, ...]:
        """The approaches of the report, in its order (read anew each call).

        Raises
        ------
        SourceFormatError
            If the text is not a SOCRATES report keepout can read (see
            :func:`~keepout.sources.socrates.read_socrates`).
        """
        return read_socrates(self.text)


def group_url(name: str) -> str:
    """The CSV URL of the GP group ``name`` (case-insensitive).

    Raises
    ------
    ValueError
        If ``name`` is not a group name: letters, digits and single hyphens.
    """
    return f"{GP_URL}?GROUP={_group_name(name)}&FORMAT=CSV"


def fetch_group(
    name: str,
    *,
    cache: Path | None = None,
    max_age: float = TTL,
    download: Downloader = http_download,
    now: Callable[[], float] = time.time,
) -> GpGroup:
    """The element sets of one GP group, from the cache or CelesTrak.

    A cached copy last checked less than ``max_age`` ago is used as is.
    Otherwise the group is downloaded once; HTTP 403 (no change since the
    last download) returns the cached copy, however old.

    Parameters
    ----------
    name
        The group, e.g. ``active`` (case-insensitive).
    cache
        Cache root; :func:`~keepout.sources.cache.cache_dir` if ``None``.
    max_age
        Seconds a cached copy is used without asking CelesTrak; at least
        :data:`TTL`.
    download
        Downloads a URL.
    now
        Gives the current Unix time [s], by which cached copies age.

    Raises
    ------
    ValueError
        If ``name`` is not a group name or ``max_age`` is below :data:`TTL`.
    SourceUnavailable
        If CelesTrak answers 403 and there is no usable cached copy, or
        anything but 200 or 403 (:class:`~keepout.errors.HttpStatusError`),
        or cannot be reached.
    SourceFormatError
        If the download is not CelesTrak OMM CSV. Nothing is cached then.
    """
    group = _group_name(name)
    url = group_url(group)
    target = _root(cache) / "gp" / f"{group}.csv"
    omms, retrieval = _fetch(url, target, _gp_reader(url), max_age, download, now)
    return GpGroup(group, omms, retrieval)


def fetch_catalog(
    groups: Iterable[str] = PUBLIC_GROUPS,
    *,
    cache: Path | None = None,
    max_age: float = TTL,
    download: Downloader = http_download,
    now: Callable[[], float] = time.time,
) -> Catalog:
    """The element sets of several GP groups, one per object.

    The groups are fetched in turn with :func:`fetch_group`; the first
    failure stops the fetch (groups fetched before it stay cached). An
    object in several groups keeps its newest element set.

    Parameters
    ----------
    groups
        Group names; a name given twice is fetched once.
    cache
        Cache root; :func:`~keepout.sources.cache.cache_dir` if ``None``.
    max_age
        Seconds a cached copy is used without asking CelesTrak; at least
        :data:`TTL`.
    download
        Downloads a URL.
    now
        Gives the current Unix time [s], by which cached copies age.

    Raises
    ------
    TypeError
        If ``groups`` is a single string.
    ValueError
        If there are no groups, a name is not a group name, or ``max_age``
        is below :data:`TTL`.
    SourceUnavailable, SourceFormatError
        As :func:`fetch_group`.
    """
    if isinstance(groups, str):
        raise TypeError("groups must be a collection of group names, not a string")
    names = tuple(dict.fromkeys(_group_name(name) for name in groups))
    if not names:
        raise ValueError("no groups to fetch")
    _check_max_age(max_age)
    fetched = tuple(
        fetch_group(name, cache=cache, max_age=max_age, download=download, now=now)
        for name in names
    )
    kept, dropped = deduplicate(omm for group in fetched for omm in group.omms)
    ordered = tuple(sorted(kept, key=lambda omm: omm.catalog_number))
    return Catalog(ordered, dropped, fetched)


def fetch_socrates(
    *,
    cache: Path | None = None,
    max_age: float = TTL,
    download: Downloader = http_download,
    now: Callable[[], float] = time.time,
) -> SocratesReport:
    """The latest SOCRATES Plus report (:data:`SOCRATES_URL`).

    Cached and fetched like a GP group (see :func:`fetch_group`). The text
    is given as CelesTrak sends it; it is only checked to be UTF-8 CSV with
    the report's ``NORAD_CAT_ID_1`` column.

    Parameters
    ----------
    cache
        Cache root; :func:`~keepout.sources.cache.cache_dir` if ``None``.
    max_age
        Seconds a cached copy is used without asking CelesTrak; at least
        :data:`TTL`.
    download
        Downloads a URL.
    now
        Gives the current Unix time [s], by which cached copies age.

    Raises
    ------
    ValueError
        If ``max_age`` is below :data:`TTL`.
    SourceUnavailable, SourceFormatError
        As :func:`fetch_group`.
    """
    target = _root(cache) / "socrates" / "sort-minRange.csv"
    text, retrieval = _fetch(
        SOCRATES_URL, target, _read_socrates, max_age, download, now
    )
    return SocratesReport(text, retrieval)


def _group_name(name: str) -> str:
    group = name.lower()
    if not _GROUP_NAME.fullmatch(group):
        raise ValueError(f"not a CelesTrak group name: {name!r}")
    return group


def _check_max_age(max_age: float) -> None:
    if not max_age >= TTL:
        raise ValueError(
            f"max_age must be at least {TTL:g} s (CelesTrak updates GP data "
            f"every two hours), got {max_age}"
        )


def _root(cache: Path | None) -> Path:
    return cache_dir(cache) / "celestrak"


def _gp_reader(url: str) -> Callable[[bytes], tuple[Omm, ...]]:
    def read(data: bytes) -> tuple[Omm, ...]:
        try:
            return loads_csv(data.decode("utf-8"))
        except (UnicodeDecodeError, OmmParseError) as exc:
            raise SourceFormatError(
                f"{url} did not send CelesTrak OMM CSV ({exc}); "
                f"it starts with {data[:60]!r}"
            ) from exc

    return read


def _read_socrates(data: bytes) -> str:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SourceFormatError(f"{SOCRATES_URL} did not send UTF-8 text") from exc
    header = text.removeprefix("\ufeff").partition("\n")[0]
    if _SOCRATES_COLUMN not in header.split(","):
        raise SourceFormatError(
            f"{SOCRATES_URL} did not send a SOCRATES report (no "
            f"{_SOCRATES_COLUMN} column); it starts with {data[:60]!r}"
        )
    return text


@dataclass(frozen=True, slots=True)
class _Meta:
    """What the ``.meta.json`` file of a cached copy records."""

    url: str
    sha256: str
    retrieved: float
    """Unix time of the download."""
    checked: float
    """Unix time CelesTrak last confirmed the copy current."""


@dataclass(frozen=True, slots=True)
class _Cached:
    meta: _Meta
    data: bytes


def _meta_path(target: Path) -> Path:
    return target.with_name(f"{target.name}.meta.json")


def _read_cache(target: Path, url: str) -> _Cached | None:
    """The cached copy of ``url`` at ``target``, if it is complete and intact."""
    try:
        record = json.loads(_meta_path(target).read_bytes())
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
    meta = _Meta(url=url, sha256=digest, retrieved=retrieved, checked=checked)
    return _Cached(meta, data)


def _is_time(value: object) -> TypeGuard[float]:
    """Whether ``value`` is a Unix time as :func:`_write_meta` writes one."""
    return isinstance(value, float) and math.isfinite(value)


def _write_meta(target: Path, meta: _Meta) -> None:
    record = {"format": _META_FORMAT, **dataclasses.asdict(meta)}
    store_bytes(_meta_path(target), (json.dumps(record, indent=2) + "\n").encode())


def _retrieval(target: Path, meta: _Meta) -> Retrieval:
    return Retrieval(
        url=meta.url,
        path=target,
        retrieved=Time(meta.retrieved, format="unix", scale="utc"),
        checked=Time(meta.checked, format="unix", scale="utc"),
    )


def _parsed(read: Callable[[bytes], _T], cached: _Cached | None) -> _T | None:
    """The cached copy read, or ``None`` if there is none or it cannot be read."""
    if cached is None:
        return None
    try:
        return read(cached.data)
    except SourceFormatError:
        return None


def _fetch(
    url: str,
    target: Path,
    read: Callable[[bytes], _T],
    max_age: float,
    download: Downloader,
    clock: Callable[[], float],
) -> tuple[_T, Retrieval]:
    """``url`` read from the cache at ``target`` or downloaded, by the policy."""
    _check_max_age(max_age)
    cached = _read_cache(target, url)
    now = clock()
    if cached is not None and 0.0 <= now - cached.meta.checked < max_age:
        value = _parsed(read, cached)
        if value is not None:
            return value, _retrieval(target, cached.meta)
    try:
        data = download(url)
    except HttpStatusError as exc:
        if exc.status != HTTPStatus.FORBIDDEN:
            raise
        value = _parsed(read, cached)
        if cached is None or value is None:
            raise SourceUnavailable(
                f"{url}: CelesTrak answered HTTP 403, which means the data has "
                "not changed since this address last downloaded it, but there "
                f"is no usable copy at {target}; it can be downloaded again "
                "after CelesTrak's next update (about every two hours)"
            ) from exc
        meta = dataclasses.replace(cached.meta, checked=now)
        _write_meta(target, meta)
        return value, _retrieval(target, meta)
    value = read(data)
    digest = store_bytes(target, data)
    meta = _Meta(url=url, sha256=digest, retrieved=now, checked=now)
    _write_meta(target, meta)
    return value, _retrieval(target, meta)
