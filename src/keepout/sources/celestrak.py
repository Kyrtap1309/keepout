"""CelesTrak: public GP element sets and SOCRATES close-approach reports.

CelesTrak (https://celestrak.org) republishes the general perturbations (GP)
element sets of the 18th Space Defense Squadron from Space-Track.org in
groups; there is no single query for every object. The groups of
:data:`PUBLIC_GROUPS` together form the public catalog, about 20 000
objects: not the whole catalog of about 32 000, as most rocket bodies and
debris are in no group (the full catalog is Space-Track's ``gp`` query,
:func:`keepout.sources.spacetrack.fetch_catalog`, which needs an account).
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
  ``User-Agent``.

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

import re
import time
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Final, TypeVar

from keepout._defaults import CELESTRAK_TTL
from keepout.ccsds.files import strip_bom
from keepout.errors import HttpStatusError, SourceFormatError
from keepout.omm.celestrak import loads_csv
from keepout.sources._format import omm_reader, refuse_format
from keepout.sources._http import Downloader, http_download
from keepout.sources._store import cache_dir
from keepout.sources.cache import Unchanged, cached_fetch
from keepout.sources.catalog import Catalog, CatalogPart, Retrieval
from keepout.sources.socrates import COLUMNS, SocratesApproach, read_socrates

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

__all__ = [
    "GP_URL",
    "PUBLIC_GROUPS",
    "SOCRATES_URL",
    "TTL",
    "SocratesReport",
    "download_once",
    "fetch_catalog",
    "fetch_group",
    "fetch_socrates",
    "group_url",
]

GP_URL: Final = "https://celestrak.org/NORAD/elements/gp.php"
"""CelesTrak's GP query; :func:`group_url` builds a group's URL from it."""

SOCRATES_URL: Final = "https://celestrak.org/socrates/sort-minRange.csv"
"""The latest SOCRATES Plus report, sorted by miss distance (CSV)."""

TTL: Final = CELESTRAK_TTL
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

_TTL_REASON: Final = "CelesTrak updates GP data every two hours"
"""Why ``max_age`` may not be below :data:`TTL`, for the error message."""

_GROUP_NAME: Final = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
"""A CelesTrak group name, lower case: ``active``, ``last-30-days``."""

_SOCRATES_COLUMN: Final = COLUMNS[0]
"""A column of the SOCRATES CSV header, to recognise the report."""

_T = TypeVar("_T")


def download_once(url: str) -> bytes:
    """The body at ``url``, downloaded once: CelesTrak blocks clients that
    retry after an error, so nothing is retried.

    Parameters
    ----------
    url
        A CelesTrak URL, e.g. of :func:`group_url`.

    Raises
    ------
    HttpStatusError
        If CelesTrak answers with a status other than 200 OK.
    SourceUnavailable
        On any other network or HTTP error.
    """
    return http_download(url, retry_delays=())


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

    Parameters
    ----------
    name
        The name of a CelesTrak GP group, e.g. ``active`` or ``stations``.

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
    download: Downloader = download_once,
    now: Callable[[], float] = time.time,
) -> CatalogPart:
    """The element sets of one GP group, from the cache or CelesTrak.

    A cached copy last checked less than ``max_age`` ago is used as is.
    Otherwise the group is downloaded once; HTTP 403 (no change since the
    last download) returns the cached copy, however old.

    Parameters
    ----------
    name
        The group, e.g. ``active`` (case-insensitive).
    cache
        Cache root; if ``None``, ``$KEEPOUT_CACHE_DIR`` or else the
        platform's user cache directory for keepout.
    max_age
        Seconds a cached copy is used without asking CelesTrak; at least
        :data:`TTL`.
    download
        Downloads a URL; :func:`download_once` (no retries) by default.
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
    omms, retrieval = _fetch(
        url,
        target,
        omm_reader(loads_csv, url, "CelesTrak OMM CSV"),
        max_age,
        download,
        now,
    )
    return CatalogPart(group, omms, retrieval)


def fetch_catalog(
    groups: Iterable[str] = PUBLIC_GROUPS,
    *,
    cache: Path | None = None,
    max_age: float = TTL,
    download: Downloader = download_once,
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
        Cache root; if ``None``, ``$KEEPOUT_CACHE_DIR`` or else the
        platform's user cache directory for keepout.
    max_age
        Seconds a cached copy is used without asking CelesTrak; at least
        :data:`TTL`.
    download
        Downloads a URL; :func:`download_once` (no retries) by default.
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
    fetched = tuple(
        fetch_group(name, cache=cache, max_age=max_age, download=download, now=now)
        for name in names
    )
    return Catalog.of(fetched)


def fetch_socrates(
    *,
    cache: Path | None = None,
    max_age: float = TTL,
    download: Downloader = download_once,
    now: Callable[[], float] = time.time,
) -> SocratesReport:
    """The latest SOCRATES Plus report (:data:`SOCRATES_URL`).

    Cached and fetched like a GP group (see :func:`fetch_group`). The text
    is given as CelesTrak sends it; it is only checked to be UTF-8 CSV with
    the report's ``NORAD_CAT_ID_1`` column.

    Parameters
    ----------
    cache
        Cache root; if ``None``, ``$KEEPOUT_CACHE_DIR`` or else the
        platform's user cache directory for keepout.
    max_age
        Seconds a cached copy is used without asking CelesTrak; at least
        :data:`TTL`.
    download
        Downloads a URL; :func:`download_once` (no retries) by default.
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


def _root(cache: Path | None) -> Path:
    return cache_dir(cache) / "celestrak"


def _read_socrates(data: bytes) -> str:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SourceFormatError(f"{SOCRATES_URL} did not send UTF-8 text") from exc
    header = strip_bom(text).partition("\n")[0]
    if _SOCRATES_COLUMN not in header.split(","):
        refuse_format(
            SOCRATES_URL,
            f"a SOCRATES report (no {_SOCRATES_COLUMN} column)",
            data,
        )
    return text


def _fetch(
    url: str,
    target: Path,
    read: Callable[[bytes], _T],
    max_age: float,
    download: Downloader,
    now: Callable[[], float],
) -> tuple[_T, Retrieval]:
    """``url`` read from the cache at ``target`` or downloaded, by the policy
    (:func:`~keepout.sources.cache.cached_fetch`, HTTP 403 meaning unchanged).

    Raises
    ------
    ValueError
        If ``max_age`` is shorter than CelesTrak's update interval.
    HttpStatusError
        If the download fails with an HTTP status other than 403 (re-raised).
    SourceUnavailable
        If CelesTrak answers HTTP 403 and there is no usable cached copy.
    """

    def get() -> bytes:
        try:
            return download(url)
        except HttpStatusError as exc:
            if exc.status != HTTPStatus.FORBIDDEN:
                raise
            raise Unchanged(
                f"{url}: CelesTrak answered HTTP 403, which means the data has "
                "not changed since this address last downloaded it, but there "
                f"is no usable copy at {target}; it can be downloaded again "
                "after CelesTrak's next update (about every two hours)"
            ) from exc

    value, meta = cached_fetch(
        target,
        url,
        read,
        get,
        max_age=max_age,
        interval=TTL,
        reason=_TTL_REASON,
        now=now,
    )
    return value, Retrieval.of(target, meta)
