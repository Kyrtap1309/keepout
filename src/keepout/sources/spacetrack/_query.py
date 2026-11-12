"""What keepout asks Space-Track: the URLs, the request classes and their
published limits, and the queries."""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

from keepout._defaults import CDM_PUBLIC_INTERVAL, GP_INTERVAL

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "BASE_URL",
    "CATALOG",
    "CLASS_INTERVALS",
    "LOGIN_URL",
    "LOGOUT_URL",
    "PUBLIC_CDMS",
    "QUERY_URL",
    "Query",
    "RequestClass",
]

BASE_URL: Final = "https://www.space-track.org"
"""Space-Track.org."""

LOGIN_URL: Final = f"{BASE_URL}/ajaxauth/login"
"""Where the credentials are posted; the answer sets the session cookie."""

LOGOUT_URL: Final = f"{BASE_URL}/ajaxauth/logout"
"""Where the session is ended."""

QUERY_URL: Final = f"{BASE_URL}/basicspacedata/query/class"
"""Queries of the ``basicspacedata`` controller (``gp``, ``cdm_public``)."""

RequestClass: TypeAlias = Literal["gp", "cdm_public"]
"""A Space-Track request class keepout queries (:data:`CLASS_INTERVALS`)."""

CLASS_INTERVALS: Final[Mapping[RequestClass, float]] = MappingProxyType(
    {"gp": GP_INTERVAL, "cdm_public": CDM_PUBLIC_INTERVAL}
)
"""Least seconds between two queries of each request class, as Space-Track
publishes them: ``gp`` once an hour, ``cdm_public`` (every public CDM) once
every 8 hours. Only these classes can be queried."""

_NAME: Final = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")
"""A request class or cache file name: ``gp``, ``cdm_public``, ``catalog``."""


@dataclass(frozen=True, slots=True)
class Query:
    """A Space-Track query whose answer is cached in one file.

    Parameters
    ----------
    request_class
        The request class, a key of :data:`CLASS_INTERVALS` (``gp``).
    path
        The predicates after the class, as Space-Track's documentation
        writes them (``decay_date/null-val/epoch/>now-10/format/json``);
        they are percent-encoded in :attr:`url`.
    name
        The cache file's name, under ``<cache>/spacetrack/<class>/``.

    Raises
    ------
    ValueError
        If the class has no published limit here, the name is not lower-case
        letters and digits joined by single ``-`` or ``_``, or the path is
        empty, starts or ends with ``/``, or holds ``?``, ``#`` or ``..``.
    """

    request_class: RequestClass
    path: str
    name: str

    def __post_init__(self) -> None:
        if self.request_class not in CLASS_INTERVALS:
            raise ValueError(
                f"no Space-Track limit known for class {self.request_class!r}; "
                f"known: {', '.join(CLASS_INTERVALS)}"
            )
        if not _NAME.fullmatch(self.name):
            raise ValueError(f"not a cache file name: {self.name!r}")
        if (
            not self.path
            or self.path.startswith("/")
            or self.path.endswith("/")
            or any(part in self.path for part in ("?", "#", ".."))
        ):
            raise ValueError(f"not a Space-Track query path: {self.path!r}")

    @property
    def interval(self) -> float:
        """Least seconds between two queries of its class."""
        return CLASS_INTERVALS[self.request_class]

    @property
    def url(self) -> str:
        """The query's URL."""
        path = urllib.parse.quote(self.path, safe="/,")
        return f"{QUERY_URL}/{self.request_class}/{path}"


CATALOG: Final = Query(
    "gp",
    "decay_date/null-val/epoch/>now-10/orderby/norad_cat_id/format/json",
    "catalog",
)
"""Space-Track's recommended query for the whole GP catalog: objects not
decayed with an epoch in the last 10 days, by catalog number, as OMM JSON."""

PUBLIC_CDMS: Final = Query(
    "cdm_public", "TCA/>now/orderby/TCA asc/format/json", "upcoming"
)
"""The public CDM summaries of every approach still to come, by TCA, as
JSON. Space-Track allows this query once every 8 hours."""
