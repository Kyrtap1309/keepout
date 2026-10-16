"""Space-Track.org: the full GP catalog, within the site's published limits.

Space-Track.org (https://www.space-track.org) is where the 18th Space
Defense Squadron publishes its general perturbations (GP) element sets, for
every tracked object in orbit: about 30 000, against the 20 000 of
CelesTrak's public groups (:mod:`keepout.sources.celestrak`). It needs a
free account. :func:`fetch_catalog` downloads the catalog with the query
Space-Track recommends for it (:data:`CATALOG`: element sets of objects not
decayed, with an epoch in the last 10 days, as OMM JSON) and gives the same
:class:`~keepout.sources.celestrak.Catalog` as CelesTrak's client.
:class:`SpaceTrackClient` is the session underneath, for other queries.

**Credentials** come only from the environment: ``SPACETRACK_IDENTITY``
(the account's user name) and ``SPACETRACK_PASSWORD``
(:meth:`Credentials.from_env`), read only when a request must be sent.
Each user needs an account of their own (Space-Track user agreement).
Neither value appears in a ``repr``, an error message or a cached file:
server text quoted in an error has both replaced by ``<hidden>``.

**Limits.** Space-Track suspends accounts that query too often
(https://www.space-track.org/documentation#/api, "API Use Guidelines"), so
the client enforces the published limits itself:

- Fewer than 30 requests a minute and 300 an hour: :class:`Throttle` waits
  before a request that would make more than :data:`PER_MINUTE` (20) in a
  minute or :data:`PER_HOUR` (200) in an hour, a third below the limits to
  leave room for other use of the same account (the website, a browser).
  Logins and logouts count. The count is the client's own; the class
  limits below keep separate clients and processes far below it.
- Each request class has its own limit (:data:`CLASS_INTERVALS`): ``gp``
  once an hour, ``cdm_public`` (all public CDM summaries) once every 8
  hours. Every answer is cached on disk, and the time of each class's last
  query is recorded there too (``<cache>/spacetrack/requests.json``),
  before the query is sent, so a failed query counts. A copy younger than
  ``max_age`` is returned without contacting Space-Track; a query of a
  class asked for less than its interval ago raises
  :class:`~keepout.errors.RateLimited` without contacting it either. The
  record is per cache directory: processes sharing one share the limit.
  It is read again after a login, so a query another process recorded
  meanwhile is kept and still refuses its class; two processes recording
  at the same instant may both send.
- An answer saying the rate limit was violated (HTTP 500 "violated your
  query rate limit", or 429) raises :class:`~keepout.errors.RateLimited`
  and stops the client: it sends nothing more, not even the logout.
  Nothing is retried, except one fresh login after HTTP 401 (the session,
  two hours, expired) followed by the same query once more.
- The session is closed (``/ajaxauth/logout``) when the client is closed.

Space-Track asks that the hourly ``gp`` query is sent at a random minute,
not at the top or bottom of the hour; a scheduled job should pick one.

Cached copies live under ``<cache>/spacetrack/<class>/`` (the catalog in
``gp/catalog.json``), each with a ``.meta.json`` file as CelesTrak's
(:func:`~keepout.sources.cache.read_copy`).

GP element sets are basic space situational awareness data of USSPACECOM
(18th Space Defense Squadron), which Space-Track.org's user agreement
allows to redistribute with citation; cite them as such. Other classes
(``cdm_public`` among them) may not be redistributed without approval.
"""

from __future__ import annotations

import contextlib
import http.client
import http.cookiejar
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http import HTTPStatus
from pathlib import Path
from types import MappingProxyType, TracebackType
from typing import Final, Literal, Protocol, Self, TypeVar

from astropy.time import Time

from keepout.errors import (
    AuthError,
    HttpStatusError,
    OmmParseError,
    RateLimited,
    SourceFormatError,
    SourceUnavailable,
)
from keepout.omm.celestrak import loads_json
from keepout.omm.model import Omm, deduplicate
from keepout.sources.cache import (
    DOWNLOAD_TIMEOUT,
    USER_AGENT,
    CopyMeta,
    cache_dir,
    read_copy,
    store_bytes,
    write_meta,
)
from keepout.sources.celestrak import Catalog, GpGroup, Retrieval

__all__ = [
    "BASE_URL",
    "CATALOG",
    "CLASS_INTERVALS",
    "GROUP",
    "HIDDEN",
    "IDENTITY_VARIABLE",
    "LOGIN_URL",
    "LOGOUT_URL",
    "PASSWORD_VARIABLE",
    "PER_HOUR",
    "PER_MINUTE",
    "QUERY_URL",
    "Credentials",
    "Query",
    "Request",
    "Response",
    "SpaceTrackClient",
    "Throttle",
    "Transport",
    "UrllibTransport",
    "fetch_catalog",
    "read_catalog",
]

BASE_URL: Final = "https://www.space-track.org"
"""Space-Track.org."""

LOGIN_URL: Final = f"{BASE_URL}/ajaxauth/login"
"""Where the credentials are posted; the answer sets the session cookie."""

LOGOUT_URL: Final = f"{BASE_URL}/ajaxauth/logout"
"""Ends the session."""

QUERY_URL: Final = f"{BASE_URL}/basicspacedata/query/class"
"""Queries of the ``basicspacedata`` controller (``gp``, ``cdm_public``)."""

IDENTITY_VARIABLE: Final = "SPACETRACK_IDENTITY"
"""The environment variable holding the account's user name."""

# The name of the variable, not a password.
PASSWORD_VARIABLE: Final = "SPACETRACK_PASSWORD"  # noqa: S105
"""The environment variable holding the account's password."""

PER_MINUTE: Final = 20
"""Most requests sent in any 60 s; Space-Track allows fewer than 30."""

PER_HOUR: Final = 200
"""Most requests sent in any hour; Space-Track allows fewer than 300."""

CLASS_INTERVALS: Final[Mapping[str, float]] = MappingProxyType(
    {"gp": 3600.0, "cdm_public": 8 * 3600.0}
)
"""Least seconds between two queries of each request class, as Space-Track
publishes them: ``gp`` once an hour, ``cdm_public`` (every public CDM) once
every 8 hours. Only these classes can be queried."""

GROUP: Final = "space-track"
"""The name of the one :class:`~keepout.sources.celestrak.GpGroup` of a
catalog from Space-Track."""

HIDDEN: Final = "<hidden>"
"""What stands for a credential in text shown to the user."""

_RATE_LIMIT_TEXT: Final = b"violated your query rate limit"
"""What Space-Track's answer says when a query broke the per-minute limit."""

_EXCERPT: Final = 200
"""Characters of a server answer quoted in an error message."""

_LEDGER: Final = "requests.json"
"""The file recording when each request class was last queried."""

_LEDGER_FORMAT: Final = 1
"""Version of the ledger layout; another version is an error."""

_NAME: Final = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")
"""A request class or cache file name: ``gp``, ``cdm_public``, ``catalog``."""

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True, repr=False)
class Credentials:
    """A Space-Track account's user name and password.

    Neither is shown by ``repr`` or ``str``.

    Parameters
    ----------
    identity
        The user name (the e-mail address the account was opened with).
    password
        The password.

    Raises
    ------
    ValueError
        If either is empty.
    """

    identity: str
    password: str

    def __post_init__(self) -> None:
        if not self.identity or not self.password:
            raise ValueError("Space-Track identity and password must not be empty")

    def __repr__(self) -> str:
        return f"Credentials({HIDDEN})"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Credentials:
        """The credentials in :data:`IDENTITY_VARIABLE` and
        :data:`PASSWORD_VARIABLE`.

        Parameters
        ----------
        environ
            The environment; :data:`os.environ` if ``None``.

        Raises
        ------
        AuthError
            If a variable is not set or empty (the message names it).
        """
        if environ is None:
            environ = os.environ
        missing = [
            name
            for name in (IDENTITY_VARIABLE, PASSWORD_VARIABLE)
            if not environ.get(name)
        ]
        if missing:
            raise AuthError(
                f"no Space-Track credentials: set {' and '.join(missing)} "
                "(an account is free at https://www.space-track.org)"
            )
        return cls(environ[IDENTITY_VARIABLE], environ[PASSWORD_VARIABLE])

    def redact(self, text: str) -> str:
        """``text`` with the identity and password, as typed or URL-encoded,
        replaced by ``<hidden>``."""
        # Longest first, so a secret inside another is not left half-shown.
        forms = sorted(_secret_forms(self), key=len, reverse=True)
        return re.sub("|".join(map(re.escape, forms)), HIDDEN, text)


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

    request_class: str
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


@dataclass(frozen=True, slots=True)
class Request:
    """An HTTP request a :class:`Transport` sends.

    Parameters
    ----------
    method
        ``GET`` or ``POST``.
    url
        Where to.
    form
        The body of a ``POST``, form-encoded; not shown by ``repr`` (the
        login's holds the password).
    """

    method: Literal["GET", "POST"]
    url: str
    form: bytes | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class Response:
    """A server's answer, whatever its status.

    Parameters
    ----------
    status
        The HTTP status code.
    body
        The body (an error page for an error status).
    """

    status: int
    body: bytes = field(repr=False)


class Transport(Protocol):
    """Sends a request and gives the answer, keeping the session cookie.

    Answers of any HTTP status are returned, not raised; a failure to get
    one (network, protocol) raises :class:`~keepout.errors.SourceUnavailable`.
    """

    def __call__(self, request: Request, /) -> Response:
        """Send ``request`` and return the answer."""


class UrllibTransport:
    """A :class:`Transport` on :mod:`urllib.request`, with a cookie jar.

    Requests name keepout in the ``User-Agent``
    (:data:`~keepout.sources.cache.USER_AGENT`); redirects are followed.

    Parameters
    ----------
    timeout
        Seconds to wait for the server.
    handlers
        More :mod:`urllib.request` handlers, e.g. a proxy handler.
    """

    def __init__(
        self,
        *,
        timeout: float = DOWNLOAD_TIMEOUT,
        handlers: Iterable[urllib.request.BaseHandler] = (),
    ) -> None:
        self.timeout = timeout
        self.cookies = http.cookiejar.CookieJar()
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.cookies), *handlers
        )

    def __call__(self, request: Request, /) -> Response:
        """Send ``request`` and return the answer.

        Raises
        ------
        ValueError
            If the URL is not ``https``.
        SourceUnavailable
            If no answer comes (network or protocol error, timeout).
        """
        if urllib.parse.urlsplit(request.url).scheme != "https":
            raise ValueError(f"not an HTTPS URL: {request.url!r}")
        headers = {"User-Agent": USER_AGENT}
        if request.form is not None:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        # The scheme is checked above, so urllib cannot open a local file.
        prepared = urllib.request.Request(  # noqa: S310
            request.url, data=request.form, headers=headers, method=request.method
        )
        try:
            with self._opener.open(prepared, timeout=self.timeout) as answer:
                return Response(answer.status, answer.read())
        except urllib.error.HTTPError as exc:
            return Response(exc.code, _error_body(exc))
        except (OSError, http.client.HTTPException) as exc:
            raise SourceUnavailable(f"cannot reach {request.url}: {exc}") from exc


class Throttle:
    """Spaces requests: at most ``per_minute`` in any 60 s and ``per_hour``
    in any hour, waiting before a request that would make more.

    Parameters
    ----------
    per_minute, per_hour
        The limits.
    clock
        Gives the time [s] (a monotonic clock).
    sleep
        Waits the given seconds.

    Raises
    ------
    ValueError
        If a limit is below 1.
    """

    def __init__(
        self,
        *,
        per_minute: int = PER_MINUTE,
        per_hour: int = PER_HOUR,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], object] = time.sleep,
    ) -> None:
        if per_minute < 1 or per_hour < 1:
            raise ValueError(
                f"limits must be at least 1, got {per_minute}/min and {per_hour}/h"
            )
        self._limits = ((60.0, per_minute), (3600.0, per_hour))
        self._clock = clock
        self._sleep = sleep
        self._sent: deque[float] = deque(maxlen=max(per_minute, per_hour))

    def wait(self) -> None:
        """Wait until a request may be sent, and count it as sent."""
        now = self._clock()
        due = now
        for window, limit in self._limits:
            if len(self._sent) >= limit:
                due = max(due, self._sent[-limit] + window)
        if due > now:
            self._sleep(due - now)
        self._sent.append(due)


class SpaceTrackClient:
    """A Space-Track session that keeps to the published limits.

    Logs in on the first query that must go to Space-Track (a cached or
    refused one needs no credentials) and out when closed; use it as a
    context manager. See the module notes for the limits it enforces.

    Parameters
    ----------
    credentials
        The account; from the environment (:meth:`Credentials.from_env`)
        at the login if ``None``.
    cache
        Cache root; :func:`~keepout.sources.cache.cache_dir` if ``None``.
    transport
        Sends the requests; a new :class:`UrllibTransport` if ``None``.
    throttle
        Spaces the requests; a new :class:`Throttle` if ``None``.
    now
        Gives the current Unix time [s], by which cached copies and the
        class limits age.
    """

    def __init__(
        self,
        credentials: Credentials | None = None,
        *,
        cache: Path | None = None,
        transport: Transport | None = None,
        throttle: Throttle | None = None,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._credentials = credentials
        self._root = cache_dir(cache) / "spacetrack"
        self._transport = UrllibTransport() if transport is None else transport
        self._throttle = Throttle() if throttle is None else throttle
        self._now = now
        self._logged_in = False
        self._stopped: RateLimited | None = None

    def __repr__(self) -> str:
        if self._stopped is not None:
            state = " (stopped)"
        else:
            state = " (logged in)" if self._logged_in else ""
        return f"<SpaceTrackClient {self._root}{state}>"

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def ledger(self) -> Path:
        """The file recording when each request class was last queried."""
        return self._root / _LEDGER

    def fetch(
        self,
        query: Query,
        read: Callable[[bytes], _T],
        *,
        max_age: float | None = None,
    ) -> tuple[_T, Retrieval]:
        """The answer to ``query``, from the cache or Space-Track, read.

        A cached answer younger than ``max_age`` is read and returned.
        Otherwise, if the query's class was asked for less than its interval
        ago (by any query, in any process using this cache), nothing is sent
        and :class:`~keepout.errors.RateLimited` is raised. Otherwise the
        client logs in if needed, checks the record again (another process
        may have queried meanwhile; the session stays open for
        :meth:`close`), records the query at the time after the login, and
        sends, reads and caches it.

        Parameters
        ----------
        query
            What to ask.
        read
            Reads an answer; raises
            :class:`~keepout.errors.SourceFormatError` if it cannot (a
            cached copy it cannot read is not used; a download it cannot
            read is not cached).
        max_age
            Seconds a cached answer is used; the class's interval if
            ``None``, and not less.

        Returns
        -------
        tuple[T, Retrieval]
            What ``read`` made of the answer, and where and when it came
            from.

        Raises
        ------
        ValueError
            If ``max_age`` is below the class's interval.
        RateLimited
            If the class's interval has not passed, Space-Track answered that
            its rate limit was violated, or it did so earlier to this client.
        AuthError
            If there are no credentials or Space-Track refuses them.
        SourceUnavailable
            If Space-Track answers with another error
            (:class:`~keepout.errors.HttpStatusError`) or cannot be reached.
        SourceFormatError
            If ``read`` cannot read the answer, or the ledger is not one
            keepout wrote.
        """
        interval = query.interval
        if max_age is None:
            max_age = interval
        if not max_age >= interval:
            raise ValueError(
                f"max_age must be at least {interval:g} s (Space-Track allows "
                f"one {query.request_class} query per {interval:g} s), got {max_age}"
            )
        url = query.url
        target = self._root / query.request_class / f"{query.name}.json"
        now = self._now()
        cached = read_copy(target, url)
        if cached is not None and 0.0 <= now - cached.meta.retrieved < max_age:
            try:
                return read(cached.data), _retrieval(target, cached.meta)
            except SourceFormatError:
                pass
        self._check_interval(query, self._read_ledger(), now)
        self._check_running()
        if not self._logged_in:
            self._login()
            now = self._now()
        # Another process may have queried while this one logged in: check
        # the record again, and keep what it holds.
        ledger = self._read_ledger()
        self._check_interval(query, ledger, now)
        self._write_ledger({**ledger, query.request_class: now})
        data = self._query(url)
        value = read(data)
        digest = store_bytes(target, data)
        meta = CopyMeta(url=url, sha256=digest, retrieved=now, checked=now)
        write_meta(target, meta)
        return value, _retrieval(target, meta)

    def fetch_catalog(self, *, max_age: float | None = None) -> Catalog:
        """The full GP catalog (:data:`CATALOG`), from the cache or Space-Track.

        Parameters
        ----------
        max_age
            Seconds a cached copy is used; one hour (the ``gp`` interval)
            if ``None``, and not less.

        Returns
        -------
        Catalog
            The newest element set of each object, by catalog number, in one
            group named :data:`GROUP`.

        Raises
        ------
        ValueError, RateLimited, AuthError, SourceUnavailable, SourceFormatError
            As :meth:`fetch`.
        """
        omms, retrieval = self.fetch(CATALOG, read_catalog, max_age=max_age)
        kept, dropped = deduplicate(omms)
        ordered = tuple(sorted(kept, key=lambda omm: omm.catalog_number))
        return Catalog(ordered, dropped, (GpGroup(GROUP, omms, retrieval),))

    def close(self) -> None:
        """Log out, if logged in and not stopped by a rate-limit answer.

        A failed logout is ignored: the session expires on its own.
        """
        if not self._logged_in or self._stopped is not None:
            return
        self._logged_in = False
        with contextlib.suppress(SourceUnavailable):
            self._send(Request("GET", LOGOUT_URL))

    def _check_interval(
        self, query: Query, ledger: Mapping[str, float], now: float
    ) -> None:
        """Raise :class:`~keepout.errors.RateLimited` if ``query``'s class was
        asked for less than its interval before ``now``, by ``ledger``."""
        interval = query.interval
        last = ledger.get(query.request_class)
        # A last query "in the future" (the clock was set back) also waits.
        if last is not None and now - last < interval:
            wait = interval - (now - last)
            raise RateLimited(
                f"Space-Track allows one {query.request_class} query per "
                f"{interval / 3600:g} h; the last was sent at {_utc(last)} "
                f"(recorded in {self.ledger}); try again in {_duration(wait)}",
                retry_after=wait,
            )

    def _check_running(self) -> None:
        if self._stopped is not None:
            raise RateLimited(
                f"this client stopped after Space-Track's rate-limit answer: "
                f"{self._stopped}"
            )

    def _send(self, request: Request) -> Response:
        """Send ``request`` in turn; a rate-limit answer stops the client."""
        self._check_running()
        self._throttle.wait()
        response = self._transport(request)
        if response.status != HTTPStatus.OK and (
            response.status == HTTPStatus.TOO_MANY_REQUESTS
            or _RATE_LIMIT_TEXT in response.body.lower()
        ):
            self._stopped = RateLimited(
                f"Space-Track answered {request.url} with HTTP {response.status}: "
                f"{self._excerpt(response.body)}; nothing more is sent"
            )
            raise self._stopped
        return response

    def _login(self) -> None:
        if self._credentials is None:
            self._credentials = Credentials.from_env()
        credentials = self._credentials
        form = urllib.parse.urlencode(
            {"identity": credentials.identity, "password": credentials.password}
        ).encode()
        response = self._send(Request("POST", LOGIN_URL, form))
        refusal = _login_refusal(response.body)
        if response.status != HTTPStatus.OK or refusal is not None:
            reason = refusal or self._excerpt(response.body)
            raise AuthError(
                f"Space-Track refused the login (HTTP {response.status}): "
                f"{credentials.redact(reason)}"
            )
        self._logged_in = True

    def _query(self, url: str) -> bytes:
        """The body of Space-Track's answer to ``url``; logs in again once
        after HTTP 401."""
        response = self._send(Request("GET", url))
        if response.status == HTTPStatus.UNAUTHORIZED:
            self._logged_in = False
            self._login()
            response = self._send(Request("GET", url))
            if response.status == HTTPStatus.UNAUTHORIZED:
                raise AuthError(
                    f"Space-Track answered {url} with HTTP 401 right after a "
                    f"login: {self._excerpt(response.body)}"
                )
        if response.status != HTTPStatus.OK:
            raise HttpStatusError(
                f"Space-Track answered {url} with HTTP {response.status}: "
                f"{self._excerpt(response.body)}",
                url=url,
                status=response.status,
            )
        return response.body

    def _excerpt(self, body: bytes) -> str:
        """The start of a server answer, credentials hidden.

        The whole answer is redacted before it is cut, so no part of a
        credential crossing the cut is shown.
        """
        text = body.decode("utf-8", "replace")
        credentials = self._credentials
        excerpt = (
            _redacted_excerpt(credentials, text, _EXCERPT)
            if credentials is not None
            else text[:_EXCERPT]
        ).strip()
        return f"{excerpt} ..." if len(text) > _EXCERPT else excerpt

    def _read_ledger(self) -> dict[str, float]:
        try:
            record = json.loads(self.ledger.read_bytes())
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            raise SourceFormatError(
                f"cannot read the Space-Track request record {self.ledger}: {exc}"
            ) from exc
        queried = record.get("queried") if isinstance(record, dict) else None
        if (
            not isinstance(queried, dict)
            or record.get("format") != _LEDGER_FORMAT
            or not all(
                isinstance(sent, float) and math.isfinite(sent)
                for sent in queried.values()
            )
        ):
            raise SourceFormatError(
                f"{self.ledger} is not a Space-Track request record of this "
                "keepout version; delete it only if no query was sent in the "
                "last 8 hours"
            )
        return queried

    def _write_ledger(self, queried: Mapping[str, float]) -> None:
        record = {"format": _LEDGER_FORMAT, "queried": dict(sorted(queried.items()))}
        store_bytes(self.ledger, (json.dumps(record, indent=2) + "\n").encode())


def read_catalog(data: bytes) -> tuple[Omm, ...]:
    """The element sets of a Space-Track ``gp`` answer in JSON.

    Raises
    ------
    SourceFormatError
        If ``data`` is not UTF-8 OMM JSON (e.g. an error object instead of
        an array).
    """
    try:
        return loads_json(data.decode("utf-8"))
    except (UnicodeDecodeError, OmmParseError) as exc:
        raise SourceFormatError(
            f"Space-Track did not send OMM JSON ({exc}); it starts with {data[:60]!r}"
        ) from exc


def fetch_catalog(
    *,
    cache: Path | None = None,
    max_age: float | None = None,
    credentials: Credentials | None = None,
    transport: Transport | None = None,
    now: Callable[[], float] = time.time,
) -> Catalog:
    """The full GP catalog from Space-Track (:data:`CATALOG`), or its cache.

    Opens a :class:`SpaceTrackClient`, calls its
    :meth:`~SpaceTrackClient.fetch_catalog` and closes it (logging out).

    Parameters
    ----------
    cache
        Cache root; :func:`~keepout.sources.cache.cache_dir` if ``None``.
    max_age
        Seconds a cached copy is used; one hour if ``None``, and not less.
    credentials
        The account; from the environment if ``None`` and needed.
    transport
        Sends the requests; a new :class:`UrllibTransport` if ``None``.
    now
        Gives the current Unix time [s].

    Raises
    ------
    ValueError, RateLimited, AuthError, SourceUnavailable, SourceFormatError
        As :meth:`SpaceTrackClient.fetch`.
    """
    with SpaceTrackClient(
        credentials, cache=cache, transport=transport, now=now
    ) as client:
        return client.fetch_catalog(max_age=max_age)


def _secret_forms(credentials: Credentials) -> set[str]:
    """The identity and password as typed, path- and form-encoded."""
    return {
        form
        for secret in (credentials.identity, credentials.password)
        for form in (
            secret,
            urllib.parse.quote(secret, safe=""),
            urllib.parse.quote_plus(secret),
        )
    }


def _redacted_excerpt(credentials: Credentials, text: str, length: int) -> str:
    """The first ``length`` characters of ``text``, credentials hidden.

    The whole text is redacted before it is cut, so a secret crossing the
    cut is hidden whole; a cut text that ends in the start of a secret (as
    :meth:`Credentials.redact` spells it) loses that end too.
    """
    excerpt = credentials.redact(text)[:length]
    if len(text) <= length:
        return excerpt
    dangling = max(
        (
            size
            for form in _secret_forms(credentials)
            for size in range(1, len(form))
            if excerpt.endswith(form[:size])
        ),
        default=0,
    )
    return excerpt[: len(excerpt) - dangling]


def _login_refusal(body: bytes) -> str | None:
    """Space-Track's reason for refusing a login (``{"Login": "Failed"}``)."""
    try:
        answer = json.loads(body)
    except ValueError:
        return None
    if isinstance(answer, dict) and "Login" in answer:
        return str(answer["Login"])
    return None


def _retrieval(target: Path, meta: CopyMeta) -> Retrieval:
    return Retrieval(
        url=meta.url,
        path=target,
        retrieved=Time(meta.retrieved, format="unix", scale="utc"),
        checked=Time(meta.checked, format="unix", scale="utc"),
    )


def _error_body(error: urllib.error.HTTPError) -> bytes:
    """The body of an HTTP error answer; empty if it cannot be read."""
    try:
        with error:
            return error.read()
    except (OSError, http.client.HTTPException):
        return b""


def _utc(unix: float) -> str:
    """A Unix time for a message: ``2026-10-15 12:00:00 UTC``."""
    return f"{datetime.fromtimestamp(unix, UTC):%Y-%m-%d %H:%M:%S} UTC"


def _duration(seconds: float) -> str:
    """A wait for a message, rounded up: ``1 h 05 min``, ``12 min``, ``40 s``."""
    whole = math.ceil(seconds)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    if minutes:
        return f"{minutes} min {secs:02d} s"
    return f"{secs} s"
