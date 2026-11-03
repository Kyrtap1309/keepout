"""The Space-Track session: login, the published limits and the cache."""

from __future__ import annotations

import contextlib
import json
import math
import time
import urllib.parse
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from http import HTTPStatus
from pathlib import Path
from types import TracebackType
from typing import Final, Self, TypeVar

from keepout._types import JsonValue, override
from keepout.errors import (
    AuthError,
    HttpStatusError,
    RateLimited,
    SourceUnavailable,
)
from keepout.omm.celestrak import loads_json
from keepout.omm.model import Omm
from keepout.sources._format import omm_reader
from keepout.sources._store import cache_dir
from keepout.sources.cache import cached_fetch
from keepout.sources.catalog import Catalog, CatalogPart, Retrieval
from keepout.sources.spacetrack._credentials import Credentials, redacted_excerpt
from keepout.sources.spacetrack._ledger import LEDGER_FILE, RequestLedger
from keepout.sources.spacetrack._query import (
    CATALOG,
    LOGIN_URL,
    LOGOUT_URL,
    PUBLIC_CDMS,
    Query,
)
from keepout.sources.spacetrack._throttle import Throttle
from keepout.sources.spacetrack._transport import (
    Request,
    Response,
    Transport,
    UrllibTransport,
)
from keepout.sources.spacetrack.public_cdm import PublicCdms, read_public_cdms
from keepout.units import HOUR, MINUTE

__all__ = [
    "SpaceTrackClient",
    "fetch_catalog",
    "fetch_public_cdms",
    "read_catalog",
]

_RATE_LIMIT_TEXT: Final = b"violated your query rate limit"
"""What Space-Track's answer says when a query broke the per-minute limit."""

_ANSWER_EXCERPT_CHARS: Final = 200
"""Characters of a server answer quoted in the error of a refused request."""

_T = TypeVar("_T")


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
        Cache root; if ``None``, ``$KEEPOUT_CACHE_DIR`` or else the
        platform's user cache directory for keepout.
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
        self._ledger = RequestLedger(self._root / LEDGER_FILE)
        self._transport = UrllibTransport() if transport is None else transport
        self._throttle = Throttle() if throttle is None else throttle
        self._now = now
        self._logged_in = False
        self._stopped: RateLimited | None = None

    @override
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
        return self._ledger.path

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

        def send() -> bytes:
            self._reserve(query)
            return self._query(query.url)

        target = self._root / query.request_class / f"{query.name}.json.gz"
        value, meta = cached_fetch(
            target,
            query.url,
            read,
            send,
            max_age=interval if max_age is None else max_age,
            interval=interval,
            reason=_allowance(query),
            now=self._now,
            compressed=True,
        )
        return value, Retrieval.of(target, meta)

    def fetch_catalog(self, *, max_age: float | None = None) -> Catalog:
        """The full GP catalog (:data:`CATALOG`), from the cache or Space-Track.

        Parameters
        ----------
        max_age
            Seconds a cached copy is used; the ``gp`` interval of
            :data:`~keepout.sources.spacetrack.CLASS_INTERVALS` (1 h) if
            ``None``, and not less.

        Returns
        -------
        Catalog
            The newest element set of each object, by catalog number, from
            one part named after the query class (``gp``).

        Raises
        ------
        ValueError, RateLimited, AuthError, SourceUnavailable, SourceFormatError
            As :meth:`fetch`.
        """
        omms, retrieval = self.fetch(CATALOG, read_catalog, max_age=max_age)
        return Catalog.of((CatalogPart(CATALOG.request_class, omms, retrieval),))

    def fetch_public_cdms(self, *, max_age: float | None = None) -> PublicCdms:
        """The public CDM summaries of the approaches still to come
        (:data:`PUBLIC_CDMS`), from the cache or Space-Track.

        Parameters
        ----------
        max_age
            Seconds a cached copy is used; the ``cdm_public`` interval of
            :data:`~keepout.sources.spacetrack.CLASS_INTERVALS` (8 h) if
            ``None``, and not less.

        Raises
        ------
        ValueError, RateLimited, AuthError, SourceUnavailable, SourceFormatError
            As :meth:`fetch`.
        """
        cdms, retrieval = self.fetch(PUBLIC_CDMS, read_public_cdms, max_age=max_age)
        return PublicCdms(cdms, retrieval)

    def close(self) -> None:
        """Log out, if logged in and not stopped by a rate-limit answer.

        A failed logout is ignored: the session expires on its own.
        """
        if not self._logged_in or self._stopped is not None:
            return
        self._logged_in = False
        with contextlib.suppress(SourceUnavailable):
            self._send(Request("GET", LOGOUT_URL))

    def _reserve(self, query: Query) -> None:
        """Record in the ledger that ``query`` is sent now, after checking
        its class's interval, logging in if needed, and checking again.

        Another process may have queried while this one logged in, so the
        ledger is read again after the login and what it holds is kept.

        Raises
        ------
        RateLimited
            If the class's interval has not passed (before or after the
            login), or this client was stopped by a rate-limit answer.
        AuthError
            If there are no credentials or Space-Track refuses them.
        SourceUnavailable
            If the login gets no answer.
        SourceFormatError
            If the ledger is not one keepout wrote.
        """
        now = self._now()
        self._check_interval(query, self._ledger.read(), now)
        self._check_running()
        if not self._logged_in:
            self._login()
            now = self._now()
        ledger = self._ledger.read()
        self._check_interval(query, ledger, now)
        self._ledger.write({**ledger, query.request_class: now})

    def _check_interval(
        self, query: Query, ledger: Mapping[str, float], now: float
    ) -> None:
        """A check that ``query``'s class was not asked for less than its
        interval before ``now``, by ``ledger``.

        Raises
        ------
        RateLimited
            If ``ledger`` records a query of the class less than its
            interval before ``now`` (or after it).
        """
        interval = query.interval
        last = ledger.get(query.request_class)
        # A last query "in the future" (the clock was set back) also waits.
        if last is not None and now - last < interval:
            wait = interval - (now - last)
            raise RateLimited(
                f"{_allowance(query)}; the last was sent at {_utc(last)} "
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
        """The answer to ``request``, sent in turn; a rate-limit answer stops
        the client."""
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
        after HTTP 401.

        Raises
        ------
        AuthError
            If Space-Track answers HTTP 401 again right after the new login.
        HttpStatusError
            If the answer has an HTTP status other than 200.
        """
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
            redacted_excerpt(credentials, text, _ANSWER_EXCERPT_CHARS)
            if credentials is not None
            else text[:_ANSWER_EXCERPT_CHARS]
        ).strip()
        return f"{excerpt} ..." if len(text) > _ANSWER_EXCERPT_CHARS else excerpt


def _allowance(query: Query) -> str:
    """What Space-Track allows of ``query``'s class, for messages: ``Space-Track
    allows one gp query per 1 h``."""
    return (
        f"Space-Track allows one {query.request_class} query per "
        f"{query.interval / HOUR:g} h"
    )


_read_gp: Final = omm_reader(loads_json, "Space-Track", "OMM JSON")


def read_catalog(data: bytes) -> tuple[Omm, ...]:
    """The element sets of a Space-Track ``gp`` answer in JSON.

    Raises
    ------
    SourceFormatError
        If ``data`` is not UTF-8 OMM JSON (e.g. an error object instead of
        an array).
    """
    return _read_gp(data)


def fetch_public_cdms(
    *,
    cache: Path | None = None,
    max_age: float | None = None,
    credentials: Credentials | None = None,
    transport: Transport | None = None,
    now: Callable[[], float] = time.time,
) -> PublicCdms:
    """The public CDM summaries of the approaches still to come
    (:data:`PUBLIC_CDMS`), from Space-Track or its cache.

    Opens a :class:`SpaceTrackClient`, calls its
    :meth:`~SpaceTrackClient.fetch_public_cdms` and closes it (logging out).
    The rows may not be redistributed without Space-Track's approval.

    Parameters
    ----------
    cache
        Cache root; if ``None``, ``$KEEPOUT_CACHE_DIR`` or else the
        platform's user cache directory for keepout.
    max_age
        Seconds a cached copy is used; the ``cdm_public`` interval of
        :data:`~keepout.sources.spacetrack.CLASS_INTERVALS` (8 h) if
        ``None``, and not less.
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
    return _in_session(
        lambda client: client.fetch_public_cdms(max_age=max_age),
        cache=cache,
        credentials=credentials,
        transport=transport,
        now=now,
    )


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
        Cache root; if ``None``, ``$KEEPOUT_CACHE_DIR`` or else the
        platform's user cache directory for keepout.
    max_age
        Seconds a cached copy is used; the ``gp`` interval of
        :data:`~keepout.sources.spacetrack.CLASS_INTERVALS` (1 h) if
        ``None``, and not less.
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
    return _in_session(
        lambda client: client.fetch_catalog(max_age=max_age),
        cache=cache,
        credentials=credentials,
        transport=transport,
        now=now,
    )


def _in_session(
    fetch: Callable[[SpaceTrackClient], _T],
    *,
    cache: Path | None,
    credentials: Credentials | None,
    transport: Transport | None,
    now: Callable[[], float],
) -> _T:
    """``fetch`` on a new :class:`SpaceTrackClient`, closed (logged out) after."""
    with SpaceTrackClient(
        credentials, cache=cache, transport=transport, now=now
    ) as client:
        return fetch(client)


def _login_refusal(body: bytes) -> str | None:
    """Space-Track's reason for refusing a login (``{"Login": "Failed"}``)."""
    try:
        answer: JsonValue = json.loads(body)
    except ValueError:
        return None
    if isinstance(answer, dict) and "Login" in answer:
        return str(answer["Login"])
    return None


def _utc(unix: float) -> str:
    """A Unix time for a message: ``2026-10-15 12:00:00 UTC``."""
    return f"{datetime.fromtimestamp(unix, UTC):%Y-%m-%d %H:%M:%S} UTC"


def _duration(seconds: float) -> str:
    """A wait for a message, rounded up: ``1 h 05 min``, ``12 min 00 s``, ``40 s``."""
    whole = math.ceil(seconds)
    hours, rest = divmod(whole, int(HOUR))
    minutes, secs = divmod(rest, int(MINUTE))
    if hours:
        return f"{hours} h {minutes:02d} min"
    if minutes:
        return f"{minutes} min {secs:02d} s"
    return f"{secs} s"
