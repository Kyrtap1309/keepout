"""The HTTP layer under the Space-Track client: requests, answers and a
:mod:`urllib.request` transport that keeps the session cookie."""

from __future__ import annotations

import http.client
import http.cookiejar
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

from keepout.errors import SourceUnavailable
from keepout.sources._http import DOWNLOAD_TIMEOUT, gunzip, http_request, network_errors

if TYPE_CHECKING:
    import email.message
    from collections.abc import Iterable

__all__ = ["Request", "Response", "Transport", "UrllibTransport"]


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
    """A sender of requests, which gives their answers and keeps the session
    cookie.

    Answers of any HTTP status are returned, not raised; a failure to get
    one (network, protocol) raises :class:`~keepout.errors.SourceUnavailable`.
    """

    def __call__(self, request: Request, /) -> Response:
        """The answer to ``request``, once sent.

        Parameters
        ----------
        request
            The HTTP request to send.

        Returns
        -------
        Response
            The status, headers and body of the answer.
        """
        ...


class UrllibTransport:
    """A :class:`Transport` on :mod:`urllib.request`, with a cookie jar.

    Requests name keepout in the ``User-Agent``
    (:data:`~keepout.sources._http.USER_AGENT`); redirects are followed.

    Parameters
    ----------
    timeout
        Seconds to wait for the server.
    handlers
        More :mod:`urllib.request` handlers, e.g. a proxy handler.

    Attributes
    ----------
    timeout
        Seconds to wait for the server, as given.
    cookies
        The cookie jar of every request sent: it keeps Space-Track's session
        cookie from the login to the requests after it.
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
        """The answer to ``request``, once sent.

        Parameters
        ----------
        request
            The HTTP request to send.

        Returns
        -------
        Response
            The status, headers and body of the answer.

        Raises
        ------
        ValueError
            If the URL is not ``https``.
        SourceUnavailable
            If no answer comes (network or protocol error, timeout).
        """
        headers = {"Accept-Encoding": "gzip"}
        if request.form is not None:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        prepared = http_request(
            request.url,
            schemes=("https",),
            data=request.form,
            method=request.method,
            headers=headers,
        )
        with network_errors(f"cannot reach {request.url}"):
            try:
                with self._opener.open(prepared, timeout=self.timeout) as answer:
                    body = _decoded(request.url, answer.read(), answer.headers)
                    return Response(answer.status, body)
            except urllib.error.HTTPError as exc:
                body = _decoded(request.url, _error_body(exc), exc.headers)
                return Response(exc.code, body)


def _decoded(url: str, body: bytes, headers: email.message.Message) -> bytes:
    """``body`` as sent: decompressed if the server gzipped it
    (``Content-Encoding: gzip``, asked for by ``Accept-Encoding``).

    Raises
    ------
    SourceUnavailable
        If a gzipped body does not decompress (cut off on the way).
    """
    if headers.get("Content-Encoding", "").strip().lower() != "gzip":
        return body
    data = gunzip(body)
    if data is None:
        raise SourceUnavailable(f"{url}: the gzipped answer does not decompress")
    return data


def _error_body(error: urllib.error.HTTPError) -> bytes:
    """The body of an HTTP error answer; empty if it cannot be read."""
    try:
        with error:
            return error.read()
    except (OSError, http.client.HTTPException):
        return b""
