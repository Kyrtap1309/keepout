"""HTTP for the data sources: requests that name keepout, streamed and
retried downloads.

Every request names keepout in its ``User-Agent`` (:data:`USER_AGENT`);
:func:`http_stream` streams a body into a :class:`Sink`, and
:func:`http_download` downloads one into memory, trying again after a
transient failure (:func:`retrying`).
"""

from __future__ import annotations

import contextlib
import gzip
import http.client
import io
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections.abc import Callable, Collection, Generator, Mapping, Sequence
from http import HTTPStatus
from importlib.metadata import version
from typing import Final, Protocol, TypeAlias, TypeVar

from keepout.errors import HttpStatusError, SourceUnavailable

__all__ = [
    "CHUNK_SIZE",
    "DOWNLOAD_TIMEOUT",
    "HOME_PAGE",
    "RETRY_DELAYS",
    "USER_AGENT",
    "Downloader",
    "Sink",
    "gunzip",
    "http_download",
    "http_request",
    "http_stream",
    "network_errors",
    "retrying",
]

_T = TypeVar("_T")

DOWNLOAD_TIMEOUT: Final = 60.0
"""Seconds :func:`http_download` and :func:`http_stream` wait for the server."""

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
    """A context in which a network or protocol error becomes
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
    """Download ``url`` into ``sink``, in chunks of :data:`CHUNK_SIZE` bytes.

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


def _transient(error: SourceUnavailable) -> bool:
    """Whether a download that failed with ``error`` may succeed if tried again."""
    if not isinstance(error, HttpStatusError):
        return type(error) is SourceUnavailable
    return error.status == HTTPStatus.TOO_MANY_REQUESTS or error.status >= 500


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


Downloader: TypeAlias = Callable[[str], bytes]
"""A download of a URL's body, raising
:class:`~keepout.errors.HttpStatusError` for an HTTP status other than 200
and :class:`~keepout.errors.SourceUnavailable` for other failures, like
:func:`http_download`."""


def http_download(url: str, *, retry_delays: Sequence[float] = RETRY_DELAYS) -> bytes:
    """The body at ``url``, downloaded as :func:`http_stream` does.

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


def gunzip(data: bytes) -> bytes | None:
    """``data`` decompressed, or ``None`` if it is not intact gzip data."""
    try:
        return gzip.decompress(data)
    except (OSError, EOFError, zlib.error):
        return None
