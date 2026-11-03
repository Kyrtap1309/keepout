"""Fakes of the web shared by the sources, command-line and tools tests: a
downloader, ``urlopen`` and what it returns, a clock, digests and pinned
archives."""

import hashlib
import http.client
import io
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Mapping
from email.message import Message
from http import HTTPStatus
from typing import Final, TypeAlias

import pytest
from typing_extensions import override

from keepout.errors import HttpStatusError
from keepout.sources import kelvins

START: Final = 1_790_000_000.0
"""A Unix time in September 2026."""


class Clock:
    """A settable Unix time, starting at :data:`START`; pass it as a
    client's ``now``."""

    def __init__(self, now: float = START) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def digest(data: bytes) -> str:
    """The SHA-256 hex digest of ``data``, as the cache pins one."""
    return hashlib.sha256(data).hexdigest()


class FakeResponse(io.BytesIO):
    """What ``urlopen`` returns: the body, with an HTTP status."""

    def __init__(self, body: bytes = b"", status: int = 200) -> None:
        super().__init__(body)
        self.status = status
        self.reason = HTTPStatus(status).phrase


class BrokenBody(FakeResponse):
    """A body the connection drops while it is read."""

    @override
    def read(self, size: int | None = -1, /) -> bytes:
        raise http.client.IncompleteRead(b"")


Reply: TypeAlias = bytes | int | FakeResponse | Exception
"""What :class:`FakeUrlopen` answers a request with: a body (HTTP 200), an
HTTP status, a response, or a failure to raise."""


class FakeUrlopen:
    """``urllib.request.urlopen`` and ``time.sleep``, faked for a test.

    Each request gets the next of ``replies``, the last one again once they
    run out; a status from 400 up is raised as urllib raises it. The
    requests and the waits (no time passes) are recorded.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *replies: Reply) -> None:
        self.replies = replies
        self.requests: list[urllib.request.Request] = []
        self.waits: list[float] = []
        monkeypatch.setattr(urllib.request, "urlopen", self)
        monkeypatch.setattr(time, "sleep", self.waits.append)

    @property
    def urls(self) -> list[str]:
        """The URLs asked for, in order."""
        return [request.full_url for request in self.requests]

    def __call__(self, request: urllib.request.Request, timeout: float) -> FakeResponse:
        self.requests.append(request)
        reply = self.replies[min(len(self.requests), len(self.replies)) - 1]
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, bytes):
            return FakeResponse(reply)
        if isinstance(reply, int):
            if reply >= 400:
                phrase = HTTPStatus(reply).phrase
                raise urllib.error.HTTPError(
                    request.full_url, reply, phrase, Message(), None
                )
            return FakeResponse(status=reply)
        return reply


Answer: TypeAlias = bytes | int | Exception
"""A fake server's answer: a body, an HTTP status, or a failure."""


class FakeDownloader:
    """A ``download`` function: serves ``answers`` for every URL if it is
    bytes, else each URL as the mapping (kept in :attr:`answers`, which a
    test may change) says: an int is an HTTP error status, an exception is
    raised.

    Records the URLs asked for; safe to call from several threads.
    """

    def __init__(self, answers: bytes | Mapping[str, Answer]) -> None:
        self.every: bytes | None = None
        self.answers: dict[str, Answer] = {}
        if isinstance(answers, bytes):
            self.every = answers
        else:
            self.answers.update(answers)
        self.urls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.urls.append(url)
        answer = self.every if self.every is not None else self.answers[url]
        if isinstance(answer, int):
            raise HttpStatusError(f"HTTP {answer}", url=url, status=answer)
        if isinstance(answer, Exception):
            raise answer
        return answer


def zipped(members: Mapping[str, bytes]) -> bytes:
    """A zip archive holding ``members`` (name to contents)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)
    return buffer.getvalue()


def pin_kelvins(monkeypatch: pytest.MonkeyPatch, archive: bytes, table: bytes) -> None:
    """``archive`` and ``table`` made the Kelvins archive and raw table whose
    digests :mod:`keepout.sources.kelvins` pins."""
    monkeypatch.setattr(kelvins, "ARCHIVE_SHA256", digest(archive))
    monkeypatch.setattr(kelvins, "RAW_SHA256", digest(table))
