"""Fakes of the web shared by the sources, command-line and tools tests: a
downloader, what ``urlopen`` returns, a clock, digests and pinned archives."""

import hashlib
import io
import zipfile
from collections.abc import Mapping
from typing import Final, TypeAlias

import pytest

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
        self.reason = "OK" if status == 200 else "No Content"


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
    """Make ``archive`` and ``table`` the Kelvins archive and raw table
    whose digests :mod:`keepout.sources.kelvins` pins."""
    monkeypatch.setattr(kelvins, "ARCHIVE_SHA256", digest(archive))
    monkeypatch.setattr(kelvins, "RAW_SHA256", digest(table))
