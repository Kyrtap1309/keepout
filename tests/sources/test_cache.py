import gzip
import hashlib
import http.client
import io
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from email.message import Message
from importlib.metadata import version
from pathlib import Path

import platformdirs
import pytest
from typing_extensions import override
from web_builders import FakeResponse, digest

from keepout.errors import (
    AuthError,
    ChecksumMismatch,
    HttpStatusError,
    SourceFormatError,
    SourceUnavailable,
)
from keepout.sources import _http, _store, cache

PAYLOAD = b"reference bytes"
DIGEST = digest(PAYLOAD)


def test_cache_dir_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("KEEPOUT_CACHE_DIR", str(tmp_path / "custom"))
    assert _store.cache_dir() == tmp_path / "custom"
    assert _store.cache_dir(tmp_path / "given") == tmp_path / "given"


linux_only = pytest.mark.skipif(
    sys.platform != "linux", reason="XDG cache directories are Linux's"
)


def test_cache_dir_defaults_to_the_platforms_user_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KEEPOUT_CACHE_DIR", raising=False)
    assert _store.cache_dir() == platformdirs.user_cache_path("keepout")


@linux_only
def test_cache_dir_uses_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("KEEPOUT_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert _store.cache_dir() == tmp_path / "keepout"


@linux_only
def test_cache_dir_defaults_to_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("KEEPOUT_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert _store.cache_dir() == tmp_path / ".cache" / "keepout"


def test_the_user_agent_names_keepout_its_version_and_home_page() -> None:
    assert (
        f"keepout/{version('keepout')} (+https://github.com/Kyrtap1309/keepout)"
    ) == _http.USER_AGENT


def test_http_download_returns_body(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(url: str, timeout: float) -> io.BytesIO:
        return FakeResponse(b"body")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert _http.http_download("https://example.invalid/x") == b"body"


def test_every_request_names_keepout_in_the_user_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[urllib.request.Request] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> io.BytesIO:
        requests.append(request)
        return FakeResponse(b"body")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    _http.http_download("https://example.invalid/x")
    _http.http_stream("https://example.invalid/y", io.BytesIO())
    assert [request.full_url for request in requests] == [
        "https://example.invalid/x",
        "https://example.invalid/y",
    ]
    for request in requests:
        assert request.header_items() == [("User-agent", _http.USER_AGENT)]


def test_http_download_wraps_network_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def offline(url: str, timeout: float) -> io.BytesIO:
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", offline)
    with pytest.raises(SourceUnavailable, match=r"example\.invalid.*offline"):
        _http.http_download("https://example.invalid/x", retry_delays=())


def test_http_download_wraps_http_404(monkeypatch: pytest.MonkeyPatch) -> None:
    def not_found(url: str, timeout: float) -> io.BytesIO:
        raise urllib.error.HTTPError(url, 404, "Not Found", Message(), None)

    monkeypatch.setattr(urllib.request, "urlopen", not_found)
    with pytest.raises(HttpStatusError, match="404") as caught:
        _http.http_download("https://example.invalid/x")
    assert caught.value.status == 404
    assert caught.value.url == "https://example.invalid/x"


def test_http_download_refuses_a_success_other_than_200(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda url, timeout: FakeResponse(b"", status=204)
    )
    with pytest.raises(HttpStatusError, match="HTTP 204 No Content") as caught:
        _http.http_download("https://example.invalid/x")
    assert caught.value.status == 204


class _BrokenBody(FakeResponse):
    @override
    def read(self, size: int | None = -1) -> bytes:
        raise http.client.IncompleteRead(b"partial")


def test_http_download_wraps_mid_read_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def dropped(url: str, timeout: float) -> io.BytesIO:
        return _BrokenBody()

    monkeypatch.setattr(urllib.request, "urlopen", dropped)
    with pytest.raises(SourceUnavailable, match=r"example\.invalid"):
        _http.http_download("https://example.invalid/x", retry_delays=())


def test_http_download_wraps_connection_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reset(url: str, timeout: float) -> io.BytesIO:
        raise ConnectionResetError("reset by peer")

    monkeypatch.setattr(urllib.request, "urlopen", reset)
    with pytest.raises(SourceUnavailable, match="reset by peer"):
        _http.http_download("https://example.invalid/x", retry_delays=())


def test_http_stream_writes_the_body_in_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = bytes(range(256)) * 10
    monkeypatch.setattr(_http, "CHUNK_SIZE", 1000)
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda url, timeout: FakeResponse(body)
    )
    chunks: list[bytes] = []

    class Collect:
        def write(self, data: bytes, /) -> object:
            chunks.append(data)
            return len(data)

    _http.http_stream("https://example.invalid/x", Collect())
    assert b"".join(chunks) == body
    assert [len(chunk) for chunk in chunks] == [1000, 1000, 560]


def test_http_stream_wraps_mid_read_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: _BrokenBody())
    with pytest.raises(SourceUnavailable, match=r"example\.invalid"):
        _http.http_stream("https://example.invalid/x", io.BytesIO())


def test_store_writes_a_verified_file_and_its_parents(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "file.bin"
    _store.store(target, lambda sink: sink.write(PAYLOAD), DIGEST, "test")
    assert target.read_bytes() == PAYLOAD
    assert [p.name for p in target.parent.iterdir()] == ["file.bin"]


def test_store_keeps_the_old_file_on_a_checksum_mismatch(tmp_path: Path) -> None:
    target = tmp_path / "file.bin"
    target.write_bytes(b"previous")
    with pytest.raises(ChecksumMismatch, match=r"source-name: expected sha256"):
        _store.store(
            target, lambda sink: sink.write(b"tampered"), DIGEST, "source-name"
        )
    assert target.read_bytes() == b"previous"
    assert [p.name for p in tmp_path.iterdir()] == ["file.bin"]


def test_store_removes_the_partial_file_when_writing_fails(tmp_path: Path) -> None:
    def broken(sink: _http.Sink) -> None:
        sink.write(b"half")
        raise SourceUnavailable("connection dropped")

    with pytest.raises(SourceUnavailable, match="connection dropped"):
        _store.store(tmp_path / "file.bin", broken, DIGEST, "test")
    assert list(tmp_path.iterdir()) == []


def refuse_hashing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A guard that fails the test if a file is hashed from disk."""

    def refuse(*args: object) -> object:
        raise AssertionError("hashed from disk")

    monkeypatch.setattr(hashlib, "file_digest", refuse)


def test_a_stored_file_is_not_hashed_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "file.bin"
    _store.store(target, lambda sink: sink.write(PAYLOAD), DIGEST, "test")
    refuse_hashing(monkeypatch)
    assert _store.has_digest(target, DIGEST)


def test_a_changed_file_is_hashed_again(tmp_path: Path) -> None:
    target = tmp_path / "file.bin"
    _store.store(target, lambda sink: sink.write(PAYLOAD), DIGEST, "test")
    target.write_bytes(b"tampered!")
    assert not _store.has_digest(target, DIGEST)


def test_a_forgotten_file_is_hashed_again(tmp_path: Path) -> None:
    target = tmp_path / "file.bin"
    _store.store(target, lambda sink: sink.write(PAYLOAD), DIGEST, "test")
    before = target.stat()
    # Same inode, size and modification time: only hashing shows the change.
    with target.open("r+b") as stream:
        stream.write(b"X")
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert _store.has_digest(target, DIGEST)
    _store.forget_verified()
    assert not _store.has_digest(target, DIGEST)


def test_has_digest_rejects_a_missing_or_different_file(tmp_path: Path) -> None:
    target = tmp_path / "file.bin"
    assert not _store.has_digest(target, DIGEST)
    target.write_bytes(b"other")
    assert not _store.has_digest(target, DIGEST)
    assert not _store.has_digest(tmp_path, DIGEST)


def test_store_bytes_writes_the_file_and_gives_its_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "a" / "file.bin"
    target.parent.mkdir()
    target.write_bytes(b"previous")
    assert _store.store_bytes(target, PAYLOAD) == DIGEST
    assert target.read_bytes() == PAYLOAD
    assert [p.name for p in target.parent.iterdir()] == ["file.bin"]
    refuse_hashing(monkeypatch)
    assert _store.has_digest(target, DIGEST)


def test_store_bytes_creates_the_parents(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "file.bin"
    _store.store_bytes(target, b"")
    assert target.read_bytes() == b""


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.invalid/x"])
def test_only_http_urls_are_opened(url: str) -> None:
    with pytest.raises(ValueError, match="not an HTTP or HTTPS URL"):
        _http.http_stream(url, io.BytesIO())


def test_a_cached_copy_is_found_with_its_metadata(tmp_path: Path) -> None:
    target = tmp_path / "source" / "data.json"
    meta = cache.CopyMeta(
        url="https://example.org/data",
        sha256=_store.store_bytes(target, PAYLOAD),
        retrieved=1_790_000_000.0,
        checked=1_790_000_060.0,
    )
    cache.write_meta(target, meta)
    assert cache.meta_path(target) == tmp_path / "source" / "data.json.meta.json"
    assert cache.read_copy(target, meta.url) == cache.CachedCopy(PAYLOAD, meta)
    assert cache.read_copy(target, "https://example.org/other") is None
    target.write_bytes(b"changed")
    assert cache.read_copy(target, meta.url) is None


def test_a_request_names_keepout_and_adds_the_headers_given() -> None:
    request = _http.http_request(
        "https://example.org/x",
        data=b"a=1",
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert request.get_method() == "POST"
    assert request.data == b"a=1"
    assert dict(request.header_items()) == {
        "User-agent": _http.USER_AGENT,
        "Content-type": "application/x-www-form-urlencoded",
    }


@pytest.mark.parametrize(
    ("url", "schemes", "message"),
    [
        ("file:///etc/passwd", ("http", "https"), "not an HTTP or HTTPS URL"),
        ("http://example.org/", ("https",), "not an HTTPS URL"),
    ],
)
def test_a_request_refuses_other_schemes(
    url: str, schemes: tuple[str, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _http.http_request(url, schemes=schemes)


@pytest.mark.parametrize("error", [OSError("refused"), http.client.IncompleteRead(b"")])
def test_network_errors_become_source_unavailable(error: Exception) -> None:
    with (
        pytest.raises(SourceUnavailable, match=r"^cannot reach x: ") as raised,
        _http.network_errors("cannot reach x"),
    ):
        raise error
    assert raised.value.__cause__ is error


def test_other_errors_pass_through_network_errors() -> None:
    with pytest.raises(KeyError), _http.network_errors("cannot reach x"):
        raise KeyError("bug")


def test_store_json_writes_indented_text_with_a_final_newline(tmp_path: Path) -> None:
    path = tmp_path / "index.json"
    written = _store.store_json(path, {"format": 1})
    assert path.read_text(encoding="utf-8") == '{\n  "format": 1\n}\n'
    assert written == digest(path.read_bytes())


def test_gunzip_decompresses_gzip_data() -> None:
    assert _http.gunzip(gzip.compress(b"rows")) == b"rows"


@pytest.mark.parametrize("data", [b"not gzip", b"\x1f\x8b\x08truncated"])
def test_gunzip_of_something_else_is_none(data: bytes) -> None:
    assert _http.gunzip(data) is None


def test_a_dropped_download_is_tried_again(monkeypatch: pytest.MonkeyPatch) -> None:
    answers: list[Exception | bytes] = [ConnectionResetError("reset"), b"body"]

    def flaky(url: str, timeout: float) -> io.BytesIO:
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return FakeResponse(answer)

    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    assert _http.http_download("https://example.invalid/x", retry_delays=(0.0,)) == (
        b"body"
    )
    assert answers == []


def _failing(*errors: Exception) -> Callable[[], str]:
    remaining = list(errors)

    def attempt() -> str:
        if remaining:
            raise remaining.pop(0)
        return "done"

    return attempt


@pytest.mark.parametrize(
    "error",
    [
        SourceUnavailable("offline"),
        HttpStatusError("busy", url="u", status=503),
        HttpStatusError("slow down", url="u", status=429),
    ],
    ids=["network", "server error", "too many requests"],
)
def test_retrying_waits_and_tries_a_transient_failure_again(error: Exception) -> None:
    waits: list[float] = []
    assert _http.retrying(
        _failing(error, error), delays=(1.0, 2.0), sleep=waits.append
    ) == ("done")
    assert waits == [1.0, 2.0]


@pytest.mark.parametrize(
    "error",
    [
        HttpStatusError("gone", url="u", status=404),
        HttpStatusError("unchanged", url="u", status=403),
        AuthError("refused"),
    ],
    ids=["not found", "forbidden", "login refused"],
)
def test_retrying_raises_a_lasting_failure_at_once(error: Exception) -> None:
    waits: list[float] = []
    with pytest.raises(type(error)):
        _http.retrying(_failing(error), delays=(1.0,), sleep=waits.append)
    assert waits == []


def test_retrying_raises_the_last_failure_when_attempts_run_out() -> None:
    waits: list[float] = []
    attempt = _failing(SourceUnavailable("first"), SourceUnavailable("second"))
    with pytest.raises(SourceUnavailable, match="second"):
        _http.retrying(attempt, delays=(1.0,), sleep=waits.append)
    assert waits == [1.0]


URL = "https://example.org/data"


class Server:
    """A download for :func:`cache.cached_fetch` that counts its calls."""

    def __init__(self, answer: bytes | Exception = PAYLOAD) -> None:
        self.answer = answer
        self.calls = 0

    def __call__(self) -> bytes:
        self.calls += 1
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def strict(data: bytes) -> bytes:
    if not data.startswith(b"reference"):
        raise SourceFormatError(f"not reference data: {data[:10]!r}")
    return data


def fetch(
    target: Path,
    server: Server,
    now: float,
    *,
    max_age: float = 100.0,
    compressed: bool = False,
) -> tuple[bytes, cache.CopyMeta]:
    return cache.cached_fetch(
        target,
        URL,
        strict,
        server,
        max_age=max_age,
        interval=100.0,
        reason="the source updates every 100 s",
        now=lambda: now,
        compressed=compressed,
    )


def test_cached_fetch_downloads_then_uses_the_copy_until_max_age(
    tmp_path: Path,
) -> None:
    target = tmp_path / "data.csv"
    server = Server()
    value, meta = fetch(target, server, 1000.0)
    assert value == PAYLOAD
    assert meta == cache.CopyMeta(URL, DIGEST, 1000.0, 1000.0)
    assert fetch(target, server, 1099.0) == (PAYLOAD, meta)
    assert server.calls == 1
    assert fetch(target, server, 1100.0)[1].retrieved == 1100.0
    assert server.calls == 2


def test_cached_fetch_downloads_again_over_an_unreadable_copy(
    tmp_path: Path,
) -> None:
    target = tmp_path / "data.csv"
    cache.save_copy(target, URL, b"garbage", 1000.0)
    server = Server()
    assert fetch(target, server, 1001.0)[0] == PAYLOAD
    assert server.calls == 1


def test_cached_fetch_does_not_cache_a_download_it_cannot_read(
    tmp_path: Path,
) -> None:
    target = tmp_path / "data.csv"
    with pytest.raises(SourceFormatError, match="not reference data"):
        fetch(target, Server(b"garbage"), 1000.0)
    assert not target.exists()


def test_cached_fetch_keeps_a_compressed_copy(tmp_path: Path) -> None:
    target = tmp_path / "data.json.gz"
    server = Server()
    fetch(target, server, 1000.0, compressed=True)
    assert gzip.decompress(target.read_bytes()) == PAYLOAD
    assert fetch(target, server, 1001.0, compressed=True)[0] == PAYLOAD
    assert server.calls == 1


def test_cached_fetch_downloads_over_a_copy_that_does_not_decompress(
    tmp_path: Path,
) -> None:
    target = tmp_path / "data.json.gz"
    cache.save_copy(target, URL, PAYLOAD, 1000.0)
    server = Server()
    fetch(target, server, 1001.0, compressed=True)
    assert server.calls == 1


def test_unchanged_renews_the_check_of_an_old_copy(tmp_path: Path) -> None:
    target = tmp_path / "data.csv"
    fetch(target, Server(), 1000.0)
    value, meta = fetch(target, Server(cache.Unchanged("no copy")), 5000.0)
    assert value == PAYLOAD
    assert meta == cache.CopyMeta(URL, DIGEST, 1000.0, 5000.0)
    assert cache.read_copy(target, URL) == cache.CachedCopy(PAYLOAD, meta)


@pytest.mark.parametrize("cached", [None, b"garbage"])
def test_unchanged_without_a_readable_copy_is_raised(
    tmp_path: Path, cached: bytes | None
) -> None:
    target = tmp_path / "data.csv"
    if cached is not None:
        cache.save_copy(target, URL, cached, 1000.0)
    with pytest.raises(cache.Unchanged, match="no copy"):
        fetch(target, Server(cache.Unchanged("no copy")), 5000.0)


def test_unchanged_over_a_fresh_unreadable_copy_reads_it_once(
    tmp_path: Path,
) -> None:
    target = tmp_path / "data.csv"
    cache.save_copy(target, URL, b"garbage", 1000.0)
    reads: list[bytes] = []

    def counted(data: bytes) -> bytes:
        reads.append(data)
        return strict(data)

    with pytest.raises(cache.Unchanged, match="no copy"):
        cache.cached_fetch(
            target,
            URL,
            counted,
            Server(cache.Unchanged("no copy")),
            max_age=100.0,
            interval=100.0,
            reason="the source updates every 100 s",
            now=lambda: 1001.0,
        )
    assert reads == [b"garbage"]


def test_cached_fetch_refuses_a_max_age_below_the_interval(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError,
        match=r"at least 100 s \(the source updates every 100 s\), got 99",
    ):
        fetch(tmp_path / "data.csv", Server(), 1000.0, max_age=99.0)
