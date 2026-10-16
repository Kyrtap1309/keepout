import hashlib
import http.client
import io
import sys
import urllib.error
import urllib.request
from email.message import Message
from importlib.metadata import version
from pathlib import Path

import platformdirs
import pytest

from keepout.errors import ChecksumMismatch, HttpStatusError, SourceUnavailable
from keepout.sources import cache

PAYLOAD = b"reference bytes"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


def test_cache_dir_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("KEEPOUT_CACHE_DIR", str(tmp_path / "custom"))
    assert cache.cache_dir() == tmp_path / "custom"
    assert cache.cache_dir(tmp_path / "given") == tmp_path / "given"


linux_only = pytest.mark.skipif(
    sys.platform != "linux", reason="XDG cache directories are Linux's"
)


def test_cache_dir_defaults_to_the_platforms_user_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KEEPOUT_CACHE_DIR", raising=False)
    assert cache.cache_dir() == platformdirs.user_cache_path("keepout")


@linux_only
def test_cache_dir_uses_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("KEEPOUT_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert cache.cache_dir() == tmp_path / "keepout"


@linux_only
def test_cache_dir_defaults_to_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("KEEPOUT_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert cache.cache_dir() == tmp_path / ".cache" / "keepout"


def test_the_user_agent_names_keepout_its_version_and_home_page() -> None:
    assert (
        f"keepout/{version('keepout')} (+https://github.com/Kyrtap1309/keepout)"
    ) == cache.USER_AGENT


class _Response(io.BytesIO):
    """What ``urlopen`` returns: the body, with an HTTP status."""

    def __init__(self, body: bytes = b"", status: int = 200) -> None:
        super().__init__(body)
        self.status = status
        self.reason = "OK" if status == 200 else "No Content"


def test_http_download_returns_body(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(url: str, timeout: float) -> io.BytesIO:
        return _Response(b"body")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert cache.http_download("https://example.invalid/x") == b"body"


def test_every_request_names_keepout_in_the_user_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[urllib.request.Request] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> io.BytesIO:
        requests.append(request)
        return _Response(b"body")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    cache.http_download("https://example.invalid/x")
    cache.http_stream("https://example.invalid/y", io.BytesIO())
    assert [request.full_url for request in requests] == [
        "https://example.invalid/x",
        "https://example.invalid/y",
    ]
    for request in requests:
        assert request.header_items() == [("User-agent", cache.USER_AGENT)]


def test_http_download_wraps_network_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def offline(url: str, timeout: float) -> io.BytesIO:
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", offline)
    with pytest.raises(SourceUnavailable, match=r"example\.invalid.*offline"):
        cache.http_download("https://example.invalid/x")


def test_http_download_wraps_http_404(monkeypatch: pytest.MonkeyPatch) -> None:
    def not_found(url: str, timeout: float) -> io.BytesIO:
        raise urllib.error.HTTPError(url, 404, "Not Found", Message(), None)

    monkeypatch.setattr(urllib.request, "urlopen", not_found)
    with pytest.raises(HttpStatusError, match="404") as caught:
        cache.http_download("https://example.invalid/x")
    assert caught.value.status == 404
    assert caught.value.url == "https://example.invalid/x"


def test_http_download_refuses_a_success_other_than_200(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda url, timeout: _Response(b"", status=204)
    )
    with pytest.raises(HttpStatusError, match="HTTP 204 No Content") as caught:
        cache.http_download("https://example.invalid/x")
    assert caught.value.status == 204


class _BrokenBody(_Response):
    def read(self, size: int | None = -1) -> bytes:
        raise http.client.IncompleteRead(b"partial")


def test_http_download_wraps_mid_read_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def dropped(url: str, timeout: float) -> io.BytesIO:
        return _BrokenBody()

    monkeypatch.setattr(urllib.request, "urlopen", dropped)
    with pytest.raises(SourceUnavailable, match=r"example\.invalid"):
        cache.http_download("https://example.invalid/x")


def test_http_download_wraps_connection_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reset(url: str, timeout: float) -> io.BytesIO:
        raise ConnectionResetError("reset by peer")

    monkeypatch.setattr(urllib.request, "urlopen", reset)
    with pytest.raises(SourceUnavailable, match="reset by peer"):
        cache.http_download("https://example.invalid/x")


def test_http_stream_writes_the_body_in_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = bytes(range(256)) * 10
    monkeypatch.setattr(cache, "CHUNK_SIZE", 1000)
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: _Response(body))
    chunks: list[bytes] = []

    class Collect:
        def write(self, data: bytes, /) -> object:
            chunks.append(data)
            return len(data)

    cache.http_stream("https://example.invalid/x", Collect())
    assert b"".join(chunks) == body
    assert [len(chunk) for chunk in chunks] == [1000, 1000, 560]


def test_http_stream_wraps_mid_read_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: _BrokenBody())
    with pytest.raises(SourceUnavailable, match=r"example\.invalid"):
        cache.http_stream("https://example.invalid/x", io.BytesIO())


def test_store_writes_a_verified_file_and_its_parents(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "file.bin"
    cache.store(target, lambda sink: sink.write(PAYLOAD), DIGEST, "test")
    assert target.read_bytes() == PAYLOAD
    assert [p.name for p in target.parent.iterdir()] == ["file.bin"]


def test_store_keeps_the_old_file_on_a_checksum_mismatch(tmp_path: Path) -> None:
    target = tmp_path / "file.bin"
    target.write_bytes(b"previous")
    with pytest.raises(ChecksumMismatch, match=r"source-name: expected sha256"):
        cache.store(target, lambda sink: sink.write(b"tampered"), DIGEST, "source-name")
    assert target.read_bytes() == b"previous"
    assert [p.name for p in tmp_path.iterdir()] == ["file.bin"]


def test_store_removes_the_partial_file_when_writing_fails(tmp_path: Path) -> None:
    def broken(sink: cache.Sink) -> None:
        sink.write(b"half")
        raise SourceUnavailable("connection dropped")

    with pytest.raises(SourceUnavailable, match="connection dropped"):
        cache.store(tmp_path / "file.bin", broken, DIGEST, "test")
    assert list(tmp_path.iterdir()) == []


def test_a_stored_file_is_not_hashed_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "file.bin"
    cache.store(target, lambda sink: sink.write(PAYLOAD), DIGEST, "test")

    def refuse(*args: object) -> object:
        raise AssertionError("hashed from disk")

    monkeypatch.setattr(hashlib, "file_digest", refuse)
    assert cache.has_digest(target, DIGEST)


def test_a_changed_file_is_hashed_again(tmp_path: Path) -> None:
    target = tmp_path / "file.bin"
    cache.store(target, lambda sink: sink.write(PAYLOAD), DIGEST, "test")
    target.write_bytes(b"tampered!")
    assert not cache.has_digest(target, DIGEST)


def test_has_digest_rejects_a_missing_or_different_file(tmp_path: Path) -> None:
    target = tmp_path / "file.bin"
    assert not cache.has_digest(target, DIGEST)
    target.write_bytes(b"other")
    assert not cache.has_digest(target, DIGEST)
    assert not cache.has_digest(tmp_path, DIGEST)


def test_store_bytes_writes_the_file_and_gives_its_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "a" / "file.bin"
    target.parent.mkdir()
    target.write_bytes(b"previous")
    assert cache.store_bytes(target, PAYLOAD) == DIGEST
    assert target.read_bytes() == PAYLOAD
    assert [p.name for p in target.parent.iterdir()] == ["file.bin"]

    def refuse(*args: object) -> object:
        raise AssertionError("hashed from disk")

    monkeypatch.setattr(hashlib, "file_digest", refuse)
    assert cache.has_digest(target, DIGEST)


def test_store_bytes_creates_the_parents(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "file.bin"
    cache.store_bytes(target, b"")
    assert target.read_bytes() == b""


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.invalid/x"])
def test_only_http_urls_are_opened(url: str) -> None:
    with pytest.raises(ValueError, match="not an HTTP URL"):
        cache.http_stream(url, io.BytesIO())


def test_a_cached_copy_is_found_with_its_metadata(tmp_path: Path) -> None:
    target = tmp_path / "source" / "data.json"
    meta = cache.CopyMeta(
        url="https://example.org/data",
        sha256=cache.store_bytes(target, PAYLOAD),
        retrieved=1_790_000_000.0,
        checked=1_790_000_060.0,
    )
    cache.write_meta(target, meta)
    assert cache.meta_path(target) == tmp_path / "source" / "data.json.meta.json"
    assert cache.read_copy(target, meta.url) == cache.CachedCopy(PAYLOAD, meta)
    assert cache.read_copy(target, "https://example.org/other") is None
    target.write_bytes(b"changed")
    assert cache.read_copy(target, meta.url) is None
