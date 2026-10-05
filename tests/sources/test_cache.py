import hashlib
import http.client
import io
import sys
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path

import platformdirs
import pytest

from keepout.errors import ChecksumMismatch, SourceUnavailable
from keepout.sources import cache

PAYLOAD = b"reference bytes"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture(autouse=True)
def _forget_verified(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test without the process-wide memo of verified files."""
    monkeypatch.setattr(cache, "_verified", {})


def test_cache_dir_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("KEEPOUT_CACHE_DIR", str(tmp_path / "custom"))
    assert cache.cache_dir() == tmp_path / "custom"


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


def test_http_download_returns_body(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(url: str, timeout: float) -> io.BytesIO:
        return io.BytesIO(b"body")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert cache.http_download("https://example.invalid/x") == b"body"


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
    with pytest.raises(SourceUnavailable, match="404"):
        cache.http_download("https://example.invalid/x")


class _BrokenBody(io.BytesIO):
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
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda url, timeout: io.BytesIO(body)
    )
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
