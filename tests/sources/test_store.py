import hashlib
import os
import sys
from pathlib import Path

import platformdirs
import pytest
from web_builders import digest

from keepout.errors import ChecksumMismatch, SourceUnavailable
from keepout.sources import _http, _store

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


def test_store_json_writes_indented_text_with_a_final_newline(tmp_path: Path) -> None:
    path = tmp_path / "index.json"
    written = _store.store_json(path, {"format": 1})
    assert path.read_text(encoding="utf-8") == '{\n  "format": 1\n}\n'
    assert written == digest(path.read_bytes())
