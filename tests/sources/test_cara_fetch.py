import hashlib
import http.client
import io
import re
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path

import pytest

from keepout.errors import ChecksumMismatch, SourceUnavailable
from keepout.sources import cara
from keepout.sources.cara import CaraFile, fetch

PAYLOAD = b"cara reference bytes"
FILE = CaraFile("Dir/Sub/data.mat", hashlib.sha256(PAYLOAD).hexdigest())


class FakeDownloader:
    """Serves ``data``, or each URL its own bytes from a mapping, and records
    the URLs asked for; safe to call from several threads."""

    def __init__(self, data: bytes | dict[str, bytes]) -> None:
        self.data = data
        self.urls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.urls.append(url)
        return self.data if isinstance(self.data, bytes) else self.data[url]


def cached_path(root: Path, file: CaraFile) -> Path:
    return root / "cara" / cara.CARA_COMMIT / file.path


def test_url_is_pinned_to_commit() -> None:
    assert FILE.url == (
        "https://raw.githubusercontent.com/nasa/CARA_Analysis_Tools/"
        f"{cara.CARA_COMMIT}/Dir/Sub/data.mat"
    )


def test_manifest_entries_are_well_formed() -> None:
    for entry in cara.MANIFEST:
        assert entry.path.startswith(("DistributedMatlab/", "DataFiles/"))
        assert re.fullmatch(r"[0-9a-f]{64}", entry.sha256)


def test_fetch_downloads_into_cache(tmp_path: Path) -> None:
    download = FakeDownloader(PAYLOAD)
    path = fetch(FILE, cache=tmp_path, download=download)
    assert path == cached_path(tmp_path, FILE)
    assert path.read_bytes() == PAYLOAD
    assert download.urls == [FILE.url]
    assert [p.name for p in path.parent.iterdir()] == [path.name]


def test_fetch_reuses_valid_cache(tmp_path: Path) -> None:
    fetch(FILE, cache=tmp_path, download=FakeDownloader(PAYLOAD))
    download = FakeDownloader(b"must not be used")
    fetch(FILE, cache=tmp_path, download=download)
    assert download.urls == []


def test_fetch_replaces_corrupted_cache(tmp_path: Path) -> None:
    target = cached_path(tmp_path, FILE)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"truncated")
    download = FakeDownloader(PAYLOAD)
    assert fetch(FILE, cache=tmp_path, download=download).read_bytes() == PAYLOAD
    assert download.urls == [FILE.url]


def test_fetch_rejects_checksum_mismatch_without_writing(tmp_path: Path) -> None:
    with pytest.raises(ChecksumMismatch, match="expected sha256"):
        fetch(FILE, cache=tmp_path, download=FakeDownloader(b"tampered"))
    assert not [p for p in tmp_path.rglob("*") if p.is_file()]


def test_failed_write_leaves_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(self: Path, target: Path) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", refuse)
    with pytest.raises(OSError, match="disk full"):
        fetch(FILE, cache=tmp_path, download=FakeDownloader(PAYLOAD))
    assert not [p for p in tmp_path.rglob("*") if p.is_file()]


def test_prefetch_caches_every_file(tmp_path: Path) -> None:
    payloads = [b"first", b"second", b"third"]
    files = [
        CaraFile(f"Dir/{i}.csv", hashlib.sha256(data).hexdigest())
        for i, data in enumerate(payloads)
    ]
    by_url = {file.url: data for file, data in zip(files, payloads, strict=True)}
    download = FakeDownloader(by_url)
    cara.prefetch(files, cache=tmp_path, download=download)
    assert sorted(download.urls) == sorted(by_url)
    for file, data in zip(files, payloads, strict=True):
        assert cached_path(tmp_path, file).read_bytes() == data


def test_prefetch_reports_a_failed_file(tmp_path: Path) -> None:
    with pytest.raises(ChecksumMismatch):
        cara.prefetch([FILE], cache=tmp_path, download=FakeDownloader(b"tampered"))


def test_cache_dir_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("KEEPOUT_CACHE_DIR", str(tmp_path / "custom"))
    assert cara.cache_dir() == tmp_path / "custom"


def test_cache_dir_uses_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("KEEPOUT_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert cara.cache_dir() == tmp_path / "keepout"


def test_cache_dir_defaults_to_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("KEEPOUT_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert cara.cache_dir() == tmp_path / ".cache" / "keepout"


def test_fetch_uses_cache_dir_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("KEEPOUT_CACHE_DIR", str(tmp_path / "not-yet-created"))
    path = fetch(FILE, download=FakeDownloader(PAYLOAD))
    assert path == cached_path(tmp_path / "not-yet-created", FILE)


def test_http_download_returns_body(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(url: str, timeout: float) -> io.BytesIO:
        return io.BytesIO(b"body")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert cara.http_download("https://example.invalid/x") == b"body"


def test_http_download_wraps_network_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def offline(url: str, timeout: float) -> io.BytesIO:
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", offline)
    with pytest.raises(SourceUnavailable, match=r"example\.invalid.*offline"):
        cara.http_download("https://example.invalid/x")


def test_http_download_wraps_http_404(monkeypatch: pytest.MonkeyPatch) -> None:
    def not_found(url: str, timeout: float) -> io.BytesIO:
        raise urllib.error.HTTPError(url, 404, "Not Found", Message(), None)

    monkeypatch.setattr(urllib.request, "urlopen", not_found)
    with pytest.raises(SourceUnavailable, match="404"):
        cara.http_download("https://example.invalid/x")


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
        cara.http_download("https://example.invalid/x")


def test_http_download_wraps_connection_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reset(url: str, timeout: float) -> io.BytesIO:
        raise ConnectionResetError("reset by peer")

    monkeypatch.setattr(urllib.request, "urlopen", reset)
    with pytest.raises(SourceUnavailable, match="reset by peer"):
        cara.http_download("https://example.invalid/x")


def test_manifest_reads_text_and_binary_sha256sum_lines() -> None:
    digest = "0" * 64
    text = f"{digest}  a/one.txt\n\n{'f' * 64} *b/two.xls\n"
    assert cara._parse_manifest(text) == {"a/one.txt": digest, "b/two.xls": "f" * 64}


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("not a digest  path\n", "line 1 is not sha256sum output"),
        (f"{'0' * 64}  \n", "line 1 is not sha256sum output"),
        (f"{'0' * 64}  p\n{'1' * 64}  p\n", "line 2 repeats 'p'"),
    ],
)
def test_malformed_manifest_is_rejected(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        cara._parse_manifest(text)
