import gzip
from pathlib import Path

import pytest
from web_builders import digest

from keepout.errors import SourceFormatError
from keepout.sources import _store, cache

PAYLOAD = b"reference bytes"
DIGEST = digest(PAYLOAD)


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
