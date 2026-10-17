import dataclasses
import hashlib
import io
import json
import math
import urllib.request
from collections.abc import Callable
from pathlib import Path

import astropy.units as u
import pytest
from celestrak_builders import SAMPLE, FakeCelesTrak

from keepout.errors import HttpStatusError, SourceFormatError, SourceUnavailable
from keepout.omm import Omm
from keepout.omm.celestrak import dumps_csv
from keepout.sources import cache
from keepout.sources.celestrak import (
    PUBLIC_GROUPS,
    SOCRATES_URL,
    TTL,
    fetch_catalog,
    fetch_group,
    fetch_socrates,
    group_url,
)

ACTIVE = group_url("active")
STATIONS = group_url("stations")
ACTIVE_CSV = dumps_csv(SAMPLE[:4]).encode()
SOCRATES_CSV = (
    "NORAD_CAT_ID_1,OBJECT_NAME_1,DSE_1,NORAD_CAT_ID_2,OBJECT_NAME_2,DSE_2,"
    "TCA,TCA_RANGE,TCA_RELATIVE_SPEED,MAX_PROB,DILUTION\n"
    "25544,ISS (ZARYA),0.123,900,CALSPHERE 1,0.456,"
    "2026-10-07 12:00:00.000,1.234,14.567,1.0E-05,0.010\n"
)
START = 1_790_000_000.0
"""A Unix time in September 2026."""


class Clock:
    """A settable Unix time."""

    def __init__(self) -> None:
        self.now = START

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


def later(omm: Omm, days: float) -> Omm:
    return dataclasses.replace(omm, epoch=omm.epoch + days * u.day)


def meta_path(root: Path, name: str) -> Path:
    return root / "celestrak" / "gp" / f"{name}.csv.meta.json"


def test_group_urls_ask_for_csv() -> None:
    assert group_url("active") == (
        "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=CSV"
    )
    assert group_url("Last-30-Days").endswith("GROUP=last-30-days&FORMAT=CSV")


@pytest.mark.parametrize("name", ["", "active&FORMAT=TLE", "../x", "a--b", "-a", "a b"])
def test_a_group_name_is_checked(name: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a CelesTrak group name"):
        group_url(name)
    with pytest.raises(ValueError, match="not a CelesTrak group name"):
        fetch_group(name, cache=tmp_path, download=FakeCelesTrak({}))


def test_the_public_groups() -> None:
    assert PUBLIC_GROUPS == (
        "active",
        "last-30-days",
        "analyst",
        "fengyun-1c-debris",
        "cosmos-2251-debris",
        "iridium-33-debris",
    )
    for name in PUBLIC_GROUPS:
        assert group_url(name).endswith(f"?GROUP={name}&FORMAT=CSV")


def test_a_group_is_downloaded_read_and_cached(tmp_path: Path, clock: Clock) -> None:
    server = FakeCelesTrak({ACTIVE: ACTIVE_CSV})
    group = fetch_group("active", cache=tmp_path, download=server, now=clock)
    assert server.urls == [ACTIVE]
    assert group.name == "active"
    assert group.omms == SAMPLE[:4]
    path = tmp_path / "celestrak" / "gp" / "active.csv"
    assert group.retrieval.path == path
    assert group.retrieval.url == ACTIVE
    assert group.retrieval.retrieved.unix == START
    assert group.retrieval.checked.unix == START
    assert group.retrieval.retrieved.scale == "utc"
    assert path.read_bytes() == ACTIVE_CSV
    assert sorted(p.name for p in path.parent.iterdir()) == [
        "active.csv",
        "active.csv.meta.json",
    ]
    assert json.loads(meta_path(tmp_path, "active").read_text()) == {
        "format": 1,
        "url": ACTIVE,
        "sha256": hashlib.sha256(ACTIVE_CSV).hexdigest(),
        "retrieved": START,
        "checked": START,
    }


def test_a_copy_younger_than_the_ttl_is_used_without_asking(
    tmp_path: Path, clock: Clock
) -> None:
    fetch_group(
        "active",
        cache=tmp_path,
        download=FakeCelesTrak({ACTIVE: ACTIVE_CSV}),
        now=clock,
    )
    clock.now = START + TTL - 1.0
    server = FakeCelesTrak({})
    group = fetch_group("ACTIVE", cache=tmp_path, download=server, now=clock)
    assert server.urls == []
    assert group.omms == SAMPLE[:4]
    assert group.retrieval.checked.unix == START


def test_an_expired_copy_is_downloaded_again(tmp_path: Path, clock: Clock) -> None:
    fetch_group(
        "active",
        cache=tmp_path,
        download=FakeCelesTrak({ACTIVE: ACTIVE_CSV}),
        now=clock,
    )
    clock.now = START + TTL
    newer = dumps_csv(SAMPLE[4:]).encode()
    server = FakeCelesTrak({ACTIVE: newer})
    group = fetch_group("active", cache=tmp_path, download=server, now=clock)
    assert server.urls == [ACTIVE]
    assert group.omms == SAMPLE[4:]
    assert group.retrieval.retrieved.unix == START + TTL
    assert (tmp_path / "celestrak" / "gp" / "active.csv").read_bytes() == newer


def test_a_longer_max_age_keeps_a_copy_longer(tmp_path: Path, clock: Clock) -> None:
    fetch_group(
        "active",
        cache=tmp_path,
        download=FakeCelesTrak({ACTIVE: ACTIVE_CSV}),
        now=clock,
    )
    clock.now = START + 10 * TTL
    server = FakeCelesTrak({})
    fetch_group("active", cache=tmp_path, max_age=math.inf, download=server, now=clock)
    assert server.urls == []


@pytest.mark.parametrize("max_age", [TTL - 1.0, 0.0, -1.0, math.nan])
def test_a_max_age_below_the_ttl_is_refused(max_age: float, tmp_path: Path) -> None:
    server = FakeCelesTrak({})
    with pytest.raises(ValueError, match="max_age must be at least 7200 s"):
        fetch_group("active", cache=tmp_path, max_age=max_age, download=server)
    with pytest.raises(ValueError, match="max_age must be at least 7200 s"):
        fetch_catalog(cache=tmp_path, max_age=max_age, download=server)
    with pytest.raises(ValueError, match="max_age must be at least 7200 s"):
        fetch_socrates(cache=tmp_path, max_age=max_age, download=server)
    assert server.urls == []


def test_a_copy_checked_in_the_future_is_not_trusted(
    tmp_path: Path, clock: Clock
) -> None:
    fetch_group(
        "active",
        cache=tmp_path,
        download=FakeCelesTrak({ACTIVE: ACTIVE_CSV}),
        now=clock,
    )
    clock.now = START - 1.0
    server = FakeCelesTrak({ACTIVE: ACTIVE_CSV})
    fetch_group("active", cache=tmp_path, download=server, now=clock)
    assert server.urls == [ACTIVE]


def test_403_means_unchanged_and_returns_the_cached_copy(
    tmp_path: Path, clock: Clock
) -> None:
    fetch_group(
        "active",
        cache=tmp_path,
        download=FakeCelesTrak({ACTIVE: ACTIVE_CSV}),
        now=clock,
    )
    clock.now = START + 3 * TTL
    server = FakeCelesTrak({ACTIVE: 403})
    group = fetch_group("active", cache=tmp_path, download=server, now=clock)
    assert server.urls == [ACTIVE]
    assert group.omms == SAMPLE[:4]
    assert group.retrieval.retrieved.unix == START
    assert group.retrieval.checked.unix == START + 3 * TTL
    # The check counts like a download: CelesTrak is not asked again for a TTL.
    clock.now = START + 4 * TTL - 1.0
    fetch_group("active", cache=tmp_path, download=server, now=clock)
    assert server.urls == [ACTIVE]


def test_403_without_a_cached_copy_explains_itself(
    tmp_path: Path, clock: Clock
) -> None:
    server = FakeCelesTrak({ACTIVE: 403})
    with pytest.raises(SourceUnavailable, match=r"403.*not changed.*no usable copy"):
        fetch_group("active", cache=tmp_path, download=server, now=clock)
    assert server.urls == [ACTIVE]
    assert not (tmp_path / "celestrak" / "gp" / "active.csv").exists()


@pytest.mark.parametrize("status", [204, 301, 404, 429, 500, 503])
def test_any_other_status_raises_once_and_keeps_the_cache(
    status: int, tmp_path: Path, clock: Clock
) -> None:
    fetch_group(
        "active",
        cache=tmp_path,
        download=FakeCelesTrak({ACTIVE: ACTIVE_CSV}),
        now=clock,
    )
    meta = meta_path(tmp_path, "active").read_bytes()
    clock.now = START + TTL
    server = FakeCelesTrak({ACTIVE: status})
    with pytest.raises(HttpStatusError) as caught:
        fetch_group("active", cache=tmp_path, download=server, now=clock)
    assert caught.value.status == status
    assert server.urls == [ACTIVE]
    assert meta_path(tmp_path, "active").read_bytes() == meta


def test_a_network_failure_raises_once(tmp_path: Path, clock: Clock) -> None:
    server = FakeCelesTrak({ACTIVE: SourceUnavailable("offline")})
    with pytest.raises(SourceUnavailable, match="offline"):
        fetch_group("active", cache=tmp_path, download=server, now=clock)
    assert server.urls == [ACTIVE]


@pytest.mark.parametrize(
    "body", [b"No GP data found", b"\xff\xfe", b"OBJECT_NAME,NORAD_CAT_ID\nX,1\n"]
)
def test_a_body_that_is_not_omm_csv_is_refused_and_not_cached(
    body: bytes, tmp_path: Path, clock: Clock
) -> None:
    server = FakeCelesTrak({ACTIVE: body})
    with pytest.raises(SourceFormatError, match="did not send CelesTrak OMM CSV"):
        fetch_group("active", cache=tmp_path, download=server, now=clock)
    assert not (tmp_path / "celestrak").exists()


def _corrupt_data(root: Path) -> None:
    (root / "celestrak" / "gp" / "active.csv").write_bytes(ACTIVE_CSV + b"\n")


def _corrupt_meta(root: Path) -> None:
    meta_path(root, "active").write_text("{not json")


def _remove_meta(root: Path) -> None:
    meta_path(root, "active").unlink()


def _remove_data(root: Path) -> None:
    (root / "celestrak" / "gp" / "active.csv").unlink()


def _edit_meta(**changes: object) -> Callable[[Path], None]:
    def edit(root: Path) -> None:
        path = meta_path(root, "active")
        record = json.loads(path.read_text())
        record.update(changes)
        path.write_text(json.dumps(record))

    return edit


def _meta_list(root: Path) -> None:
    meta_path(root, "active").write_text("[]")


CORRUPTIONS: dict[str, Callable[[Path], None]] = {
    "data changed": _corrupt_data,
    "data missing": _remove_data,
    "metadata not json": _corrupt_meta,
    "metadata missing": _remove_meta,
    "metadata not an object": _meta_list,
    "another format": _edit_meta(format=2),
    "another url": _edit_meta(url=STATIONS),
    "time not a number": _edit_meta(checked="today"),
    "time an integer": _edit_meta(retrieved=1_790_000_000),
    "time not finite": _edit_meta(retrieved=math.inf),
    "digest missing": _edit_meta(sha256=None),
}


@pytest.mark.parametrize("corrupt", CORRUPTIONS.values(), ids=CORRUPTIONS.keys())
def test_a_corrupt_cached_copy_is_downloaded_again(
    corrupt: Callable[[Path], None], tmp_path: Path, clock: Clock
) -> None:
    fetch_group(
        "active",
        cache=tmp_path,
        download=FakeCelesTrak({ACTIVE: ACTIVE_CSV}),
        now=clock,
    )
    corrupt(tmp_path)
    server = FakeCelesTrak({ACTIVE: ACTIVE_CSV})
    group = fetch_group("active", cache=tmp_path, download=server, now=clock)
    assert server.urls == [ACTIVE]
    assert group.omms == SAMPLE[:4]
    assert (
        fetch_group("active", cache=tmp_path, download=server, now=clock).omms
        == SAMPLE[:4]
    )
    assert server.urls == [ACTIVE]


@pytest.mark.parametrize("corrupt", CORRUPTIONS.values(), ids=CORRUPTIONS.keys())
def test_403_with_a_corrupt_cached_copy_raises(
    corrupt: Callable[[Path], None], tmp_path: Path, clock: Clock
) -> None:
    fetch_group(
        "active",
        cache=tmp_path,
        download=FakeCelesTrak({ACTIVE: ACTIVE_CSV}),
        now=clock,
    )
    corrupt(tmp_path)
    with pytest.raises(SourceUnavailable, match="no usable copy"):
        fetch_group(
            "active", cache=tmp_path, download=FakeCelesTrak({ACTIVE: 403}), now=clock
        )


def test_a_cached_copy_keepout_can_no_longer_read_is_not_used(
    tmp_path: Path, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetch_group(
        "active",
        cache=tmp_path,
        download=FakeCelesTrak({ACTIVE: ACTIVE_CSV}),
        now=clock,
    )
    path = tmp_path / "celestrak" / "gp" / "active.csv"
    garbage = b"garbage"
    path.write_bytes(garbage)
    _edit_meta(sha256=hashlib.sha256(garbage).hexdigest())(tmp_path)
    server = FakeCelesTrak({ACTIVE: 403})
    with pytest.raises(SourceUnavailable, match="no usable copy"):
        fetch_group("active", cache=tmp_path, download=server, now=clock)
    server.answers[ACTIVE] = ACTIVE_CSV
    assert (
        fetch_group("active", cache=tmp_path, download=server, now=clock).omms
        == SAMPLE[:4]
    )
    assert server.urls == [ACTIVE, ACTIVE]


def test_the_catalog_keeps_the_newest_set_of_each_object(
    tmp_path: Path, clock: Clock
) -> None:
    calsphere, iss, goes, themis = SAMPLE[:4]
    newer_iss = later(iss, 0.25)
    debris = dumps_csv([SAMPLE[4]]).encode()
    server = FakeCelesTrak(
        {
            ACTIVE: ACTIVE_CSV,
            group_url("last-30-days"): dumps_csv(
                [newer_iss, goes, later(themis, -1.0)]
            ).encode(),
            group_url("analyst"): dumps_csv(SAMPLE[5:7]).encode(),
            **{group_url(name): debris for name in PUBLIC_GROUPS[3:]},
        }
    )
    catalog = fetch_catalog(cache=tmp_path, download=server, now=clock)
    assert server.urls == [group_url(name) for name in PUBLIC_GROUPS]
    assert [part.name for part in catalog.parts] == list(PUBLIC_GROUPS)
    assert catalog.omms == (calsphere, newer_iss, themis, SAMPLE[4], goes, *SAMPLE[5:7])
    assert [omm.catalog_number for omm in catalog.omms] == sorted(
        omm.catalog_number for omm in catalog.omms
    )
    assert catalog.duplicates == (iss, later(themis, -1.0))


def test_a_catalog_fetch_stops_at_the_first_failure(
    tmp_path: Path, clock: Clock
) -> None:
    server = FakeCelesTrak({ACTIVE: ACTIVE_CSV, STATIONS: 500})
    with pytest.raises(HttpStatusError) as raised:
        fetch_catalog(
            ["active", "stations", "analyst"],
            cache=tmp_path,
            download=server,
            now=clock,
        )
    assert raised.value.status == 500
    assert server.urls == [ACTIVE, STATIONS]
    # The group fetched before the failure stays cached.
    assert (
        fetch_group("active", cache=tmp_path, download=server, now=clock).omms
        == SAMPLE[:4]
    )
    assert server.urls == [ACTIVE, STATIONS]


def test_a_group_named_twice_is_fetched_once(tmp_path: Path, clock: Clock) -> None:
    server = FakeCelesTrak({ACTIVE: ACTIVE_CSV})
    catalog = fetch_catalog(
        ["active", "ACTIVE"], cache=tmp_path, download=server, now=clock
    )
    assert server.urls == [ACTIVE]
    assert catalog.omms == tuple(sorted(SAMPLE[:4], key=lambda o: o.catalog_number))
    assert catalog.duplicates == ()


def test_the_catalog_needs_a_collection_of_groups(tmp_path: Path) -> None:
    server = FakeCelesTrak({})
    with pytest.raises(TypeError, match="not a string"):
        fetch_catalog("active", cache=tmp_path, download=server)
    with pytest.raises(ValueError, match="no groups"):
        fetch_catalog([], cache=tmp_path, download=server)
    assert server.urls == []


def test_socrates_is_downloaded_cached_and_given_as_text(
    tmp_path: Path, clock: Clock
) -> None:
    server = FakeCelesTrak({SOCRATES_URL: SOCRATES_CSV.encode()})
    report = fetch_socrates(cache=tmp_path, download=server, now=clock)
    assert report.text == SOCRATES_CSV
    path = tmp_path / "celestrak" / "socrates" / "sort-minRange.csv"
    assert report.retrieval.path == path
    assert path.read_text() == SOCRATES_CSV
    clock.now = START + TTL
    server.answers[SOCRATES_URL] = 403
    again = fetch_socrates(cache=tmp_path, download=server, now=clock)
    assert again.text == SOCRATES_CSV
    assert again.retrieval.checked.unix == START + TTL
    assert server.urls == [SOCRATES_URL, SOCRATES_URL]


def test_socrates_reads_a_header_with_a_byte_order_mark(
    tmp_path: Path, clock: Clock
) -> None:
    text = "\ufeff" + SOCRATES_CSV.replace("\n", "\r\n")
    server = FakeCelesTrak({SOCRATES_URL: text.encode()})
    assert fetch_socrates(cache=tmp_path, download=server, now=clock).text == text


@pytest.mark.parametrize(
    "body", [b"<html>Service unavailable</html>", b"\xff\xfe", b"NORAD_CAT_ID_10\n"]
)
def test_a_body_that_is_not_a_socrates_report_is_refused(
    body: bytes, tmp_path: Path, clock: Clock
) -> None:
    server = FakeCelesTrak({SOCRATES_URL: body})
    with pytest.raises(SourceFormatError, match="did not send"):
        fetch_socrates(cache=tmp_path, download=server, now=clock)
    assert not (tmp_path / "celestrak").exists()


def test_the_default_cache_is_keepouts(
    tmp_path: Path, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KEEPOUT_CACHE_DIR", str(tmp_path))
    group = fetch_group(
        "active", download=FakeCelesTrak({ACTIVE: ACTIVE_CSV}), now=clock
    )
    assert group.retrieval.path == tmp_path / "celestrak" / "gp" / "active.csv"


def test_requests_name_keepout_in_the_user_agent(
    tmp_path: Path, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[urllib.request.Request] = []

    class Response(io.BytesIO):
        status = 200
        reason = "OK"

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> io.BytesIO:
        requests.append(request)
        return Response(ACTIVE_CSV)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert fetch_group("active", cache=tmp_path, now=clock).omms == SAMPLE[:4]
    assert [request.full_url for request in requests] == [ACTIVE]
    assert requests[0].get_header("User-agent") == cache.USER_AGENT
