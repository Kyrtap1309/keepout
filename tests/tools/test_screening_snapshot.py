import dataclasses
import gzip
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import astropy.units as u
import pytest
import screening_snapshot as snapshot
import snapshot_files
from celestrak_builders import SAMPLE, SOCRATES_HEADER, FakeCelesTrak, socrates_row
from hypothesis import given
from hypothesis import strategies as st

from keepout.errors import SourceFormatError, SourceUnavailable
from keepout.omm import Omm, load
from keepout.omm.celestrak import dumps_csv
from keepout.sources.celestrak import PUBLIC_GROUPS, SOCRATES_URL, group_url
from keepout.sources.socrates import read_socrates

CALSPHERE, ISS, GOES, THEMIS, EXPRESS, SARAMAGO, STARLINK, DRAGON = SAMPLE
OLD_ISS = dataclasses.replace(ISS, epoch=ISS.epoch - 1.0 * u.day)
"""An older element set of the ISS: a duplicate the catalog leaves out."""

DOCKED = socrates_row(ISS, DRAGON, ("1.000", "1.000"), "0.004", "0.000")
FAST = socrates_row(CALSPHERE, STARLINK, ("0.838", "0.764"), "2.500", "7.500")
SOCRATES_CSV = (
    SOCRATES_HEADER
    + DOCKED
    + socrates_row(CALSPHERE, GOES, ("2.000", "2.000"), "0.100", "3.000")
    + socrates_row(
        ISS,
        dataclasses.replace(ISS, catalog_number=99999),
        ("1.000", "1.000"),
        "0.200",
        "7.500",
    )
    + FAST
)
"""A report: the ISS and Crew Dragon (docked, with the sample's epochs), a
pair whose epochs are not the sample's, a pair with an object not in the
sample, and Calsphere 1 and Starlink-38128 (the sample's epochs: DSE to 43 s)."""

BODIES: Mapping[str, bytes] = {
    group_url("active"): dumps_csv(SAMPLE[:5]).encode(),
    group_url("last-30-days"): dumps_csv([*SAMPLE[5:], OLD_ISS]).encode(),
    group_url("analyst"): dumps_csv(SAMPLE[5:]).encode(),
    group_url("fengyun-1c-debris"): dumps_csv([]).encode(),
    group_url("cosmos-2251-debris"): dumps_csv([]).encode(),
    group_url("iridium-33-debris"): dumps_csv([]).encode(),
    SOCRATES_URL: SOCRATES_CSV.encode(),
}
URLS = [*(group_url(name) for name in PUBLIC_GROUPS), SOCRATES_URL]


@pytest.fixture
def celestrak(monkeypatch: pytest.MonkeyPatch) -> FakeCelesTrak:
    fake = FakeCelesTrak(BODIES)
    monkeypatch.setattr(snapshot, "http_download", fake)
    return fake


@pytest.fixture
def raw(tmp_path: Path, celestrak: FakeCelesTrak) -> Path:
    directory = tmp_path / "raw"
    assert snapshot.main(["download", str(directory)]) == 0
    return directory


def build(raw: Path, data: Path, *options: str) -> int:
    return snapshot.main(["build", str(raw), "--data", str(data), *options])


def test_download_asks_for_each_url_once_and_keeps_the_bodies(
    raw: Path, celestrak: FakeCelesTrak
) -> None:
    assert celestrak.urls == URLS
    index = snapshot.INDEX.read(raw)
    assert [body.url for body in index] == URLS
    for body in index:
        data = (raw / body.file).read_bytes()
        assert data == BODIES[body.url]
        assert body.sha256 == hashlib.sha256(data).hexdigest()
        assert body.size == len(data)
        assert body.retrieved.endswith("Z")
    assert index[0].file == "gp-active.csv"
    assert index[-1].file == "socrates-sort-minRange.csv"


def test_download_does_not_ask_again_for_a_saved_url(
    raw: Path, celestrak: FakeCelesTrak
) -> None:
    assert snapshot.main(["download", str(raw)]) == 0
    assert celestrak.urls == URLS


def test_download_stops_at_the_first_error_and_resumes_after_it(
    tmp_path: Path, celestrak: FakeCelesTrak
) -> None:
    celestrak.answers[group_url("analyst")] = 403
    raw = tmp_path / "raw"
    with pytest.raises(SourceUnavailable, match="403"):
        snapshot.main(["download", str(raw)])
    assert celestrak.urls == URLS[:3]
    saved = [body.url for body in snapshot.INDEX.read(raw)]
    assert saved == URLS[:2]
    celestrak.answers[group_url("analyst")] = BODIES[group_url("analyst")]
    celestrak.urls.clear()
    assert snapshot.main(["download", str(raw)]) == 0
    assert celestrak.urls == URLS[2:]


def test_download_keeps_a_body_keepout_cannot_read(
    tmp_path: Path, celestrak: FakeCelesTrak
) -> None:
    celestrak.answers[group_url("active")] = b"No GP data found"
    raw = tmp_path / "raw"
    with pytest.raises(SourceFormatError):
        snapshot.main(["download", str(raw)])
    assert (raw / "gp-active.csv").read_bytes() == b"No GP data found"
    assert celestrak.urls == URLS[:1]


def test_offline_bodies_refuse_an_unsaved_url(tmp_path: Path) -> None:
    with pytest.raises(SourceUnavailable, match="not saved"):
        snapshot.RawBodies(tmp_path, None)(SOCRATES_URL)


def test_a_changed_body_is_refused(raw: Path) -> None:
    (raw / "gp-analyst.csv").write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest"):
        snapshot.fetch(raw, None)


def test_an_unknown_raw_index_is_refused(tmp_path: Path) -> None:
    (tmp_path / snapshot_files.RAW_INDEX).write_text('{"format": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="format 1"):
        snapshot.INDEX.read(tmp_path)


@pytest.mark.parametrize(
    ("url", "name"),
    [
        (group_url("active"), "gp-active.csv"),
        (group_url("last-30-days"), "gp-last-30-days.csv"),
        (SOCRATES_URL, "socrates-sort-minRange.csv"),
    ],
)
def test_raw_name(url: str, name: str) -> None:
    assert snapshot.raw_name(url) == name


@pytest.mark.parametrize(
    "url",
    [
        "https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=CSV",
        "https://example.org/gp.php?GROUP=active",
    ],
)
def test_raw_name_refuses_other_urls(url: str) -> None:
    with pytest.raises(ValueError, match="not a CelesTrak URL"):
        snapshot.raw_name(url)


def test_build_writes_the_data_then_checks_it(raw: Path, tmp_path: Path) -> None:
    data = tmp_path / "data"
    assert build(raw, data, "--check") == 1
    assert build(raw, data) == 0
    assert build(raw, data, "--check") == 0
    assert snapshot.main(["verify", "--data", str(data)]) == 0
    assert load(data / snapshot.SNAPSHOT) == tuple(
        sorted(SAMPLE, key=lambda omm: omm.catalog_number)
    )
    excerpt = gzip.decompress((data / snapshot.SOCRATES).read_bytes()).decode()
    assert excerpt == SOCRATES_HEADER + DOCKED + FAST
    manifest = json.loads((data / snapshot_files.MANIFEST).read_text(encoding="utf-8"))
    assert manifest["retrieved"] == {
        "first": manifest["downloads"][0]["retrieved"],
        "last": manifest["downloads"][-1]["retrieved"],
    }
    assert [body["url"] for body in manifest["downloads"]] == URLS
    assert manifest["catalog"] == {
        "group_rows": {
            "active": 5,
            "last-30-days": 4,
            "analyst": 3,
            "fengyun-1c-debris": 0,
            "cosmos-2251-debris": 0,
            "iridium-33-debris": 0,
        },
        "duplicates": 1,
        "objects": 8,
        "six_digit": 3,
        "regimes": {"LEO": 5, "MEO": 1, "GEO": 1, "HEO": 1},
        "oldest_epoch": THEMIS.epoch.isot,
        "newest_epoch": GOES.epoch.isot,
    }
    assert manifest["socrates"] == {
        "approaches": 4,
        "both_in_catalog": 3,
        "comparable": 2,
        "first_tca": "2026-10-07T00:20:47.861",
        "last_tca": "2026-10-07T00:20:47.861",
        "newest_epoch": "2026-10-06T06:00",
    }
    assert manifest["fixture"] == {
        "objects": 8,
        "regimes": {"LEO": 5, "MEO": 1, "GEO": 1, "HEO": 1},
    }
    assert set(manifest["files"]) == {
        snapshot.SNAPSHOT,
        snapshot.SOCRATES,
        snapshot.FIXTURE,
        snapshot.NOTICE,
    }
    for name, entry in manifest["files"].items():
        stored = (data / name).read_bytes()
        assert entry == {
            "bytes": len(stored),
            "sha256": hashlib.sha256(stored).hexdigest(),
        }


def test_build_is_deterministic(raw: Path, tmp_path: Path) -> None:
    assert build(raw, tmp_path / "one") == 0
    assert build(raw, tmp_path / "two") == 0
    for name in [*snapshot.GZIPPED, snapshot.FIXTURE, snapshot.NOTICE]:
        one = (tmp_path / "one" / name).read_bytes()
        assert one == (tmp_path / "two" / name).read_bytes()
    stored = (tmp_path / "one" / snapshot.SNAPSHOT).read_bytes()
    assert stored == gzip.compress(gzip.decompress(stored), mtime=0)


def test_build_keeps_a_gzipped_file_with_the_same_contents(
    raw: Path, tmp_path: Path
) -> None:
    data = tmp_path / "data"
    assert build(raw, data) == 0
    path = data / snapshot.SNAPSHOT
    other = gzip.compress(gzip.decompress(path.read_bytes()), compresslevel=1)
    path.write_bytes(other)
    assert build(raw, data) == 0
    assert path.read_bytes() == other
    assert snapshot.main(["verify", "--data", str(data)]) == 0


@pytest.mark.parametrize("name", [snapshot.FIXTURE, snapshot.NOTICE])
def test_verify_finds_a_changed_file(raw: Path, tmp_path: Path, name: str) -> None:
    data = tmp_path / "data"
    assert build(raw, data) == 0
    (data / name).write_bytes((data / name).read_bytes() + b"\n")
    assert snapshot.main(["verify", "--data", str(data)]) == 1
    assert build(raw, data, "--check") == 1


def test_verify_finds_an_approach_that_is_not_comparable(
    raw: Path, tmp_path: Path
) -> None:
    data = tmp_path / "data"
    assert build(raw, data) == 0
    changed = SOCRATES_HEADER + DOCKED.replace("1.000", "1.100") + FAST
    (data / snapshot.SOCRATES).write_bytes(gzip.compress(changed.encode()))
    assert snapshot.main(["verify", "--data", str(data)]) == 1


def test_build_replaces_a_broken_gzip_file(raw: Path, tmp_path: Path) -> None:
    data = tmp_path / "data"
    assert build(raw, data) == 0
    path = data / snapshot.SOCRATES
    good = path.read_bytes()
    path.write_bytes(good[:-1] + bytes([good[-1] ^ 1]))
    assert build(raw, data, "--check") == 1
    assert build(raw, data) == 0
    assert path.read_bytes() == good


def test_notice_cites_the_source_and_date(raw: Path, tmp_path: Path) -> None:
    data = tmp_path / "data"
    assert build(raw, data) == 0
    notice = (data / snapshot.NOTICE).read_text(encoding="utf-8")
    first = snapshot.INDEX.read(raw)[0].retrieved
    assert notice.startswith(
        "Data: USSPACECOM/18 SDS via Space-Track.org, retrieved from CelesTrak\n"
        f"(celestrak.org) on {first[:10]}.\n\n"
        f"Retrieved {first[:10]} {first[11:19]} to "
    )
    assert "not covered by keepout's MIT licence" in notice


def test_notice_gives_both_dates_of_a_download_over_midnight() -> None:
    def body(retrieved: str) -> snapshot.Body:
        return snapshot.Body("url", "file", "0" * 64, 0, retrieved)

    inputs = snapshot.Inputs(
        downloads=(body("2026-10-06T23:59:58.000Z"), body("2026-10-07T00:00:03.000Z")),
        group_rows={"active": 0},
        duplicates=0,
        omms=(),
        socrates=SOCRATES_HEADER,
        summary=snapshot.SocratesSummary(
            approaches=0,
            both_in_catalog=0,
            comparable=0,
            first_tca="",
            last_tca="",
            newest_epoch="",
        ),
    )
    assert "Retrieved 2026-10-06 23:59:58 to 2026-10-07 00:00:03 UTC" in (
        snapshot.render_notice(inputs)
    )


def test_socrates_header_keeps_a_byte_order_mark() -> None:
    text = "\ufeff" + SOCRATES_CSV
    assert snapshot.socrates_header(text) == "\ufeff" + SOCRATES_HEADER


def test_comparable_approaches() -> None:
    approaches = read_socrates(SOCRATES_CSV)
    assert snapshot.comparable(approaches, SAMPLE) == [approaches[0], approaches[3]]


@pytest.mark.parametrize(
    ("omm", "expected"),
    [(CALSPHERE, "LEO"), (EXPRESS, "MEO"), (GOES, "GEO"), (THEMIS, "HEO")],
)
def test_regime(omm: Omm, expected: str) -> None:
    assert snapshot.regime(omm) == expected


def test_perigee_altitude_of_the_iss() -> None:
    assert snapshot.perigee_altitude(ISS) == pytest.approx(412e3, abs=10e3)


@pytest.mark.parametrize(
    ("count", "expected"),
    [(0, []), (3, [0, 3, 6]), (10, list(range(10))), (12, list(range(10)))],
)
def test_spread(count: int, expected: list[int]) -> None:
    assert snapshot.spread(list(range(10)), count) == expected


@given(st.integers(0, 50), st.integers(0, 60))
def test_spread_takes_distinct_items_in_order(size: int, count: int) -> None:
    taken = snapshot.spread(list(range(size)), count)
    assert len(taken) == min(size, count)
    assert taken == sorted(set(taken))
    if count and size:
        assert taken[0] == 0


@given(st.permutations(SAMPLE), st.permutations(range(4)))
def test_select_fixture_ignores_the_order(omms: list[Omm], order: list[int]) -> None:
    approaches = read_socrates(SOCRATES_CSV)
    chosen = snapshot.select_fixture(omms, [approaches[i] for i in order])
    assert chosen == sorted(SAMPLE, key=lambda omm: omm.catalog_number)


def no_fixture_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("CLOSEST", "CLOSEST_FAST", "CLOSEST_PER_REGIME", "EXTREMES"):
        monkeypatch.setattr(snapshot, name, 0)
    monkeypatch.setattr(snapshot, "PER_REGIME", dict.fromkeys(snapshot.REGIMES, 0))


@pytest.mark.parametrize(
    ("part", "value", "expected"),
    [
        ("CLOSEST", 1, [25544, 100882]),
        ("CLOSEST", 2, [900, 25544, 100001, 100882]),
        ("CLOSEST_FAST", 1, [900, 100001]),
        ("CLOSEST_PER_REGIME", 1, [25544, 100882]),
        ("PER_REGIME", {"LEO": 2, "MEO": 0, "GEO": 1, "HEO": 0}, [900, 51850, 100000]),
        ("PER_REGIME", {"LEO": 0, "MEO": 1, "GEO": 0, "HEO": 1}, [30798, 38745]),
        # THEMIS E: highest eccentricity and oldest epoch; EXPRESS-MD2:
        # lowest perigee; 100000: the first six-digit number.
        ("EXTREMES", 1, [30798, 38745, 100000]),
    ],
)
def test_select_fixture_parts(
    monkeypatch: pytest.MonkeyPatch, part: str, value: object, expected: list[int]
) -> None:
    no_fixture_parts(monkeypatch)
    monkeypatch.setattr(snapshot, part, value)
    chosen = snapshot.select_fixture(SAMPLE, read_socrates(SOCRATES_CSV))
    assert [omm.catalog_number for omm in chosen] == expected
