import hashlib
import math
import shutil
from pathlib import Path

import dashboard_seed as seed
import operator_example
import pytest
from astropy.time import Time
from builders import cached_pc
from generated import CARA_CDM_ID
from sqlalchemy import select

from keepout.cdm import kvn, load
from keepout.cdm.model import Section
from keepout.errors import SourceUnavailable
from keepout.settings import Settings, load_settings, settings_to_data
from keepout.sources import cara
from keepout.store import models, open_store

SEED_EVENTS = 12
"""The approaches of the example fleet in the snapshot's week (see
``docs/synthetic-operator.md``)."""

SEED_CDMS = 37
"""Their CDMs: one from each issue before TCA."""

UPCOMING = 7
"""The events whose TCA is after the last issue."""


@pytest.mark.slow
def test_seed_files_are_current() -> None:
    assert seed.main(["--check"]) == 0


def test_the_seed_files_are_named_by_message_id() -> None:
    files = seed.seed_files()
    assert len(files) == SEED_CDMS
    assert all(path.name == f"{load(path).message_id}.kvn" for path in files)


def test_the_seed_is_issued_every_twelve_hours() -> None:
    creations = {load(path).creation_date.utc.isot for path in seed.seed_files()}
    assert sorted(creations) == [
        "2026-10-06T00:00:00.000",
        "2026-10-06T12:00:00.000",
        "2026-10-07T00:00:00.000",
        "2026-10-07T12:00:00.000",
    ]


def test_seed_store_has_events(tmp_path: Path) -> None:
    path = seed.seed_store(tmp_path / "seed.db")
    settings = load_settings(seed.SETTINGS)
    with open_store(path, create=False) as store:
        upcoming = store.summaries(settings, as_of=seed.seed_as_of(), limit=200)
        page = store.summaries(
            settings, as_of=seed.seed_as_of(), include_past=True, limit=200
        )
        events = store.events()
    assert upcoming.total == UPCOMING
    assert page.total == SEED_EVENTS
    assert all(item.assessment is not None for item in page.items)
    assert sum(len(event.cdms) for event in events) == SEED_CDMS
    assert max(len(event.cdms) for event in events) == seed.ISSUES
    assert sum(len(event.cdms) >= 3 for event in events) >= 1


def test_seed_store_keeps_the_snapshot_catalog(tmp_path: Path) -> None:
    path = seed.seed_store(tmp_path / "seed.db")
    with open_store(path, create=False) as store:
        event_id = (
            store.summaries(load_settings(seed.SETTINGS), as_of=seed.seed_as_of())
            .items[0]
            .event_id
        )
        catalog = store.catalog(event_id)
    assert catalog is not None
    assert [file.name for file in catalog.files] == ["celestrak-public.csv.gz"]


def test_seed_as_of_is_latest_creation() -> None:
    creations = [load(path).creation_date for path in seed.seed_files()]
    latest = max(creations, key=lambda time: time.tai.mjd)
    assert seed.seed_as_of() == latest
    assert seed.seed_as_of() == Time("2026-10-07T12:00:00", scale="utc")


def test_settings_are_the_reports_defaults() -> None:
    settings = load_settings(seed.SETTINGS)
    defaults = Settings(thresholds=operator_example.THRESHOLDS)
    assert settings_to_data(settings) == settings_to_data(defaults)
    assert settings.combined_hard_body_radius is None  # the CDMs give HBR


def test_check_names_a_differing_and_a_stale_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cdms = [("a.kvn", "one\n"), ("b.kvn", "two\n")]
    assert seed.write_seed(cdms, tmp_path, check=False) == 0
    (tmp_path / "a.kvn").write_text("changed\n", encoding="utf-8")
    (tmp_path / "old.kvn").write_text("gone\n", encoding="utf-8")
    capsys.readouterr()

    assert seed.write_seed(cdms, tmp_path, check=True) == 1

    out = capsys.readouterr().out
    assert f"{tmp_path / 'a.kvn'} is out of date" in out
    assert f"{tmp_path / 'old.kvn'} is not generated" in out
    assert "b.kvn" not in out


def test_write_replaces_and_removes_stale_files(tmp_path: Path) -> None:
    (tmp_path / "old.kvn").write_text("gone\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("kept\n", encoding="utf-8")

    assert seed.write_seed([("a.kvn", "one\n")], tmp_path, check=False) == 0

    assert sorted(path.name for path in tmp_path.iterdir()) == ["a.kvn", "notes.txt"]
    assert seed.write_seed([("a.kvn", "one\n")], tmp_path, check=True) == 0


def test_without_cara_a_cara_file_is_neither_removed_nor_checked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "x.cdm").write_text("cara\n", encoding="utf-8")

    assert seed.write_seed([("a.kvn", "one\n")], tmp_path, check=False) == 0
    assert seed.write_seed([("a.kvn", "one\n")], tmp_path, check=True) == 0

    assert (tmp_path / "x.cdm").read_text(encoding="utf-8") == "cara\n"
    assert "x.cdm" not in capsys.readouterr().out


def test_with_cara_another_cara_file_is_stale(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cdms = [("a.kvn", "one\n"), ("new.cdm", "cara\n")]
    (tmp_path / "old.cdm").write_text("gone\n", encoding="utf-8")

    assert seed.write_seed(cdms, tmp_path, check=True) == 1
    assert f"{tmp_path / 'old.cdm'} is not generated" in capsys.readouterr().out
    assert seed.write_seed(cdms, tmp_path, check=False) == 0
    assert sorted(path.name for path in tmp_path.iterdir()) == ["a.kvn", "new.cdm"]


def seed_copy(directory: Path) -> Path:
    """A copy of the committed seed's ``.kvn`` files in ``directory``."""
    directory.mkdir()
    for file in seed.seed_files():
        if file.suffix == ".kvn":
            shutil.copy(file, directory / file.name)
    return directory


def test_seed_store_with_cara_needs_the_cara_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(seed, "SEED_DIR", seed_copy(tmp_path / "cdms"))
    with pytest.raises(FileNotFoundError, match=r"keepout fetch cara.*--cara"):
        seed.seed_store(tmp_path / "seed.db", cara=True)


def test_seed_store_ingests_the_cara_file_of_the_seed_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = seed_copy(tmp_path / "cdms")
    # A stand-in for CARA's CDM: a seed CDM under another MESSAGE_ID.
    first = min(directory.iterdir())
    text = first.read_text(encoding="utf-8").replace(first.stem, "STAND_IN")
    (directory / "STAND_IN.cdm").write_text(text, encoding="utf-8")
    monkeypatch.setattr(seed, "SEED_DIR", directory)

    without = seed.seed_store(tmp_path / "without.db")
    with_cara = seed.seed_store(tmp_path / "with.db", cara=True)

    assert origins(without) == {f.as_posix() for f in sorted(directory.glob("*.kvn"))}
    assert origins(with_cara) - origins(without) == {
        (directory / "STAND_IN.cdm").as_posix()
    }


def test_seed_store_without_cara_reads_no_cara_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = seed_copy(tmp_path / "cdms")
    (directory / "BROKEN.cdm").write_text("not a CDM\n", encoding="utf-8")
    monkeypatch.setattr(seed, "SEED_DIR", directory)
    path = seed.seed_store(tmp_path / "seed.db")
    assert origins(path) == {f.as_posix() for f in sorted(directory.glob("*.kvn"))}


def test_seed_files_default_to_the_seed_dir_of_the_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = seed_copy(tmp_path / "cdms")
    monkeypatch.setattr(seed, "SEED_DIR", directory)
    assert seed.seed_files() == seed.seed_files(directory)
    assert all(file.parent == directory for file in seed.seed_files())


def origins(path: Path) -> set[str]:
    """The origin of every CDM in the store at ``path``."""
    with open_store(path, create=False) as store, store.read() as session:
        return set(session.scalars(select(models.Cdm.origin)))


def test_seed_store_names_origins_from_the_repository(tmp_path: Path) -> None:
    names = {
        f"examples/dashboard/cdms/{file.name}"
        for file in seed.seed_files()
        if file.suffix == ".kvn"
    }
    assert origins(seed.seed_store(tmp_path / "seed.db")) == names


def test_seed_text_rounds_tca_to_the_millisecond() -> None:
    text = "TCA = 2026-10-08T12:03:59.999600\nMISS_DISTANCE = 3646.536263 [m]\n"
    assert seed.seed_text(text) == (
        "TCA = 2026-10-08T12:04:00.000\nMISS_DISTANCE = 3646.536 [m]\n"
    )


def test_cara_never_downloads(tmp_path: Path) -> None:
    with pytest.raises(SourceUnavailable, match="keepout fetch cara"):
        seed.cara_cdm(cache=tmp_path)


def cache_cara(cache: Path, text: str) -> Path:
    """``text`` as the cached copy of the CARA CDM."""
    target = cara.commit_dir(cache) / seed.CARA_FILE.path
    target.parent.mkdir(parents=True)
    target.write_text(text, encoding="utf-8")
    return target


def test_cara_rejects_a_changed_copy(tmp_path: Path) -> None:
    cache_cara(tmp_path, "not CARA's file\n")
    with pytest.raises(SourceUnavailable, match="keepout fetch cara"):
        seed.cara_cdm(cache=tmp_path)


def test_cara_reads_the_cached_cdm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = "CCSDS_CDM_VERS = 1.0\n"
    digest = hashlib.sha256(text.encode()).hexdigest()
    monkeypatch.setattr(seed, "CARA_FILE", cara.CaraFile(seed.CARA_FILE.path, digest))
    cache_cara(tmp_path, text)

    assert seed.cara_cdm(cache=tmp_path) == (f"{CARA_CDM_ID}.cdm", text)


def test_main_without_the_cara_cache_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("KEEPOUT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(seed, "synthetic_cdms", lambda: [("a.kvn", "one\n")])

    assert seed.main(["--cara", "--dir", str(tmp_path / "cdms")]) == 1

    assert "keepout fetch cara" in capsys.readouterr().err
    assert not (tmp_path / "cdms").exists()


def test_demo_event_is_upcoming_and_yellow(tmp_path: Path) -> None:
    path = seed.seed_store(tmp_path / "seed.db")
    settings = load_settings(seed.SETTINGS)
    with open_store(path, create=False) as store:
        page = store.summaries(settings, as_of=seed.seed_as_of(), limit=200)
    [summary] = [item for item in page.items if item.event_id == seed.DEMO_EVENT]
    assert summary.latest is not None
    assert summary.latest.pc is not None
    assert settings.thresholds.green_below <= summary.latest.pc
    assert summary.latest.pc < settings.thresholds.red_from


def test_demo_update_joins_event_and_is_red(tmp_path: Path) -> None:
    from keepout.store import CdmInput

    path = seed.seed_store(tmp_path / "seed.db")
    settings = load_settings(seed.SETTINGS)
    name, text = seed.demo_update()
    update = CdmInput.parse(text, name)
    with open_store(path, create=False) as store:
        report = store.ingest([update])
        store.refresh(settings)
        observations = cached_pc(store, seed.DEMO_EVENT, settings)
    assert report.events == (seed.DEMO_EVENT,)
    assert len(observations) == seed.ISSUES + 1
    latest = observations[-1]
    assert latest.message_id == update.cdm.message_id
    assert latest.pc is not None
    assert latest.pc >= settings.thresholds.red_from
    assert update.cdm.creation_date.utc.isot == seed.DEMO_AS_OF.utc.isot
    assert update.cdm.header.comments[0] == seed.DEMO_COMMENT
    assert name == f"{update.cdm.message_id}.kvn"


def sigma_km(section: Section, keyword: str) -> float:
    """The sigma [km] of the variance ``keyword`` of ``section``."""
    variance = section.get(keyword)
    assert isinstance(variance, float)
    return math.sqrt(variance) / 1e3


def test_demo_update_states_its_narrowed_sigmas() -> None:
    _, text = seed.demo_update()
    for section in kvn.loads(text).object_sections:
        radial, transverse, normal = (
            sigma_km(section, keyword) for keyword in ("CR_R", "CT_T", "CN_N")
        )
        stated = f"Sigmas: R {radial:.3f} km, T {transverse:.3f} km, N {normal:.3f} km"
        assert [c for c in section.comments if c.startswith("Sigmas:")] == [stated]


def test_demo_update_says_its_covariance_is_scaled() -> None:
    _, text = seed.demo_update()
    for section in kvn.loads(text).object_sections:
        assert seed.NARROWED_COMMENT in section.comments


def record_demo_writes(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """The ``check`` of each demo write of a run whose seed has no CDM."""
    demos: list[bool] = []

    def write_demo(*, check: bool) -> int:
        demos.append(check)
        return 0

    monkeypatch.setattr(seed, "seed_cdms", lambda *, cara: [])
    monkeypatch.setattr(seed, "write_demo", write_demo)
    return demos


def test_the_default_dir_given_explicitly_writes_the_demo_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demos = record_demo_writes(monkeypatch)
    monkeypatch.setattr(seed, "write_seed", lambda cdms, directory, *, check: 0)
    default = seed.SEED_DIR.parent / ".." / "dashboard" / seed.SEED_DIR.name

    assert seed.main(["--dir", str(default), "--check"]) == 0

    assert demos == [True]


def test_another_dir_writes_the_seed_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demos = record_demo_writes(monkeypatch)

    assert seed.main(["--dir", str(tmp_path)]) == 0

    assert demos == []


def test_demo_update_file_is_current() -> None:
    path = seed.demo_file()
    name, text = seed.demo_update()
    assert path.name == name
    assert path.read_text(encoding="utf-8") == text


@pytest.fixture(scope="module")
def demo_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The demo's store, built once (a manoeuvre plan takes about 10 s)."""
    return seed.demo_store(tmp_path_factory.mktemp("demo") / "demo.db")


@pytest.mark.slow
def test_demo_store_has_plan(demo_store: Path) -> None:
    with open_store(demo_store, create=False) as store:
        outcome = store.maneuver(seed.DEMO_EVENT, load_settings(seed.SETTINGS))
    assert outcome is not None
    assert outcome.catalog_label is not None
    assert outcome.catalog_label.startswith("stored copy of ")


@pytest.mark.slow
def test_demo_store_has_no_wal(demo_store: Path) -> None:
    assert sorted(p.name for p in demo_store.parent.iterdir()) == [
        "demo.db",
        "demo.db.catalogs",
    ]


@pytest.mark.parametrize("other", [["--check"], ["--cara"], ["--dir", "x"]])
def test_store_option_exclusive(
    other: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        seed.main(["--store", "x.db", *other])
    assert exit_info.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err
