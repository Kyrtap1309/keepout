import dataclasses
import gzip
import hashlib
import itertools
import re
import shutil
import sqlite3
import tempfile
from collections.abc import Iterator
from pathlib import Path

import astropy.units as u
import pytest
from astropy.time import Time
from builders import B1_KVN, CDM_DATA
from cdm_strategies import cdms
from cli_builders import operator_kvn, without_hbr
from hypothesis import given, settings
from sqlalchemy import event, func, select, text
from store_builders import TCA, cdm, cdm_text, message, written

import keepout.cdm
from keepout.errors import ChecksumMismatch, HardBodyRadiusError, StoreError
from keepout.events import DEFAULT_TCA_TOLERANCE, ConjunctionEvent, group_events
from keepout.pc import computation
from keepout.sources.catalog import CatalogFileRef, CatalogReference, load_reference
from keepout.store import (
    CdmInput,
    IngestReport,
    RefreshReport,
    Store,
    models,
    open_store,
)
from keepout.store.codec import observation_to_json
from keepout.store.config import Settings

TOLERANCE = DEFAULT_TCA_TOLERANCE
SAMPLE_CSV = Path(__file__).parents[1] / "omm" / "data" / "celestrak_sample.csv"
FOSTER = Settings(compute=computation("foster"), combined_hard_body_radius=20.0)


@pytest.fixture
def tmp_store(tmp_path: Path) -> Iterator[Store]:
    with open_store(tmp_path / "keepout.db") as store:
        yield store


def _partition(store: Store) -> set[frozenset[str]]:
    return {
        frozenset(c.message_id for c in event.cdms)
        for event in store.events(include_archived=True)
    }


def _count(store: Store, table: type[models.Base]) -> int:
    with store.read() as session:
        return session.scalar(select(func.count()).select_from(table)) or 0


def test_a_cdm_input_reads_kvn_xml_and_gzip(tmp_path: Path) -> None:
    kvn = CdmInput.read(B1_KVN)
    assert (kvn.format, kvn.origin) == ("kvn", str(B1_KVN))
    assert kvn.text == B1_KVN.read_text(encoding="utf-8")
    assert kvn.cdm.message_id == keepout.cdm.load(B1_KVN).message_id
    xml = CdmInput.read(CDM_DATA / "b1_example.xml")
    assert xml.format == "xml"
    packed = tmp_path / "b1.kvn.gz"
    packed.write_bytes(gzip.compress(B1_KVN.read_bytes()))
    assert CdmInput.read(packed).text == kvn.text


def test_a_cdm_input_of_a_cdm_is_its_kvn() -> None:
    made = cdm("A")
    given = CdmInput.of(made, origin="screen")
    assert given == CdmInput(made, keepout.cdm.kvn.dumps(made), "kvn", "screen")


def test_ingest_stores_text_and_groups(tmp_store: Store, tmp_path: Path) -> None:
    first = CdmInput.read(written(tmp_path / "a.kvn", cdm_text("A")))
    later = message("B", tca=0.5, created=-3_600.0)
    other_pair = message("C", secondary="40000")
    report = tmp_store.ingest([first, later, other_pair])
    assert report.added == ("A", "B", "C")
    assert report.already_stored == ()
    assert len(report.events) == 2
    assert report.new_events == report.events
    assert all(len(i) == 8 and int(i, 16) >= 0 for i in report.events)
    events = tmp_store.events()
    assert [e.event_id for e in events] == sorted(report.events)
    assert _partition(tmp_store) == {frozenset("AB"), frozenset("C")}
    with tmp_store.read() as session:
        row = session.get(models.Cdm, "A")
        assert row is not None
        assert row.content == first.text
        assert (row.format, row.origin) == ("kvn", str(tmp_path / "a.kvn"))
        assert row.tca == "2010-03-13T22:37:52.618000"
        assert row.creation_date == "2010-03-12T22:37:52.618000"
        stored = session.scalars(select(models.Event).order_by(models.Event.id)).all()
        pair = next(e for e in stored if e.object2 == "30337")
        assert (pair.object1, pair.tca_min, pair.tca_max) == (
            "12345",
            "2010-03-13T22:37:52.618000",
            "2010-03-13T22:37:53.118000",
        )
    event = tmp_store.event(pair.id)
    assert isinstance(event, ConjunctionEvent)
    assert event.event_id == pair.id
    assert [c.message_id for c in event.cdms] == ["A", "B"]
    assert event.tca_tolerance == TOLERANCE


def test_the_report_lists_messages_in_the_order_given(tmp_store: Store) -> None:
    report = tmp_store.ingest([message("B", tca=60.0), message("A")])
    assert report.added == ("B", "A")
    again = tmp_store.ingest([message("C", tca=30.0), message("B", tca=60.0)])
    assert (again.added, again.already_stored) == (("C",), ("B",))
    assert again.events == report.events
    assert again.new_events == ()


def test_reingest_is_reported_and_changes_nothing(tmp_store: Store) -> None:
    batch = [message("A"), message("B", tca=1.0)]
    first = tmp_store.ingest(batch)
    with tmp_store.read() as session:
        before = session.execute(text("SELECT * FROM cdms ORDER BY message_id")).all()
    again = tmp_store.ingest(batch)
    assert again == IngestReport(
        added=(), already_stored=("A", "B"), events=first.events, new_events=()
    )
    with tmp_store.read() as session:
        after = session.execute(text("SELECT * FROM cdms ORDER BY message_id")).all()
    assert after == before
    assert _count(tmp_store, models.Event) == 1


def test_a_message_repeated_in_one_batch_is_stored_once(tmp_store: Store) -> None:
    report = tmp_store.ingest([message("A"), message("A")])
    assert (report.added, report.already_stored) == (("A",), ("A",))
    assert _count(tmp_store, models.Cdm) == 1


def test_kvn_and_xml_of_one_message_are_one_message(tmp_store: Store) -> None:
    kvn = CdmInput.read(B1_KVN)
    xml = CdmInput.read(CDM_DATA / "b1_example.xml")
    assert tmp_store.ingest([kvn]).added == (kvn.cdm.message_id,)
    report = tmp_store.ingest([xml])
    assert (report.added, report.already_stored) == ((), (xml.cdm.message_id,))
    with tmp_store.read() as session:
        row = session.get(models.Cdm, kvn.cdm.message_id)
        assert row is not None
        assert row.format == "kvn"


def test_same_message_id_other_content_raises_and_writes_nothing(
    tmp_store: Store, tmp_path: Path
) -> None:
    tmp_store.ingest([message("A")])
    # B, of another pair, makes a new event before A fails the batch.
    batch = [message("B", secondary="40000", tca=1.0), message("A", tca=2.0)]
    with pytest.raises(StoreError, match="MESSAGE_ID A is stored with other content"):
        tmp_store.ingest(batch, catalog=_reference(tmp_path, "active.csv"))
    assert _count(tmp_store, models.Cdm) == 1
    assert _count(tmp_store, models.Event) == 1
    assert _count(tmp_store, models.Catalog) == 0
    with pytest.raises(StoreError, match="MESSAGE_ID C is stored with other content"):
        tmp_store.ingest([message("C"), message("C", tca=2.0)])
    assert _count(tmp_store, models.Cdm) == 1


def test_a_cdm_beyond_the_tolerance_starts_a_new_event(tmp_store: Store) -> None:
    first = tmp_store.ingest([message("A")])
    second = tmp_store.ingest([message("B", tca=TOLERANCE + 1.0)])
    assert second.new_events != first.new_events
    assert len(second.new_events) == 1
    # Within the tolerance of the first CDM but stretching the event over it.
    third = tmp_store.ingest([message("C", tca=-TOLERANCE + 0.5)])
    assert third.events == first.events


def test_a_cdm_fitting_two_events_raises(tmp_store: Store) -> None:
    early = tmp_store.ingest([message("A")]).events[0]
    late = tmp_store.ingest([message("B", tca=1.5 * TOLERANCE)]).events[0]
    with pytest.raises(StoreError, match="CDM M fits more than one event") as caught:
        tmp_store.ingest([message("M", tca=0.75 * TOLERANCE)])
    assert early in str(caught.value)
    assert late in str(caught.value)
    assert _count(tmp_store, models.Cdm) == 2


def test_a_cdm_between_two_events_ingested_apart_raises(tmp_store: Store) -> None:
    # 16 min and 0 min are two events; 8 min fits both. In one batch (TCA
    # order) 0 and 8 would make one event and 16 another.
    late = tmp_store.ingest([message("L", tca=16 * 60.0)]).events
    early = tmp_store.ingest([message("E")]).events
    assert late != early
    with pytest.raises(StoreError, match="CDM M fits more than one event"):
        tmp_store.ingest([message("M", tca=8 * 60.0)])


def _with_tca(given: CdmInput, tca: str) -> CdmInput:
    line = re.compile(r"^TCA\s*=.*$", re.MULTILINE)
    text = line.sub(f"TCA = {tca}", given.text, count=1)
    return CdmInput.of(keepout.cdm.loads(text), origin="test")


def test_a_tca_a_fraction_of_a_microsecond_too_far_starts_a_new_event(
    tmp_store: Store,
) -> None:
    # Spread 600.0000004 s: beyond the tolerance, though both TCAs round to
    # the same microseconds as a spread of exactly 600 s.
    first = _with_tca(message("A"), "2010-03-13T22:30:00.000")
    second = _with_tca(message("B"), "2010-03-13T22:40:00.0000004")
    assert len(group_events([first.cdm, second.cdm])) == 2
    report = tmp_store.ingest([first, second])
    assert len(report.new_events) == 2
    assert _partition(tmp_store) == {frozenset("A"), frozenset("B")}
    assert [len(tmp_store.event(i).cdms) for i in report.events] == [1, 1]


def test_a_tca_exactly_at_the_tolerance_joins(tmp_store: Store) -> None:
    first = _with_tca(message("A"), "2010-03-13T22:30:00.000")
    second = _with_tca(message("B"), "2010-03-13T22:40:00.000")
    assert len(tmp_store.ingest([first, second]).events) == 1


def test_a_conjunction_id_keeps_events_apart(tmp_store: Store) -> None:
    first = tmp_store.ingest([message("A", conjunction_id="X")]).events
    other = tmp_store.ingest([message("B", tca=1.0, conjunction_id="Y")]).events
    assert other != first
    same = tmp_store.ingest([message("C", tca=2.0, conjunction_id="X")]).events
    assert same == first
    with pytest.raises(StoreError, match="fits more than one event"):
        tmp_store.ingest([message("D", tca=3.0)])


def test_an_event_takes_the_conjunction_id_its_cdms_give(tmp_store: Store) -> None:
    (event_id,) = tmp_store.ingest([message("A")]).events
    assert tmp_store.ingest([message("B", conjunction_id="X")]).events == (event_id,)
    assert tmp_store.ingest([message("C", conjunction_id="Y")]).events != (event_id,)
    with tmp_store.read() as session:
        stored = session.get(models.Event, event_id)
        assert stored is not None
        assert stored.conjunction_id == "X"


def test_event_ids_never_change_after_more_cdms(tmp_store: Store) -> None:
    (event_id,) = tmp_store.ingest([message("A", tca=5.0)]).events
    for number, offset in enumerate((-5.0, 30.0, 1.0)):
        report = tmp_store.ingest([message(f"N{number}", tca=offset)])
        assert report.events == (event_id,)
        assert report.new_events == ()
    assert [e.event_id for e in tmp_store.events()] == [event_id]
    assert len(tmp_store.event(event_id).cdms) == 4


def test_an_event_id_already_taken_is_drawn_again(
    tmp_store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    draws = iter(["0000abcd", "0000abcd", "0000beef"])
    monkeypatch.setattr("secrets.token_hex", lambda _: next(draws))
    assert tmp_store.ingest([message("A")]).events == ("0000abcd",)
    assert tmp_store.ingest([message("B", secondary="40000")]).events == ("0000beef",)


def test_an_unknown_event_is_an_error(tmp_store: Store) -> None:
    for call in (tmp_store.event, tmp_store.catalog):
        with pytest.raises(StoreError, match="no event 12345678 in the store"):
            call("12345678")
    with pytest.raises(StoreError, match="no event 12345678 in the store"):
        tmp_store.observations("12345678", FOSTER)
    with pytest.raises(StoreError, match="no event 12345678 in the store"):
        tmp_store.refresh(FOSTER, event_ids=["12345678"])


def test_events_are_listed_by_first_tca_without_archived_ones(
    tmp_store: Store,
) -> None:
    late = tmp_store.ingest([message("A", tca=3 * TOLERANCE)]).events[0]
    early = tmp_store.ingest([message("B")]).events[0]
    gone = tmp_store.ingest([message("C", secondary="40000")]).events[0]
    with tmp_store.write() as session:
        archived = session.get(models.Event, gone)
        assert archived is not None
        archived.archived = True
    assert [e.event_id for e in tmp_store.events()] == [early, late]
    listed = [e.event_id for e in tmp_store.events(include_archived=True)]
    assert listed[-1] == late
    assert set(listed) == {early, late, gone}
    assert tmp_store.event(gone).event_id == gone


def test_events_of_given_ids_parse_only_theirs(
    tmp_store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    late = tmp_store.ingest([message("A", tca=3 * TOLERANCE)]).events[0]
    early = tmp_store.ingest([message("B")]).events[0]
    other = tmp_store.ingest([message("C", secondary="40000")]).events[0]
    with tmp_store.write() as session:
        archived = session.get(models.Event, late)
        assert archived is not None
        archived.archived = True
        broken = session.get(models.Cdm, "C")
        assert broken is not None
        broken.content = "not a CDM"
    parsed: list[str] = []
    loads = keepout.cdm.loads

    def counted(text: str) -> keepout.cdm.Cdm:
        made = loads(text)
        parsed.append(made.message_id)
        return made

    monkeypatch.setattr(keepout.cdm, "loads", counted)
    listed = tmp_store.events(event_ids=[late, early], include_archived=True)
    assert [e.event_id for e in listed] == [early, late]
    assert sorted(parsed) == ["A", "B"]
    assert [e.event_id for e in tmp_store.events(event_ids=[late, early])] == [early]
    assert tmp_store.events(event_ids=[]) == []
    with pytest.raises(StoreError, match="no event 12345678 in the store"):
        tmp_store.events(event_ids=["12345678", early])
    assert other not in {e.event_id for e in listed}


def test_an_empty_batch_writes_nothing(tmp_store: Store) -> None:
    assert tmp_store.ingest([]) == IngestReport((), (), (), ())
    assert tmp_store.events() == []


def _reference(
    tmp_path: Path, name: str, content: str | None = None
) -> CatalogReference:
    path = written(tmp_path / name, content or f"catalog {name}\n")
    return CatalogReference.of_file(path)


def _snapshots(store: Store) -> set[str]:
    folder = store.path.with_name(f"{store.path.name}.catalogs")
    return {p.name for p in folder.iterdir()} if folder.exists() else set()


def _snapshot(store: Store, file: CatalogFileRef) -> Path:
    folder = store.path.with_name(f"{store.path.name}.catalogs")
    return folder / f"{file.sha256}{''.join(file.path.suffixes)}"


def _as_stored(store: Store, reference: CatalogReference) -> CatalogReference:
    """``reference`` as the store keeps it: its files in the store's snapshots."""
    return CatalogReference(
        reference.label,
        tuple(
            dataclasses.replace(file, path=_snapshot(store, file))
            for file in reference.files
        ),
    )


def test_a_catalog_is_kept_with_the_cdms_made_from_it(
    tmp_store: Store, tmp_path: Path
) -> None:
    retrieved = Time("2026-11-01T00:00:00.123456", scale="utc")
    active = written(tmp_path / "active.csv", "active\n")
    extra = written(tmp_path / "extra.tar.gz", "extra\n")
    reference = CatalogReference(
        "celestrak",
        (
            CatalogFileRef(
                "active",
                active,
                hashlib.sha256(b"active\n").hexdigest(),
                url="https://example.org/gp",
                retrieved=retrieved,
            ),
            CatalogFileRef("extra", extra, hashlib.sha256(b"extra\n").hexdigest()),
        ),
    )
    (event_id,) = tmp_store.ingest([message("A")], catalog=reference).events
    found = tmp_store.catalog(event_id)
    assert found is not None
    assert found.label == "celestrak"
    assert [(f.name, f.path, f.sha256, f.url) for f in found.files] == [
        (f.name, _snapshot(tmp_store, f), f.sha256, f.url) for f in reference.files
    ]
    assert found.files[1].path.name.endswith(".tar.gz")
    assert found.files[0].path.read_text() == "active\n"
    first, second = found.files
    assert first.retrieved is not None
    assert abs((first.retrieved - retrieved).to_value("s")) < 1e-6
    assert second.retrieved is None
    # A CDM created later, from no catalog, is the event's latest.
    tmp_store.ingest([message("B", tca=1.0, created=-60.0)])
    assert tmp_store.catalog(event_id) is None
    # Ingesting nothing new stores no catalog.
    tmp_store.ingest([message("B", tca=1.0, created=-60.0)], catalog=reference)
    assert _count(tmp_store, models.Catalog) == 1


def test_a_stored_catalog_outlives_its_original_file(
    tmp_store: Store, tmp_path: Path
) -> None:
    path = tmp_path / "active.csv"
    shutil.copyfile(SAMPLE_CSV, path)
    loaded = load_reference(CatalogReference.of_file(path))
    (event_id,) = tmp_store.ingest(
        [message("A")], catalog=CatalogReference.of_file(path)
    ).events
    path.write_text("overwritten by the next download\n")
    stored = tmp_store.catalog(event_id)
    assert stored is not None
    assert load_reference(stored) == loaded


def test_a_catalog_changed_since_its_reference_is_not_stored(
    tmp_store: Store, tmp_path: Path
) -> None:
    reference = _reference(tmp_path, "active.csv")
    reference.files[0].path.write_text("changed\n")
    with pytest.raises(ChecksumMismatch):
        tmp_store.ingest([message("A")], catalog=reference)
    assert _count(tmp_store, models.Cdm) == 0
    assert _snapshots(tmp_store) == set()


def test_a_failed_ingest_leaves_a_snapshot_a_later_one_reuses(
    tmp_store: Store, tmp_path: Path
) -> None:
    tmp_store.ingest([message("A", tca=60.0)])
    reference = _reference(tmp_path, "active.csv")
    # B makes a new event and the catalog's snapshot, then A' fails the batch.
    batch = [message("B", secondary="40000"), message("A", tca=120.0)]
    with pytest.raises(StoreError, match="MESSAGE_ID A is stored with other content"):
        tmp_store.ingest(batch, catalog=reference)
    assert (_count(tmp_store, models.Event), _count(tmp_store, models.Cdm)) == (1, 1)
    assert _count(tmp_store, models.Catalog) == 0
    assert _count(tmp_store, models.CatalogFile) == 0
    snapshot = _snapshot(tmp_store, reference.files[0])
    assert _snapshots(tmp_store) == {snapshot.name}
    inode = snapshot.stat().st_ino
    (event_id,) = tmp_store.ingest([batch[0]], catalog=reference).events
    assert snapshot.stat().st_ino == inode
    assert tmp_store.catalog(event_id) == _as_stored(tmp_store, reference)


def test_the_catalog_of_the_latest_cdm_wins(tmp_store: Store, tmp_path: Path) -> None:
    old = _reference(tmp_path, "old.csv")
    new = _reference(tmp_path, "new.csv")
    (event_id,) = tmp_store.ingest([message("A")], catalog=old).events
    tmp_store.ingest([message("B", created=-60.0)], catalog=new)
    assert tmp_store.catalog(event_id) == _as_stored(tmp_store, new)


def test_refresh_computes_once_per_key(tmp_store: Store) -> None:
    first = tmp_store.ingest([message("A"), message("B", tca=1.0, created=-60.0)])
    second = tmp_store.ingest([message("C", secondary="40000")])
    assert tmp_store.refresh(FOSTER).computed == 3
    assert tmp_store.refresh(FOSTER).computed == 0
    event_id = first.events[0]
    seen = tmp_store.observations(event_id, FOSTER)
    expected = tmp_store.event(event_id).observations(
        compute=FOSTER.compute, combined_hard_body_radius=20.0
    )
    assert [observation_to_json(o) for o in seen] == [
        observation_to_json(o) for o in expected
    ]
    assert [o.message_id for o in seen] == ["A", "B"]
    wider = Settings(compute=computation("foster"), combined_hard_body_radius=30.0)
    assert tmp_store.refresh(wider, event_ids=second.events).computed == 1
    with pytest.raises(StoreError, match="refresh"):
        tmp_store.observations(event_id, wider)
    assert tmp_store.refresh(wider).computed == 2
    assert tmp_store.observations(event_id, wider)[0].pc != seen[0].pc
    # A new CDM is computed alone.
    tmp_store.ingest([message("D", tca=2.0, created=-30.0)])
    assert tmp_store.refresh(FOSTER).computed == 1
    assert len(tmp_store.observations(event_id, FOSTER)) == 3


def test_observations_without_refresh_raise(tmp_store: Store) -> None:
    (event_id,) = tmp_store.ingest([message("A")]).events
    with pytest.raises(StoreError, match=f"no Pc of A in event {event_id}"):
        tmp_store.observations(event_id, FOSTER)


def test_refresh_needs_a_radius_when_the_cdms_give_none(tmp_store: Store) -> None:
    report = tmp_store.ingest([message("A")])
    with pytest.raises(HardBodyRadiusError):
        tmp_store.refresh(
            Settings(compute=computation("foster")), event_ids=report.events
        )


def test_a_refresh_of_every_event_skips_cdms_without_hbr(tmp_store: Store) -> None:
    crossing, drifting = (keepout.cdm.loads(text) for text in operator_kvn())
    bare = keepout.cdm.loads(without_hbr(keepout.cdm.kvn.dumps(drifting)))
    report = tmp_store.ingest(
        [CdmInput.of(crossing, "test"), CdmInput.of(bare, "test")]
    )
    no_radius = Settings(compute=computation("foster"))
    assert tmp_store.refresh(no_radius) == RefreshReport(1, (bare.message_id,))
    assert tmp_store.refresh(no_radius) == RefreshReport(0, (bare.message_id,))
    # Events named: their CDMs are all needed, so a missing HBR is an error.
    with pytest.raises(HardBodyRadiusError, match=bare.message_id):
        tmp_store.refresh(no_radius, event_ids=report.events)
    assert tmp_store.refresh(FOSTER) == RefreshReport(2, ())


def test_a_result_written_meanwhile_is_kept(
    tmp_store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Another process caches the same Pc, or purges the CDM, while this one
    # computes it: the writes skip it.
    (event_id,) = tmp_store.ingest([message("A"), message("B", tca=1.0)]).events
    original = ConjunctionEvent.observations
    calls = itertools.count()

    def racing(event: ConjunctionEvent, **options: object) -> object:
        seen = original(event, **options)  # type: ignore[arg-type]
        if next(calls) == 0:
            with open_store(tmp_store.path) as other:
                other.refresh(FOSTER, event_ids=[event_id])
        return seen

    monkeypatch.setattr(ConjunctionEvent, "observations", racing)
    assert tmp_store.refresh(FOSTER).computed == 0
    monkeypatch.undo()
    assert len(tmp_store.observations(event_id, FOSTER)) == 2


def test_a_result_of_a_cdm_purged_meanwhile_is_dropped(
    tmp_store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_store.ingest([message("A")])
    original = ConjunctionEvent.observations

    def racing(event: ConjunctionEvent, **options: object) -> object:
        seen = original(event, **options)  # type: ignore[arg-type]
        tmp_store.purge(TCA + 1.0 * u.s)
        return seen

    monkeypatch.setattr(ConjunctionEvent, "observations", racing)
    assert tmp_store.refresh(FOSTER).computed == 0
    assert _count(tmp_store, models.Result) == 0


def test_purge_cascades(tmp_store: Store, tmp_path: Path) -> None:
    old = _reference(tmp_path, "old.csv")
    kept = _reference(tmp_path, "kept.csv")
    tmp_store.ingest([message("A"), message("B", tca=1.0)], catalog=old)
    (late,) = tmp_store.ingest([message("C", tca=3 * TOLERANCE)], catalog=kept).events
    tmp_store.refresh(FOSTER)
    assert _count(tmp_store, models.Result) == 3
    assert tmp_store.purge(TCA + TOLERANCE * u.s) == 1
    assert [e.event_id for e in tmp_store.events()] == [late]
    assert _count(tmp_store, models.Cdm) == 1
    assert _count(tmp_store, models.Result) == 1
    assert _count(tmp_store, models.Catalog) == 1
    assert _count(tmp_store, models.CatalogFile) == 1
    assert tmp_store.catalog(late) == _as_stored(tmp_store, kept)
    assert _snapshots(tmp_store) == {_snapshot(tmp_store, kept.files[0]).name}
    # An event whose latest TCA is the date itself is kept.
    assert tmp_store.purge(TCA + 3 * TOLERANCE * u.s) == 0
    assert tmp_store.purge(Time("2020-01-01", scale="utc")) == 1
    assert _count(tmp_store, models.Catalog) == 0
    assert _snapshots(tmp_store) == set()


def test_purge_keeps_a_snapshot_another_catalog_shares(
    tmp_store: Store, tmp_path: Path
) -> None:
    old = _reference(tmp_path, "old.csv", "same\n")
    other = _reference(tmp_path, "other.csv", "other\n")
    kept = _reference(tmp_path, "kept.csv", "same\n")
    tmp_store.ingest([message("A")], catalog=old)
    tmp_store.ingest([message("B", secondary="40000")], catalog=other)
    tmp_store.ingest([message("C", tca=3 * TOLERANCE)], catalog=kept)
    assert _snapshots(tmp_store) == {
        _snapshot(tmp_store, other.files[0]).name,
        _snapshot(tmp_store, kept.files[0]).name,
    }
    assert tmp_store.purge(TCA + TOLERANCE * u.s) == 2
    assert _count(tmp_store, models.Catalog) == 1
    assert _snapshots(tmp_store) == {_snapshot(tmp_store, kept.files[0]).name}


def test_a_copy_ingested_again_while_purging_survives(
    tmp_store: Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Another process ingests the same catalog between the purge's commit of
    # its deletions and its deletion of the copies: the copy stays.
    path = tmp_path / "active.csv"
    shutil.copyfile(SAMPLE_CSV, path)
    reference = CatalogReference.of_file(path)
    tmp_store.ingest([message("A")], catalog=reference)
    original = Store._delete_copies
    ingested: list[str] = []

    def interleaved(store: Store) -> None:
        with open_store(store.path) as other:
            report = other.ingest([message("B", tca=3 * TOLERANCE)], catalog=reference)
        ingested.extend(report.events)
        original(store)

    monkeypatch.setattr(Store, "_delete_copies", interleaved)
    assert tmp_store.purge(TCA + TOLERANCE * u.s) == 1
    (event_id,) = ingested
    stored = tmp_store.catalog(event_id)
    assert stored is not None
    assert load_reference(stored) == load_reference(reference)


def test_copies_are_found_from_any_path_to_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    here, elsewhere = tmp_path / "here", tmp_path / "elsewhere"
    here.mkdir()
    elsewhere.mkdir()
    reference = CatalogReference.of_file(
        shutil.copyfile(SAMPLE_CSV, tmp_path / "a.csv")
    )
    monkeypatch.chdir(here)
    with open_store("k.db") as store:
        assert store.path == here / "k.db"
        (event_id,) = store.ingest([message("A")], catalog=reference).events
        with store.read() as session:
            stored = session.scalars(select(models.CatalogFile.path)).all()
    sha256 = reference.files[0].sha256
    assert stored == [f"k.db.catalogs/{sha256}.csv"]
    monkeypatch.chdir(elsewhere)
    with open_store(here / "k.db") as store:
        found = store.catalog(event_id)
        assert found is not None
        assert found.files[0].path == here / "k.db.catalogs" / f"{sha256}.csv"
        assert load_reference(found) == load_reference(reference)
        assert store.purge(Time("2020-01-01", scale="utc")) == 1
        assert _snapshots(store) == set()


def test_a_store_moved_with_its_directory_keeps_its_catalogs(tmp_path: Path) -> None:
    reference = CatalogReference.of_file(
        shutil.copyfile(SAMPLE_CSV, tmp_path / "a.csv")
    )
    with open_store(tmp_path / "one" / "k.db") as store:
        (event_id,) = store.ingest([message("A")], catalog=reference).events
    shutil.move(tmp_path / "one", tmp_path / "two")
    with open_store(tmp_path / "two" / "k.db") as store:
        found = store.catalog(event_id)
    assert found is not None
    assert found.files[0].path.parent == tmp_path / "two" / "k.db.catalogs"
    assert load_reference(found) == load_reference(reference)


def test_purge_removes_a_copy_a_failed_ingest_left(
    tmp_store: Store, tmp_path: Path
) -> None:
    tmp_store.ingest([message("A", tca=60.0)])
    reference = _reference(tmp_path, "active.csv")
    batch = [message("B", secondary="40000"), message("A", tca=120.0)]
    with pytest.raises(StoreError, match="stored with other content"):
        tmp_store.ingest(batch, catalog=reference)
    kept = _reference(tmp_path, "kept.csv")
    tmp_store.ingest([message("C", secondary="50000")], catalog=kept)
    assert len(_snapshots(tmp_store)) == 2
    assert tmp_store.purge(Time("2000-01-01", scale="utc")) == 0
    assert _snapshots(tmp_store) == {_snapshot(tmp_store, kept.files[0]).name}


def test_a_catalog_that_cannot_be_copied_is_a_store_error(
    tmp_store: Store, tmp_path: Path
) -> None:
    tmp_store.catalog_dir.write_text("in the way\n")
    with pytest.raises(StoreError, match="cannot copy the catalog into"):
        tmp_store.ingest([message("A")], catalog=_reference(tmp_path, "a.csv"))
    assert _count(tmp_store, models.Cdm) == 0


def test_purge_deletes_only_the_stores_own_copies(
    tmp_store: Store, tmp_path: Path
) -> None:
    outside = written(tmp_path / "users.csv", "the user's own file\n")
    tmp_store.ingest([message("A")], catalog=_reference(tmp_path, "a.csv"))
    with tmp_store.write() as session:
        session.execute(text("UPDATE catalog_files SET path = :p"), {"p": str(outside)})
    assert tmp_store.purge(Time("2020-01-01", scale="utc")) == 1
    assert outside.exists()


def test_a_copy_that_cannot_be_deleted_is_a_store_error(
    tmp_store: Store, tmp_path: Path
) -> None:
    reference = _reference(tmp_path, "a.csv")
    tmp_store.ingest([message("A")], catalog=reference)
    copy = _snapshot(tmp_store, reference.files[0])
    copy.unlink()
    copy.mkdir()
    with pytest.raises(StoreError, match="cannot delete the catalog copy"):
        tmp_store.purge(Time("2020-01-01", scale="utc"))
    assert _count(tmp_store, models.Catalog) == 0


def test_a_lock_held_too_long_is_a_store_error(tmp_store: Store) -> None:
    @event.listens_for(tmp_store.engine, "connect")
    def _impatient(connection: sqlite3.Connection, _: object) -> None:
        connection.execute("PRAGMA busy_timeout=50")

    holder = sqlite3.connect(tmp_store.path, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    try:
        expected = f"cannot use the store {re.escape(str(tmp_store.path))}"
        with pytest.raises(StoreError, match=expected):
            tmp_store.ingest([message("A")])
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    assert tmp_store.ingest([message("A")]).added == ("A",)


def test_a_store_file_broken_after_opening_is_a_store_error(tmp_store: Store) -> None:
    tmp_store.ingest([message("A")])
    tmp_store.close()
    for suffix in ("-wal", "-shm"):
        tmp_store.path.with_name(tmp_store.path.name + suffix).unlink(missing_ok=True)
    tmp_store.path.write_bytes(b"not a database at all, " * 100)
    with pytest.raises(StoreError, match=r"cannot use the store .*not a database"):
        tmp_store.events()


def test_a_database_error_while_reading_is_a_store_error(tmp_store: Store) -> None:
    with (
        pytest.raises(StoreError, match="no such table: nowhere") as caught,
        tmp_store.read() as session,
    ):
        session.execute(text("SELECT * FROM nowhere"))
    assert str(tmp_store.path) in str(caught.value)


@pytest.mark.parametrize(
    ("after_tca", "expected"),
    [
        (0.0, 0),
        (1.0, 0),  # the first event's latest TCA itself: kept
        (1.5, 1),
        (3 * TOLERANCE, 1),
        (3 * TOLERANCE + 0.5, 2),
    ],
)
def test_count_before_counts_what_purge_deletes(
    tmp_store: Store, after_tca: float, expected: int
) -> None:
    tmp_store.ingest([message("A"), message("B", tca=1.0)])
    tmp_store.ingest([message("C", tca=3 * TOLERANCE)])
    before = TCA + after_tca * u.s
    assert tmp_store.count_before(before) == expected
    assert len(tmp_store.events()) == 2
    assert tmp_store.purge(before) == expected
    assert tmp_store.count_before(before) == 0


@settings(max_examples=25, deadline=None)
@given(cdms())
def test_a_cdm_comes_back_from_the_store_as_it_went_in(cdm: keepout.cdm.Cdm) -> None:
    with (
        tempfile.TemporaryDirectory() as folder,
        open_store(Path(folder) / "keepout.db") as store,
    ):
        (event_id,) = store.ingest([CdmInput.of(cdm, "test")]).events
        (stored,) = store.event(event_id).cdms
    assert keepout.cdm.kvn.dumps(stored) == keepout.cdm.kvn.dumps(cdm)
