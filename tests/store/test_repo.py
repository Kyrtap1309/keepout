import dataclasses
import gzip
import hashlib
import itertools
import re
import shutil
import sqlite3
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import astropy.units as u
import pytest
from astropy.time import Time
from builders import B1_KVN, CDM_DATA
from cdm_strategies import cdms
from cli_builders import operator_kvn, without_hbr
from hypothesis import given, settings
from sqlalchemy import event, select, text
from store_builders import FOSTER, TCA, cdm, cdm_text, count_rows, message, written

import keepout.cdm
from keepout.errors import ChecksumMismatch, HardBodyRadiusError, StoreError
from keepout.events import (
    DEFAULT_TCA_TOLERANCE,
    ConjunctionEvent,
    group_events,
    stable_event_id,
)
from keepout.maneuver import ManeuverOption, ManeuverOutcome
from keepout.pc import computation
from keepout.risk import PcObservation, RiskLevel, Thresholds
from keepout.settings import Settings
from keepout.sources.catalog import CatalogFileRef, CatalogReference, load_reference
from keepout.store import (
    CdmInput,
    IngestReport,
    MessageId,
    RefreshReport,
    Store,
    models,
    open_store,
)
from keepout.store.codec import maneuver_key, observation_to_json
from keepout.store.jobs import JobQueue
from keepout.store.maneuver_codec import outcome_to_json
from keepout.store.repo import pinned_in, save_maneuver_in

TOLERANCE = DEFAULT_TCA_TOLERANCE
SAMPLE_CSV = Path(__file__).parents[1] / "omm" / "data" / "celestrak_sample.csv"


def _partition(store: Store) -> set[frozenset[str]]:
    return {frozenset(c.message_id for c in event.cdms) for event in store.events()}


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


def test_ingest_stores_text_and_groups(store: Store, tmp_path: Path) -> None:
    first = CdmInput.read(written(tmp_path / "a.kvn", cdm_text("A")))
    later = message("B", tca=0.5, created=-3_600.0)
    other_pair = message("C", secondary="40000")
    report = store.ingest([first, later, other_pair])
    assert report.added == ("A", "B", "C")
    assert report.already_stored == ()
    assert len(report.events) == 2
    assert report.new_events == report.events
    # A, created a day before TCA, is the first CDM of its event (B an hour).
    assert report.events == (stable_event_id("A"), stable_event_id("C"))
    events = store.events()
    assert [e.event_id for e in events] == list(report.events)
    assert _partition(store) == {frozenset("AB"), frozenset("C")}
    with store.read() as session:
        row = session.get(models.Cdm, "A")
        assert row is not None
        assert row.content == first.text
        assert (row.format, row.origin) == ("kvn", str(tmp_path / "a.kvn"))
        assert row.tca.isot == "2010-03-13T22:37:52.618"
        assert row.creation_date.isot == "2010-03-12T22:37:52.618"
        stored = session.scalars(select(models.Event).order_by(models.Event.id)).all()
        pair = next(e for e in stored if e.object2 == "30337")
        assert (pair.object1, pair.tca_min.isot, pair.tca_max.isot) == (
            "12345",
            "2010-03-13T22:37:52.618",
            "2010-03-13T22:37:53.118",
        )
        # Nanoseconds, fixed width: text order is time order.
        assert session.scalar(
            text("SELECT tca_max FROM events WHERE id = :id"), {"id": pair.id}
        ) == ("2010-03-13T22:37:53.118000000")
    event = store.event(pair.id)
    assert isinstance(event, ConjunctionEvent)
    assert event.event_id == pair.id
    assert [c.message_id for c in event.cdms] == ["A", "B"]
    assert event.tca_tolerance == TOLERANCE


def test_the_report_lists_messages_in_the_order_given(store: Store) -> None:
    report = store.ingest([message("B", tca=60.0), message("A")])
    assert report.added == ("B", "A")
    again = store.ingest([message("C", tca=30.0), message("B", tca=60.0)])
    assert (again.added, again.already_stored) == (("C",), ("B",))
    assert again.events == report.events
    assert again.new_events == ()


def test_reingest_is_reported_and_changes_nothing(store: Store) -> None:
    batch = [message("A"), message("B", tca=1.0)]
    first = store.ingest(batch)
    with store.read() as session:
        before = session.execute(text("SELECT * FROM cdms ORDER BY message_id")).all()
    again = store.ingest(batch)
    assert again == IngestReport(
        added=(),
        already_stored=(MessageId("A"), MessageId("B")),
        events=first.events,
        new_events=(),
    )
    with store.read() as session:
        after = session.execute(text("SELECT * FROM cdms ORDER BY message_id")).all()
    assert after == before
    assert count_rows(store, models.Event) == 1


def test_a_message_repeated_in_one_batch_is_stored_once(store: Store) -> None:
    report = store.ingest([message("A"), message("A")])
    assert (report.added, report.already_stored) == (("A",), ("A",))
    assert count_rows(store, models.Cdm) == 1


def test_kvn_and_xml_of_one_message_are_one_message(store: Store) -> None:
    kvn = CdmInput.read(B1_KVN)
    xml = CdmInput.read(CDM_DATA / "b1_example.xml")
    assert store.ingest([kvn]).added == (kvn.cdm.message_id,)
    report = store.ingest([xml])
    assert (report.added, report.already_stored) == ((), (xml.cdm.message_id,))
    with store.read() as session:
        row = session.get(models.Cdm, kvn.cdm.message_id)
        assert row is not None
        assert row.format == "kvn"


def test_same_message_id_other_content_raises_and_writes_nothing(
    store: Store, tmp_path: Path
) -> None:
    store.ingest([message("A")])
    # B, of another pair, makes a new event before A fails the batch.
    batch = [message("B", secondary="40000", tca=1.0), message("A", tca=2.0)]
    with pytest.raises(StoreError, match="MESSAGE_ID A is stored with other content"):
        store.ingest(batch, catalog=_reference(tmp_path, "active.csv"))
    assert count_rows(store, models.Cdm) == 1
    assert count_rows(store, models.Event) == 1
    assert count_rows(store, models.Catalog) == 0
    with pytest.raises(StoreError, match="MESSAGE_ID C is stored with other content"):
        store.ingest([message("C"), message("C", tca=2.0)])
    assert count_rows(store, models.Cdm) == 1


def test_a_cdm_beyond_the_tolerance_starts_a_new_event(store: Store) -> None:
    first = store.ingest([message("A")])
    second = store.ingest([message("B", tca=TOLERANCE + 1.0)])
    assert second.new_events != first.new_events
    assert len(second.new_events) == 1
    # Within the tolerance of the first CDM but stretching the event over it.
    third = store.ingest([message("C", tca=-TOLERANCE + 0.5)])
    assert third.events == first.events


def test_a_cdm_fitting_two_events_raises(store: Store) -> None:
    early = store.ingest([message("A")]).events[0]
    late = store.ingest([message("B", tca=1.5 * TOLERANCE)]).events[0]
    with pytest.raises(StoreError, match="CDM M fits more than one event") as caught:
        store.ingest([message("M", tca=0.75 * TOLERANCE)])
    assert early in str(caught.value)
    assert late in str(caught.value)
    assert count_rows(store, models.Cdm) == 2


def test_a_cdm_between_two_events_ingested_apart_raises(store: Store) -> None:
    # 16 min and 0 min are two events; 8 min fits both. In one batch (TCA
    # order) 0 and 8 would make one event and 16 another.
    late = store.ingest([message("L", tca=16 * 60.0)]).events
    early = store.ingest([message("E")]).events
    assert late != early
    with pytest.raises(StoreError, match="CDM M fits more than one event"):
        store.ingest([message("M", tca=8 * 60.0)])


def _with_tca(given: CdmInput, tca: str) -> CdmInput:
    line = re.compile(r"^TCA\s*=.*$", re.MULTILINE)
    text = line.sub(f"TCA = {tca}", given.text, count=1)
    return CdmInput.of(keepout.cdm.loads(text), origin="test")


def test_a_tca_a_fraction_of_a_microsecond_too_far_starts_a_new_event(
    store: Store,
) -> None:
    # Spread 600.0000004 s: beyond the tolerance, though both TCAs round to
    # the same microseconds as a spread of exactly 600 s.
    first = _with_tca(message("A"), "2010-03-13T22:30:00.000")
    second = _with_tca(message("B"), "2010-03-13T22:40:00.0000004")
    assert len(group_events([first.cdm, second.cdm])) == 2
    report = store.ingest([first, second])
    assert len(report.new_events) == 2
    assert _partition(store) == {frozenset("A"), frozenset("B")}
    assert [len(store.event(i).cdms) for i in report.events] == [1, 1]


def test_a_tca_exactly_at_the_tolerance_joins(store: Store) -> None:
    first = _with_tca(message("A"), "2010-03-13T22:30:00.000")
    second = _with_tca(message("B"), "2010-03-13T22:40:00.000")
    assert len(store.ingest([first, second]).events) == 1


def test_a_conjunction_id_keeps_events_apart(store: Store) -> None:
    first = store.ingest([message("A", conjunction_id="X")]).events
    other = store.ingest([message("B", tca=1.0, conjunction_id="Y")]).events
    assert other != first
    same = store.ingest([message("C", tca=2.0, conjunction_id="X")]).events
    assert same == first
    with pytest.raises(StoreError, match="fits more than one event"):
        store.ingest([message("D", tca=3.0)])


def test_an_event_takes_the_conjunction_id_its_cdms_give(store: Store) -> None:
    (event_id,) = store.ingest([message("A")]).events
    assert store.ingest([message("B", conjunction_id="X")]).events == (event_id,)
    assert store.ingest([message("C", conjunction_id="Y")]).events != (event_id,)
    with store.read() as session:
        stored = session.get(models.Event, event_id)
        assert stored is not None
        assert stored.conjunction_id == "X"


def test_event_ids_never_change_after_more_cdms(store: Store) -> None:
    (event_id,) = store.ingest([message("A", tca=5.0)]).events
    for number, offset in enumerate((-5.0, 30.0, 1.0)):
        report = store.ingest([message(f"N{number}", tca=offset)])
        assert report.events == (event_id,)
        assert report.new_events == ()
    assert [e.event_id for e in store.events()] == [event_id]
    assert len(store.event(event_id).cdms) == 4


def test_an_event_id_already_taken_gives_the_next(store: Store) -> None:
    # Another event took the id A's would have: A's event takes the next.
    other = message("B", secondary="40000")
    with store.write() as session:
        session.execute(
            text(
                "INSERT INTO events (id, object1, object2, tca_min, tca_max,"
                " created_at) VALUES (:id, '1', '2', :t, :t, :t)"
            ),
            {"id": stable_event_id("A"), "t": "2010-03-13T22:37:52.618000000"},
        )
    assert store.ingest([message("A")]).events == (stable_event_id("A#1"),)
    assert store.ingest([other]).events == (stable_event_id("B"),)


def test_an_unknown_event_is_an_error(store: Store) -> None:
    for call in (store.event, store.catalog):
        with pytest.raises(StoreError, match="no event 12345678 in the store"):
            call("12345678")
    with pytest.raises(StoreError, match="no event 12345678 in the store"):
        store.observations("12345678", FOSTER)
    with pytest.raises(StoreError, match="no event 12345678 in the store"):
        store.refresh(FOSTER, event_ids=["12345678"])


def test_events_are_listed_by_first_tca(store: Store) -> None:
    late = store.ingest([message("A", tca=3 * TOLERANCE)]).events[0]
    early = store.ingest([message("B")]).events[0]
    other = store.ingest([message("C", secondary="40000", tca=1.0)]).events[0]
    assert [e.event_id for e in store.events()] == [early, other, late]


def test_events_of_given_ids_parse_only_theirs(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    late = store.ingest([message("A", tca=3 * TOLERANCE)]).events[0]
    early = store.ingest([message("B")]).events[0]
    other = store.ingest([message("C", secondary="40000")]).events[0]
    with store.write() as session:
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
    listed = store.events(event_ids=[late, early])
    assert [e.event_id for e in listed] == [early, late]
    assert sorted(parsed) == ["A", "B"]
    assert store.events(event_ids=[]) == []
    with pytest.raises(StoreError, match="no event 12345678 in the store"):
        store.events(event_ids=["12345678", early])
    assert other not in {e.event_id for e in listed}


def test_an_empty_batch_writes_nothing(store: Store) -> None:
    assert store.ingest([]) == IngestReport((), (), (), ())
    assert store.events() == []


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
    store: Store, tmp_path: Path
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
    (event_id,) = store.ingest([message("A")], catalog=reference).events
    found = store.catalog(event_id)
    assert found is not None
    assert found.label == "celestrak"
    assert [(f.name, f.path, f.sha256, f.url) for f in found.files] == [
        (f.name, _snapshot(store, f), f.sha256, f.url) for f in reference.files
    ]
    assert found.files[1].path.name.endswith(".tar.gz")
    assert found.files[0].path.read_text() == "active\n"
    first, second = found.files
    assert first.retrieved is not None
    assert abs((first.retrieved - retrieved).to_value("s")) < 1e-6
    assert second.retrieved is None
    # A CDM created later, from no catalog, is the event's latest.
    store.ingest([message("B", tca=1.0, created=-60.0)])
    assert store.catalog(event_id) is None
    # Ingesting nothing new stores no catalog.
    store.ingest([message("B", tca=1.0, created=-60.0)], catalog=reference)
    assert count_rows(store, models.Catalog) == 1


def test_a_stored_catalog_outlives_its_original_file(
    store: Store, tmp_path: Path
) -> None:
    path = tmp_path / "active.csv"
    shutil.copyfile(SAMPLE_CSV, path)
    loaded = load_reference(CatalogReference.of_file(path))
    (event_id,) = store.ingest(
        [message("A")], catalog=CatalogReference.of_file(path)
    ).events
    path.write_text("overwritten by the next download\n")
    stored = store.catalog(event_id)
    assert stored is not None
    assert load_reference(stored) == loaded


def test_a_catalog_changed_since_its_reference_is_not_stored(
    store: Store, tmp_path: Path
) -> None:
    reference = _reference(tmp_path, "active.csv")
    reference.files[0].path.write_text("changed\n")
    with pytest.raises(ChecksumMismatch):
        store.ingest([message("A")], catalog=reference)
    assert count_rows(store, models.Cdm) == 0
    assert _snapshots(store) == set()


def test_a_failed_ingest_leaves_a_snapshot_a_later_one_reuses(
    store: Store, tmp_path: Path
) -> None:
    store.ingest([message("A", tca=60.0)])
    reference = _reference(tmp_path, "active.csv")
    # B makes a new event and the catalog's snapshot, then A' fails the batch.
    batch = [message("B", secondary="40000"), message("A", tca=120.0)]
    with pytest.raises(StoreError, match="MESSAGE_ID A is stored with other content"):
        store.ingest(batch, catalog=reference)
    assert (count_rows(store, models.Event), count_rows(store, models.Cdm)) == (1, 1)
    assert count_rows(store, models.Catalog) == 0
    assert count_rows(store, models.CatalogFile) == 0
    snapshot = _snapshot(store, reference.files[0])
    assert _snapshots(store) == {snapshot.name}
    inode = snapshot.stat().st_ino
    (event_id,) = store.ingest([batch[0]], catalog=reference).events
    assert snapshot.stat().st_ino == inode
    assert store.catalog(event_id) == _as_stored(store, reference)


def test_the_catalog_of_the_latest_cdm_wins(store: Store, tmp_path: Path) -> None:
    old = _reference(tmp_path, "old.csv")
    new = _reference(tmp_path, "new.csv")
    (event_id,) = store.ingest([message("A")], catalog=old).events
    store.ingest([message("B", created=-60.0)], catalog=new)
    assert store.catalog(event_id) == _as_stored(store, new)


def test_refresh_computes_once_per_key(store: Store) -> None:
    first = store.ingest([message("A"), message("B", tca=1.0, created=-60.0)])
    second = store.ingest([message("C", secondary="40000")])
    assert store.refresh(FOSTER).computed == 3
    assert store.refresh(FOSTER).computed == 0
    event_id = first.events[0]
    seen = store.observations(event_id, FOSTER)
    expected = store.event(event_id).observations(
        compute=FOSTER.compute, combined_hard_body_radius=20.0
    )
    assert [observation_to_json(o) for o in seen] == [
        observation_to_json(o) for o in expected
    ]
    assert [o.message_id for o in seen] == ["A", "B"]
    wider = Settings(compute=computation("foster"), combined_hard_body_radius=30.0)
    assert store.refresh(wider, event_ids=second.events).computed == 1
    with pytest.raises(StoreError, match="refresh"):
        store.observations(event_id, wider)
    assert store.refresh(wider).computed == 2
    assert store.observations(event_id, wider)[0].pc != seen[0].pc
    # A new CDM is computed alone.
    store.ingest([message("D", tca=2.0, created=-30.0)])
    assert store.refresh(FOSTER).computed == 1
    assert len(store.observations(event_id, FOSTER)) == 3


def test_observations_without_refresh_raise(store: Store) -> None:
    (event_id,) = store.ingest([message("A")]).events
    with pytest.raises(StoreError, match=f"no Pc of A in event {event_id}"):
        store.observations(event_id, FOSTER)


def test_refresh_needs_a_radius_when_the_cdms_give_none(store: Store) -> None:
    report = store.ingest([message("A")])
    with pytest.raises(HardBodyRadiusError):
        store.refresh(Settings(compute=computation("foster")), event_ids=report.events)


def test_a_refresh_of_every_event_skips_cdms_without_hbr(store: Store) -> None:
    crossing, drifting = (keepout.cdm.loads(text) for text in operator_kvn())
    bare = keepout.cdm.loads(without_hbr(keepout.cdm.kvn.dumps(drifting)))
    report = store.ingest([CdmInput.of(crossing, "test"), CdmInput.of(bare, "test")])
    no_radius = Settings(compute=computation("foster"))
    skipped = (MessageId(bare.message_id),)
    assert store.refresh(no_radius) == RefreshReport(1, skipped)
    assert store.refresh(no_radius) == RefreshReport(0, skipped)
    # Events named: their CDMs are all needed, so a missing HBR is an error.
    with pytest.raises(HardBodyRadiusError, match=bare.message_id):
        store.refresh(no_radius, event_ids=report.events)
    assert store.refresh(FOSTER) == RefreshReport(2, ())


def test_a_result_written_meanwhile_is_kept(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Another process caches the same Pc, or purges the CDM, while this one
    # computes it: the writes skip it.
    (event_id,) = store.ingest([message("A"), message("B", tca=1.0)]).events
    original = ConjunctionEvent.observations
    calls = itertools.count()

    def racing(event: ConjunctionEvent, **options: object) -> object:
        seen = original(event, **options)  # type: ignore[arg-type]
        if next(calls) == 0:
            with open_store(store.path) as other:
                other.refresh(FOSTER, event_ids=[event_id])
        return seen

    monkeypatch.setattr(ConjunctionEvent, "observations", racing)
    assert store.refresh(FOSTER).computed == 0
    monkeypatch.undo()
    assert len(store.observations(event_id, FOSTER)) == 2


def test_a_result_of_a_cdm_purged_meanwhile_is_dropped(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.ingest([message("A")])
    original = ConjunctionEvent.observations

    def racing(event: ConjunctionEvent, **options: object) -> object:
        seen = original(event, **options)  # type: ignore[arg-type]
        store.purge(TCA + 1.0 * u.s)
        return seen

    monkeypatch.setattr(ConjunctionEvent, "observations", racing)
    assert store.refresh(FOSTER).computed == 0
    assert count_rows(store, models.Result) == 0


def test_purge_cascades(store: Store, tmp_path: Path) -> None:
    old = _reference(tmp_path, "old.csv")
    kept = _reference(tmp_path, "kept.csv")
    store.ingest([message("A"), message("B", tca=1.0)], catalog=old)
    (late,) = store.ingest([message("C", tca=3 * TOLERANCE)], catalog=kept).events
    store.refresh(FOSTER)
    assert count_rows(store, models.Result) == 3
    assert store.purge(TCA + TOLERANCE * u.s) == 1
    assert [e.event_id for e in store.events()] == [late]
    assert count_rows(store, models.Cdm) == 1
    assert count_rows(store, models.Result) == 1
    assert count_rows(store, models.Catalog) == 1
    assert count_rows(store, models.CatalogFile) == 1
    assert store.catalog(late) == _as_stored(store, kept)
    assert _snapshots(store) == {_snapshot(store, kept.files[0]).name}
    # An event whose latest TCA is the date itself is kept.
    assert store.purge(TCA + 3 * TOLERANCE * u.s) == 0
    assert store.purge(Time("2020-01-01", scale="utc")) == 1
    assert count_rows(store, models.Catalog) == 0
    assert _snapshots(store) == set()


def test_purge_keeps_a_snapshot_another_catalog_shares(
    store: Store, tmp_path: Path
) -> None:
    old = _reference(tmp_path, "old.csv", "same\n")
    other = _reference(tmp_path, "other.csv", "other\n")
    kept = _reference(tmp_path, "kept.csv", "same\n")
    store.ingest([message("A")], catalog=old)
    store.ingest([message("B", secondary="40000")], catalog=other)
    store.ingest([message("C", tca=3 * TOLERANCE)], catalog=kept)
    assert _snapshots(store) == {
        _snapshot(store, other.files[0]).name,
        _snapshot(store, kept.files[0]).name,
    }
    assert store.purge(TCA + TOLERANCE * u.s) == 2
    assert count_rows(store, models.Catalog) == 1
    assert _snapshots(store) == {_snapshot(store, kept.files[0]).name}


def test_a_copy_ingested_again_while_purging_survives(
    store: Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Another process ingests the same catalog between the purge's commit of
    # its deletions and its deletion of the copies: the copy stays.
    path = tmp_path / "active.csv"
    shutil.copyfile(SAMPLE_CSV, path)
    reference = CatalogReference.of_file(path)
    store.ingest([message("A")], catalog=reference)
    original = Store._delete_copies
    ingested: list[str] = []

    def interleaved(purging: Store) -> None:
        with open_store(purging.path) as other:
            report = other.ingest([message("B", tca=3 * TOLERANCE)], catalog=reference)
        ingested.extend(report.events)
        original(purging)

    monkeypatch.setattr(Store, "_delete_copies", interleaved)
    assert store.purge(TCA + TOLERANCE * u.s) == 1
    (event_id,) = ingested
    stored = store.catalog(event_id)
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
    store: Store, tmp_path: Path
) -> None:
    store.ingest([message("A", tca=60.0)])
    reference = _reference(tmp_path, "active.csv")
    batch = [message("B", secondary="40000"), message("A", tca=120.0)]
    with pytest.raises(StoreError, match="stored with other content"):
        store.ingest(batch, catalog=reference)
    kept = _reference(tmp_path, "kept.csv")
    store.ingest([message("C", secondary="50000")], catalog=kept)
    assert len(_snapshots(store)) == 2
    assert store.purge(Time("2000-01-01", scale="utc")) == 0
    assert _snapshots(store) == {_snapshot(store, kept.files[0]).name}


def test_a_catalog_that_cannot_be_copied_is_a_store_error(
    store: Store, tmp_path: Path
) -> None:
    store.catalog_dir.write_text("in the way\n")
    with pytest.raises(StoreError, match="cannot copy the catalog into"):
        store.ingest([message("A")], catalog=_reference(tmp_path, "a.csv"))
    assert count_rows(store, models.Cdm) == 0


def test_purge_deletes_only_the_stores_own_copies(store: Store, tmp_path: Path) -> None:
    outside = written(tmp_path / "users.csv", "the user's own file\n")
    store.ingest([message("A")], catalog=_reference(tmp_path, "a.csv"))
    with store.write() as session:
        session.execute(text("UPDATE catalog_files SET path = :p"), {"p": str(outside)})
    assert store.purge(Time("2020-01-01", scale="utc")) == 1
    assert outside.exists()


def test_a_copy_that_cannot_be_deleted_is_a_store_error(
    store: Store, tmp_path: Path
) -> None:
    reference = _reference(tmp_path, "a.csv")
    store.ingest([message("A")], catalog=reference)
    copy = _snapshot(store, reference.files[0])
    copy.unlink()
    copy.mkdir()
    with pytest.raises(StoreError, match="cannot delete the catalog copy"):
        store.purge(Time("2020-01-01", scale="utc"))
    assert count_rows(store, models.Catalog) == 0


def test_a_lock_held_too_long_is_a_store_error(store: Store) -> None:
    @event.listens_for(store.engine, "connect")
    def _impatient(connection: sqlite3.Connection, _: object) -> None:
        connection.execute("PRAGMA busy_timeout=50")

    holder = sqlite3.connect(store.path, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    try:
        expected = f"cannot use the store {re.escape(str(store.path))}"
        with pytest.raises(StoreError, match=expected):
            store.ingest([message("A")])
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    assert store.ingest([message("A")]).added == ("A",)


def test_a_store_file_broken_after_opening_is_a_store_error(store: Store) -> None:
    store.ingest([message("A")])
    store.close()
    for suffix in ("-wal", "-shm"):
        store.path.with_name(store.path.name + suffix).unlink(missing_ok=True)
    store.path.write_bytes(b"not a database at all, " * 100)
    with pytest.raises(StoreError, match=r"cannot use the store .*not a database"):
        store.events()


def test_a_database_error_while_reading_is_a_store_error(store: Store) -> None:
    with (
        pytest.raises(StoreError, match="no such table: nowhere") as caught,
        store.read() as session,
    ):
        session.execute(text("SELECT * FROM nowhere"))
    assert str(store.path) in str(caught.value)


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
    store: Store, after_tca: float, expected: int
) -> None:
    store.ingest([message("A"), message("B", tca=1.0)])
    store.ingest([message("C", tca=3 * TOLERANCE)])
    before = TCA + after_tca * u.s
    assert store.count_before(before) == expected
    assert len(store.events()) == 2
    assert store.purge(before) == expected
    assert store.count_before(before) == 0


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


def test_cache_keeps_observations_computed_elsewhere(store: Store) -> None:
    stored, other = message("A"), message("B", secondary="40000")
    event_id = store.ingest([stored]).events[0]
    computed = [
        observation
        for given in (stored, other)
        for observation in ConjunctionEvent("x", (given.cdm,)).observations(
            compute=FOSTER.compute,
            combined_hard_body_radius=FOSTER.combined_hard_body_radius,
        )
    ]
    # B is not stored: only A's is cached, once.
    assert store.cache(FOSTER, computed) == 1
    assert store.cache(FOSTER, computed) == 0
    assert store.refresh(FOSTER).computed == 0
    (seen,) = store.observations(event_id, FOSTER)
    assert observation_to_json(seen) == observation_to_json(computed[0])


def test_assess_computes_what_is_missing_and_judges_by_the_thresholds(
    store: Store,
) -> None:
    late = store.ingest([message("A", tca=3 * TOLERANCE)]).events[0]
    early = store.ingest([message("B"), message("C", tca=1.0)]).events[0]
    strict = dataclasses.replace(FOSTER, thresholds=Thresholds(1e-12, 1e-11))
    assessed = store.assess([late, early], strict)
    assert [item.event.event_id for item in assessed] == [early, late]
    assert [len(item.event.cdms) for item in assessed] == [2, 1]
    assert {item.assessment.level for item in assessed} == {RiskLevel.RED}
    assert store.refresh(FOSTER).computed == 0
    assert store.assess([], FOSTER) == []


def _five_cdms(store: Store) -> None:
    store.ingest([message(f"M{n}", tca=float(n), created=-60.0 * n) for n in range(5)])


def test_refresh_writes_in_batches(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    _five_cdms(store)
    writes: list[int] = []
    real = Store._write

    def counted(store: Store, key: str, observations: Sequence[PcObservation]) -> int:
        writes.append(len(observations))
        return real(store, key, observations)

    monkeypatch.setattr(Store, "_write", counted)
    monkeypatch.setattr("keepout.store.repo._BATCH_SIZE", 2)
    assert store.refresh(FOSTER).computed == 5
    assert writes == [2, 2, 1]
    # A batch is also written once a second of computing has passed.
    writes.clear()
    monkeypatch.setattr("keepout.store.repo._BATCH_SECONDS", 0.0)
    assert store.refresh(dataclasses.replace(FOSTER, policy="raise")).computed == 5
    assert writes == [1, 1, 1, 1, 1, 0]


def test_an_interrupted_refresh_keeps_the_batches_written(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    _five_cdms(store)
    monkeypatch.setattr("keepout.store.repo._BATCH_SIZE", 2)
    real = ConjunctionEvent.observations
    done: list[str] = []

    def interrupted(event: ConjunctionEvent, **options: Any) -> Any:
        if len(done) == 3:
            raise KeyboardInterrupt
        done.append(event.cdms[0].message_id)
        return real(event, **options)

    monkeypatch.setattr(ConjunctionEvent, "observations", interrupted)
    with pytest.raises(KeyboardInterrupt):
        store.refresh(FOSTER)
    monkeypatch.undo()
    assert store.refresh(FOSTER).computed == 3


def test_refresh_parses_each_cdm_just_before_its_pc(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    _five_cdms(store)
    steps: list[str] = []
    loads = keepout.cdm.loads
    observations = ConjunctionEvent.observations

    def parsed(text: str) -> keepout.cdm.Cdm:
        steps.append("parse")
        return loads(text)

    def computed(event: ConjunctionEvent, **options: Any) -> Any:
        steps.append("compute")
        return observations(event, **options)

    monkeypatch.setattr(keepout.cdm, "loads", parsed)
    monkeypatch.setattr(ConjunctionEvent, "observations", computed)
    store.refresh(FOSTER)
    assert steps == ["parse", "compute"] * 5


def _three_cdms(store: Store) -> str:
    """One event of three CDMs, A, B and C, created in that order; its id."""
    (event_id,) = store.ingest(
        [
            message("A"),
            message("B", tca=1.0, created=-60.0),
            message("C", tca=2.0, created=-30.0),
        ]
    ).events
    return event_id


def _pinned(
    store: Store, event_id: str, message_id: str | None
) -> tuple[ConjunctionEvent, CatalogReference | None]:
    """:func:`pinned_in`, in a read transaction of its own."""
    with store.read() as session:
        return pinned_in(session, store.path.parent, event_id, message_id)


def test_pinned_in_reads_the_event_at_a_cdm(store: Store, tmp_path: Path) -> None:
    first = _reference(tmp_path, "first.csv")
    second = _reference(tmp_path, "second.csv")
    (event_id,) = store.ingest([message("A")], catalog=first).events
    store.ingest([message("B", tca=1.0, created=-60.0)], catalog=second)
    store.ingest([message("C", tca=2.0, created=-30.0)])
    event, catalog = _pinned(store, event_id, "B")
    assert [cdm.message_id for cdm in event.cdms] == ["A", "B"]
    assert catalog == _as_stored(store, second)
    event, catalog = _pinned(store, event_id, "A")
    assert [cdm.message_id for cdm in event.cdms] == ["A"]
    assert catalog == _as_stored(store, first)
    event, catalog = _pinned(store, event_id, None)
    assert event.latest.message_id == "C"
    assert len(event.cdms) == 3
    assert catalog is None
    assert store.catalog(event_id) is None


def test_a_pin_of_another_event_is_refused(store: Store) -> None:
    event_id = _three_cdms(store)
    store.ingest([message("X", secondary="40000")])
    for pin in ("X", "nothing"):
        with pytest.raises(StoreError, match=f"no CDM {pin} in event"):
            _pinned(store, event_id, pin)
    with pytest.raises(StoreError, match="no event 12345678 in the store"):
        _pinned(store, "12345678", None)


def _outcome(event: ConjunctionEvent) -> ManeuverOutcome:
    """An outcome of ``event`` without burns: enough for the cache."""
    baseline = ManeuverOption(
        burn_lead=0.0,
        delta_v=0.0,
        pc=1e-3,
        range_at_tca=150.0,
        method="foster",
        meets_target=False,
    )
    return ManeuverOutcome(
        event=event,
        target_pc=1e-4,
        baseline=baseline,
        notes=("a note",),
        burns=(),
        grid=None,
        catalog_label=None,
    )


def test_a_saved_maneuver_is_read_back(store: Store) -> None:
    event_id = _three_cdms(store)
    assert store.maneuver(event_id, FOSTER) is None
    outcome = _outcome(store.event(event_id))
    assert store.save_maneuver("C", FOSTER, outcome)
    cached = store.maneuver(event_id, FOSTER)
    assert cached is not None
    assert outcome_to_json(cached) == outcome_to_json(outcome)
    assert cached.event.latest.message_id == "C"
    assert len(cached.event.cdms) == 3


def test_the_first_saved_maneuver_is_kept(store: Store) -> None:
    event_id = _three_cdms(store)
    outcome = _outcome(store.event(event_id))
    assert store.save_maneuver("C", FOSTER, outcome)
    other = dataclasses.replace(outcome, target_pc=1e-5)
    assert not store.save_maneuver("C", FOSTER, other)
    cached = store.maneuver(event_id, FOSTER)
    assert cached is not None
    assert cached.target_pc == outcome.target_pc


def test_a_maneuver_of_a_cdm_not_stored_is_not_saved(store: Store) -> None:
    event_id = _three_cdms(store)
    assert not store.save_maneuver("gone", FOSTER, _outcome(store.event(event_id)))
    assert count_rows(store, models.Result) == 0


def test_a_maneuver_is_read_from_the_latest_cdm_only(store: Store) -> None:
    event_id = _three_cdms(store)
    outcome = _outcome(store.event(event_id))
    assert store.save_maneuver("B", FOSTER, outcome)
    assert store.maneuver(event_id, FOSTER) is None
    assert store.save_maneuver("C", FOSTER, outcome)
    assert store.maneuver(event_id, FOSTER) is not None


def test_a_maneuver_is_read_under_its_settings_only(store: Store) -> None:
    event_id = _three_cdms(store)
    assert store.save_maneuver("C", FOSTER, _outcome(store.event(event_id)))
    other = dataclasses.replace(FOSTER, thresholds=Thresholds(red_from=1e-3))
    assert store.maneuver(event_id, other) is None


def test_the_maneuver_of_an_unknown_event(store: Store) -> None:
    with pytest.raises(StoreError, match="no event 12345678 in the store"):
        store.maneuver("12345678", FOSTER)


def test_save_maneuver_in_joins_the_caller_s_transaction(store: Store) -> None:
    event_id = _three_cdms(store)
    key = maneuver_key(FOSTER, None)
    payload = outcome_to_json(_outcome(store.event(event_id)))

    def failing() -> None:
        with store.write() as session:
            assert save_maneuver_in(session, "C", key, payload)
            raise RuntimeError

    with pytest.raises(RuntimeError):
        failing()
    assert store.maneuver(event_id, FOSTER) is None
    with store.write() as session:
        assert save_maneuver_in(session, "C", key, payload)
    assert store.maneuver(event_id, FOSTER) is not None


def test_maneuver_cache_misses_after_a_new_cdm(store: Store) -> None:
    event_id = _three_cdms(store)
    store.save_maneuver("C", FOSTER, _outcome(store.event(event_id)))
    assert store.maneuver(event_id, FOSTER) is not None
    store.ingest([message("D", tca=3.0, created=-10.0)])
    assert store.event(event_id).latest.message_id == "D"
    assert store.maneuver(event_id, FOSTER) is None


def test_purge_drops_jobs_and_results(store: Store) -> None:
    event_id = _three_cdms(store)
    store.refresh(FOSTER)
    store.save_maneuver("C", FOSTER, _outcome(store.event(event_id)))
    assert JobQueue(store).submit(event_id, Settings()).job is not None
    assert count_rows(store, models.Job) == 1
    assert store.purge(TCA + 1.0 * u.day) == 1
    assert count_rows(store, models.Job) == 0
    assert count_rows(store, models.Result) == 0
