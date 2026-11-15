"""The events list of the dashboard: a page of event summaries from the store."""

import astropy.units as u
import pytest
from sqlalchemy import delete
from store_builders import FOSTER, TCA, message

import keepout.cdm
from keepout.cdm import Cdm
from keepout.errors import StoreError
from keepout.figures import Provenance
from keepout.settings import Settings
from keepout.store import JobKind, JobQueue, Store, models
from keepout.store.codec import result_key
from keepout.store.summaries import complete_of, decoded

BEFORE = TCA - 10 * u.day
"""An ``as_of`` before every builder event's TCA."""


def _events(store: Store, count: int) -> list[str]:
    """``count`` events of one CDM each, an hour apart, the first at
    :data:`~store_builders.TCA`, their Pc cached; their ids by TCA."""
    report = store.ingest(
        [
            message(f"M{i:02d}", tca=3600.0 * i, secondary=str(40_000 + i))
            for i in range(count)
        ]
    )
    store.refresh(FOSTER)
    return list(report.events)


def test_summaries_page_and_order(store: Store) -> None:
    ids = _events(store, 30)
    first = store.summaries(FOSTER, as_of=BEFORE)
    assert (first.total, first.offset, first.limit) == (30, 0, 25)
    assert [item.event_id for item in first.items] == ids[:25]
    second = store.summaries(FOSTER, as_of=BEFORE, offset=25)
    assert (second.total, second.offset) == (30, 25)
    assert [item.event_id for item in second.items] == ids[25:]
    item = first.items[1]
    assert (item.object1, item.object2) == ("12345", "40001")
    assert (item.tca - (TCA + 3600.0 * u.s)).to_value(u.s) == pytest.approx(0.0)
    assert item.cdm_count == 1
    assert item.latest is not None
    assert item.latest.message_id == "M01"
    assert item.assessment is not None
    assert item.provenance is Provenance.CDM_COVARIANCE
    assert item.active_job is None


def test_summaries_hide_past(store: Store) -> None:
    past, future = _events(store, 2)
    as_of = TCA + 60 * u.s  # after the first event's TCA, before the second's
    shown = store.summaries(FOSTER, as_of=as_of)
    assert ([item.event_id for item in shown.items], shown.total) == ([future], 1)
    every = store.summaries(FOSTER, as_of=as_of, include_past=True)
    assert [item.event_id for item in every.items] == [past, future]


@pytest.mark.parametrize("limit", [0, 201])
def test_summaries_limit_is_bounded(store: Store, limit: int) -> None:
    with pytest.raises(ValueError, match="limit"):
        store.summaries(FOSTER, as_of=BEFORE, limit=limit)


def test_summaries_offset_is_not_negative(store: Store) -> None:
    with pytest.raises(ValueError, match="offset"):
        store.summaries(FOSTER, as_of=BEFORE, offset=-1)


def test_summary_risk_equals_assess(store: Store) -> None:
    store.ingest(
        [
            message("A", created=-86_400.0),
            message("B", tca=2.0, created=-3_600.0),
            message("C", secondary="40000"),
            message("D", secondary="40000", tca=1.0, created=-600.0),
        ]
    )
    store.refresh(FOSTER)
    page = store.summaries(FOSTER, as_of=BEFORE)
    assert len(page.items) == 2
    for item in page.items:
        (assessed,) = store.assess([item.event_id], FOSTER)
        assert item.assessment is not None
        assert item.assessment.decision.action == assessed.assessment.decision.action
        assert item.assessment.level == assessed.assessment.level
        assert item.cdm_count == 2


def test_summary_with_missing_pc(store: Store) -> None:
    (event_id,) = store.ingest(
        [message("A"), message("B", tca=1.0, created=-60.0)]
    ).events
    store.refresh(FOSTER)
    assert store.missing_pc(event_id, FOSTER) == ()
    with store.write() as session:
        session.execute(
            delete(models.Result).where(
                models.Result.message_id == "A",
                models.Result.key == result_key(FOSTER),
            )
        )
    (item,) = store.summaries(FOSTER, as_of=BEFORE).items
    assert item.assessment is None
    assert item.latest is not None
    assert item.latest.message_id == "B"
    assert item.provenance is Provenance.CDM_COVARIANCE
    assert item.cdm_count == 2
    assert store.missing_pc(event_id, FOSTER) == ("A",)


def test_summary_without_the_latest_pc(store: Store) -> None:
    (event_id,) = store.ingest([message("A")]).events
    (item,) = store.summaries(FOSTER, as_of=BEFORE).items
    assert (item.latest, item.assessment, item.provenance) == (None, None, None)
    assert store.missing_pc(event_id, FOSTER) == ("A",)


def test_summary_shows_the_active_job(store: Store) -> None:
    (event_id,) = store.ingest([message("A")]).events
    submitted = JobQueue(store).submit_refresh(event_id, FOSTER).job
    assert submitted is not None
    (item,) = store.summaries(FOSTER, as_of=BEFORE).items
    assert item.active_job is JobKind.REFRESH
    JobQueue(store).cancel(submitted.id)
    (item,) = store.summaries(FOSTER, as_of=BEFORE).items
    assert item.active_job is None


def test_summaries_parse_only_listed_latest_cdms(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    _events(store, 30)
    parsed: list[str] = []
    loads = keepout.cdm.loads

    def spy(text: str) -> Cdm:
        made = loads(text)
        parsed.append(made.message_id)
        return made

    monkeypatch.setattr(keepout.cdm, "loads", spy)
    page = store.summaries(FOSTER, as_of=BEFORE)
    latest = [item.latest for item in page.items if item.latest is not None]
    assert sorted(parsed) == sorted(str(o.message_id) for o in latest)
    assert len(parsed) == len(page.items) == 25


def test_missing_pc_of_an_unknown_event(store: Store) -> None:
    with pytest.raises(StoreError, match="no event nothing"):
        store.missing_pc("nothing", FOSTER)


def test_summaries_of_an_empty_store(store: Store) -> None:
    page = store.summaries(FOSTER, as_of=BEFORE, limit=10)
    assert (page.items, page.total, page.offset, page.limit) == ((), 0, 0, 10)


def _drop_result(store: Store, message_id: str) -> None:
    """Forget the Pc of a CDM cached for :data:`~store_builders.FOSTER`."""
    with store.write() as session:
        session.execute(
            delete(models.Result).where(
                models.Result.message_id == message_id,
                models.Result.key == result_key(FOSTER),
            )
        )


def test_cached_observations_with_one_missing(store: Store) -> None:
    (event_id,) = store.ingest(
        [message("A"), message("B", tca=1.0, created=-60.0)]
    ).events
    store.refresh(FOSTER)
    complete = store.observations(event_id, FOSTER)
    _drop_result(store, "A")
    pairs = store.cached_observations(event_id, FOSTER)
    assert [message_id for message_id, _ in pairs] == ["A", "B"]
    assert pairs[0][1] is None
    latest = pairs[1][1]
    assert latest is not None
    assert latest.pc == complete[1].pc
    with pytest.raises(StoreError, match="no Pc of A in event"):
        store.observations(event_id, FOSTER)
    with pytest.raises(StoreError, match="no event nothing"):
        store.cached_observations("nothing", FOSTER)


def test_detail_of_an_unknown_event(store: Store) -> None:
    assert store.detail("nothing", FOSTER) is None


def test_detail_reads_event_pc_and_refresh_jobs(store: Store) -> None:
    (event_id,) = store.ingest(
        [message("A"), message("B", tca=1.0, created=-60.0)]
    ).events
    store.refresh(FOSTER)
    detail = store.detail(event_id, FOSTER)
    assert detail is not None
    assert [cdm.message_id for cdm in detail.event.cdms] == ["A", "B"]
    assert detail.missing == ()
    assert detail.complete is not None
    assert [o.pc for o in detail.complete] == [
        o.pc for o in store.observations(event_id, FOSTER)
    ]
    assert detail.refresh_jobs == ()

    _drop_result(store, "A")
    queue = JobQueue(store)
    job = queue.submit_refresh(event_id, FOSTER).job
    assert job is not None
    # A refresh for other settings, and a manoeuvre job, are not listed.
    other = queue.submit_refresh(event_id, Settings(combined_hard_body_radius=5.0))
    assert other.job is not None
    queue.submit(event_id, FOSTER)
    detail = store.detail(event_id, FOSTER)
    assert detail is not None
    assert detail.missing == ("A",)
    assert detail.complete is None
    assert detail.observations[1][1] is not None
    assert detail.refresh_jobs == (job.id,)
    queue.cancel(job.id)
    detail = store.detail(event_id, FOSTER)
    assert detail is not None
    assert detail.refresh_jobs == ()


def test_complete_of_needs_every_observation(store: Store) -> None:
    (event_id,) = store.ingest([message("A")]).events
    store.refresh(FOSTER)
    (observation,) = store.observations(event_id, FOSTER)
    assert complete_of([observation, observation]) == (observation, observation)
    assert complete_of([observation, None]) is None
    assert complete_of([]) == ()
    assert decoded([None]) == (None,)
