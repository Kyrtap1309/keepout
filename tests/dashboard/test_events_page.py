"""The dashboard's events page: the stored events a page at a time, each
row's risk in colour and words, links to the event page, and the upload."""

import re
from pathlib import Path

import dashboard_seed as seed
import pytest
from astropy.time import Time
from builders import CDM_DATA, leo_crossing
from dashboard_builders import OpenApp
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import Block, PageLink

import keepout.cdm
from keepout.cdm import kvn
from keepout.dashboard import text
from keepout.dashboard.text import literal
from keepout.dashboard.upload import MAX_FILE_BYTES
from keepout.figures import RISK_LABELS
from keepout.settings import load_settings
from keepout.store import CdmInput, EventSummary, JobKind, JobQueue, open_store
from keepout.units import elapsed_seconds

COLUMNS = 10
"""Cells in a row of the list."""

SPAN = re.compile(r"<span [^>]*>(?P<text>[^<]*)</span>")


def _text(cell: Block) -> str:
    """What a cell of the list shows: its elements' values, a risk chip's
    words without its markup."""
    values: list[str] = []
    for node in cell.children.values():
        value = str(getattr(node, "value", ""))
        chip = SPAN.fullmatch(value)
        values.append(chip["text"] if chip else value)
    return " ".join(values)


def _rows(app: AppTest) -> list[list[str]]:
    """The text of each cell of each row of the list, after its header."""
    count = len(_event_links(app))
    cells = [_text(column) for column in app.columns]
    assert cells[:3] == ["**Event**", "**Objects**", "**TCA (UTC)**"] or not count
    return [
        cells[start : start + COLUMNS]
        for start in range(COLUMNS, COLUMNS * (count + 1), COLUMNS)
    ]


def _event_links(app: AppTest) -> list[PageLink]:
    """The list's links to the event page, one per row."""
    links = [link for link in app.get("page_link") if isinstance(link, PageLink)]
    return [link for link in links if link.proto.page == "event"]


def _summaries(store_path: Path, as_of: Time) -> tuple[EventSummary, ...]:
    """The first page of upcoming events of the store, as the page reads it."""
    with open_store(store_path, create=False) as store:
        return store.summaries(load_settings(seed.SETTINGS), as_of=as_of).items


def test_events_list_shows_rows(app_test: OpenApp, seed_db: Path) -> None:
    app = app_test()
    assert not app.exception
    assert not app.error
    as_of = seed.seed_as_of()
    summaries = _summaries(seed_db, as_of)
    assert len(summaries) == 7
    rows = _rows(app)
    assert len(rows) == 7
    for row, summary in zip(rows, summaries, strict=True):
        event, objects, tca, countdown, pc, origin, risk, cdms, _, plan = row
        assert event == summary.event_id
        assert objects == literal(f"{summary.object1} / {summary.object2}")
        assert tca == str(summary.tca.utc.isot)[:19]
        assert elapsed_seconds(as_of, summary.tca) >= 0
        assert countdown == text.countdown(summary.tca, as_of)
        assert summary.latest is not None
        assert summary.latest.pc is not None
        assert pc == f"{summary.latest.pc:.2e}"
        assert summary.provenance is not None
        assert summary.provenance.value in origin
        assert summary.assessment is not None
        level = summary.assessment.level
        assert level is not None
        assert risk.startswith(RISK_LABELS[level])
        assert cdms == str(summary.cdm_count)
        assert plan == "Planner"
    assert [caption.value for caption in app.caption] == ["Events 1 to 7 of 7"]


def test_risk_shown_as_text(app_test: OpenApp) -> None:
    app = app_test()
    labels = {*RISK_LABELS.values(), "not computed"}
    risks = [row[6].split(" · ")[0] for row in _rows(app)]
    assert risks
    assert set(risks) <= labels
    # And as colour: each chip is filled with its level's colour.
    chips = [item.value for item in app.markdown if "<span" in item.value]
    assert len(chips) == len(risks)
    assert all("background:#" in chip for chip in chips)


def test_risk_not_computed(
    app_test: OpenApp, seed_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sqlalchemy import delete

    from keepout.store import models

    with open_store(seed_copy, create=False) as store, store.write() as session:
        session.execute(delete(models.Result))
    monkeypatch.setenv("KEEPOUT_STORE", str(seed_copy))
    app = app_test()
    assert not app.exception
    rows = _rows(app)
    assert len(rows) == 7
    assert {(row[4], row[5], row[6]) for row in rows} == {
        ("not computed", "", "not computed")
    }


def test_past_toggle(app_test: OpenApp, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEEPOUT_AS_OF", "2027-01-01T00:00:00")
    app = app_test()
    assert not app.exception
    assert _rows(app) == []
    assert [item.value for item in app.info] == ["No upcoming events."]
    app.toggle(key="events-past").set_value(True).run()
    assert not app.exception
    rows = _rows(app)
    assert len(rows) == 12
    assert all(row[3].endswith(" ago") for row in rows)
    assert [item.value for item in app.info] == []


def _many(path: Path, count: int) -> Path:
    """A store of ``count`` events: the CCSDS example against ``count``
    secondaries, in 2010, without Pc."""
    text = (CDM_DATA / "b1_example.kvn").read_text(encoding="utf-8")
    messages = []
    for i in range(count):
        made = text.replace("= 30337", f"= {40000 + i}")
        made = re.sub(
            r"^MESSAGE_ID\s*=.*$", f"MESSAGE_ID = MANY-{i}", made, flags=re.MULTILINE
        )
        messages.append(CdmInput.of(keepout.cdm.loads(made), origin="test"))
    with open_store(path) as store:
        assert len(store.ingest(messages).new_events) == count
    return path


def test_pagination(
    app_test: OpenApp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KEEPOUT_STORE", str(_many(tmp_path / "many.db", 30)))
    monkeypatch.setenv("KEEPOUT_AS_OF", "2010-01-01T00:00:00")
    app = app_test()
    assert not app.exception
    assert len(_rows(app)) == 25
    assert [caption.value for caption in app.caption] == ["Events 1 to 25 of 30"]
    assert app.button(key="events-previous").disabled
    assert not app.button(key="events-next").disabled
    first = [row[0] for row in _rows(app)]

    app.button(key="events-next").click().run()
    assert not app.exception
    second = [row[0] for row in _rows(app)]
    assert len(second) == 5
    assert not set(first) & set(second)
    assert [caption.value for caption in app.caption] == ["Events 26 to 30 of 30"]
    assert app.button(key="events-next").disabled

    app.button(key="events-previous").click().run()
    assert [row[0] for row in _rows(app)] == first


def test_row_links_to_event(app_test: OpenApp) -> None:
    app = app_test()
    links = app.get("page_link")
    # Each row links to its event's page and to its manoeuvre planner.
    assert len(links) == 14
    events = _event_links(app)
    assert len(events) == 7
    for link in events:
        assert link.proto.query_string == f"event={link.value}"
    planners = [link for link in links if link not in events]
    assert [link.proto.page for link in planners] == ["planner"] * 7
    assert [link.proto.label for link in planners] == ["Planner"] * 7
    assert [link.proto.query_string for link in planners] == [
        link.proto.query_string for link in events
    ]


def test_no_chart_so_no_table(app_test: OpenApp) -> None:
    # Every chart is followed by its data table; this page has neither.
    app = app_test()
    assert app.get("plotly_chart") == []
    assert app.dataframe == []


def _upload(app: AppTest, *files: tuple[str, bytes]) -> AppTest:
    """The app after ``files`` are uploaded and ingested."""
    app.file_uploader[0].set_value([(name, data, "text/plain") for name, data in files])
    app.button(key="FormSubmitter:upload-Ingest").click()
    return app.run()


def test_page_uploads_and_queues_refresh(
    app_test: OpenApp, seed_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KEEPOUT_STORE", str(seed_copy))
    fresh = kvn.dumps(leo_crossing().cdms[0]).encode()
    app = _upload(app_test(), ("new.kvn", fresh))
    assert not app.exception
    assert not app.error
    assert [item.value for item in app.success] == [
        "Stored 1 CDM (0 already stored) in 1 event; queued 1 Pc refresh job."
    ]
    with open_store(seed_copy, create=False) as store:
        (job,) = JobQueue(store).jobs()
    assert job.kind is JobKind.REFRESH
    shown = "\n".join(item.value for item in app.markdown)
    assert f"Job {job.id} (refresh): **queued**" in shown
    # Kept while its job is queued, gone once the job has ended.
    app.run()
    assert len(app.success) == 1
    with open_store(seed_copy, create=False) as store:
        JobQueue(store).cancel(job.id)
    app.run()
    assert len(app.success) == 1
    assert "**cancelled**" in "\n".join(item.value for item in app.markdown)
    app.run()
    assert app.success == []


def test_page_shows_upload_errors(
    app_test: OpenApp, seed_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KEEPOUT_STORE", str(seed_copy))
    fresh = kvn.dumps(leo_crossing().cdms[0]).encode()
    app = _upload(app_test(), ("new.kvn", fresh), ("hello.txt", b"hello"))
    assert not app.exception
    assert [item.value for item in app.error] == [
        "hello.txt: expected 'KEYWORD = value', got 'hello' (line 1)"
    ]
    assert app.success == []
    # Nothing ingested, and the list still shows.
    assert len(_rows(app)) == 7
    with open_store(seed_copy, create=False) as store:
        assert JobQueue(store).jobs() == []


def test_page_names_a_file_verbatim(
    app_test: OpenApp, seed_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A file name is the user's text: shown as it is, never as Markdown.
    monkeypatch.setenv("KEEPOUT_STORE", str(seed_copy))
    name = "![](http://host/x.png).kvn"
    app = _upload(app_test(), (name, b"hello"))
    assert not app.exception
    assert [item.value for item in app.error] == [
        f"{literal(name)}: expected 'KEYWORD = value', got 'hello' (line 1)"
    ]


def test_page_names_a_large_file_once(
    app_test: OpenApp, seed_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KEEPOUT_STORE", str(seed_copy))
    big = b"x" * (MAX_FILE_BYTES + 1)
    app = _upload(app_test(), ("big.kvn", big))
    assert not app.exception
    assert [item.value for item in app.error] == [
        f"big.kvn: larger than {MAX_FILE_BYTES} bytes"
    ]
    # Shown once: the next run of the page no longer shows it.
    app.run()
    assert not app.exception
    assert app.error == []
    assert len(_rows(app)) == 7
