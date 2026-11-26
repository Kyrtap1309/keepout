"""The dashboard's events page: the stored events a page at a time, each
row's risk in colour and words, links to the event page, and the upload."""

import json
from pathlib import Path

import dashboard_seed as seed
import pytest
from astropy.time import Time
from builders import B1_TEXT, leo_crossing, set_keyword
from dashboard_builders import OpenApp
from streamlit.testing.v1 import AppTest

import keepout.cdm
from keepout._text import pc_text
from keepout.cdm import kvn
from keepout.dashboard import text
from keepout.dashboard.text import literal
from keepout.dashboard.upload import MAX_FILE_BYTES
from keepout.figures import PROVENANCE_LABELS
from keepout.risk import Action, RiskLevel
from keepout.settings import load_settings
from keepout.store import CdmInput, EventSummary, JobKind, JobQueue, open_store
from keepout.units import elapsed_seconds

Row = dict[str, object]


def _tables(app: AppTest) -> list[list[Row]]:
    """The rows of each table of events on the page: the upcoming, then the
    past."""
    return [frame.value.to_dict("records") for frame in app.dataframe]


def _rows(app: AppTest) -> list[Row]:
    """The rows of the page's tables of events, in order."""
    return [row for table in _tables(app) for row in table]


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
    (rows,) = _tables(app)
    assert len(rows) == 7
    for row, summary in zip(rows, summaries, strict=True):
        assert row["Event"] == f"event?event={summary.event_id}"
        assert row["Objects"] == f"{summary.object1} / {summary.object2}"
        assert row["TCA (UTC)"] == str(summary.tca.utc.isot)[:19]
        assert elapsed_seconds(as_of, summary.tca) >= 0
        assert row["Until TCA"] == text.countdown(summary.tca, as_of)
        assert summary.latest is not None
        assert row["Latest Pc"] == pc_text(summary.latest.pc)
        assert summary.provenance is not None
        assert row["Provenance"] == PROVENANCE_LABELS[summary.provenance]
        assert row["Risk"] == text.assessment_text(summary.assessment)
        assert row["CDMs"] == summary.cdm_count
        assert row["Manoeuvre"] == f"planner?event={summary.event_id}"
    assert [caption.value for caption in app.caption] == ["Events 1 to 7 of 7"]


def test_a_pc_has_two_exponent_digits(app_test: OpenApp) -> None:
    pcs = [str(row["Latest Pc"]) for row in _rows(app_test())]
    assert pcs
    assert all(len(pc.split("e")[1]) == 3 for pc in pcs)


def test_risk_shown_as_words_with_a_coloured_mark(app_test: OpenApp) -> None:
    risks = {row["Risk"] for row in _rows(app_test())}
    words = {text.risk_text(level, action) for level in RiskLevel for action in Action}
    assert risks
    assert risks <= words | {"not computed"}


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
    assert {(row["Latest Pc"], row["Provenance"], row["Risk"]) for row in rows} == {
        ("not computed", None, "not computed")
    }


def test_past_events_are_shown_on_the_toggle(
    app_test: OpenApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KEEPOUT_AS_OF", "2027-01-01T00:00:00")
    app = app_test()
    assert not app.exception
    assert _rows(app) == []
    assert [item.value for item in app.info] == [
        (
            "No upcoming events as of 2027-01-01T00:00:00 UTC: turn on Show "
            "past events, or start the dashboard with an earlier --as-of."
        )
    ]
    app.toggle(key="events-past").set_value(True).run()
    assert not app.exception
    (rows,) = _tables(app)
    assert len(rows) == 12
    assert all(str(row["Until TCA"]).endswith(" ago") for row in rows)
    assert [item.value for item in app.info] == []


def test_past_events_are_apart_and_have_no_planner(app_test: OpenApp) -> None:
    app = app_test()
    app.toggle(key="events-past").set_value(True).run()
    assert not app.exception
    upcoming, past = _tables(app)
    assert len(upcoming) == 7
    assert len(past) == 5
    assert all("Manoeuvre" in row for row in upcoming)
    assert not any("Manoeuvre" in row for row in past)
    assert "**Past events**" in [item.value for item in app.markdown]


def _many(path: Path, count: int) -> Path:
    """A store of ``count`` events: the CCSDS example against ``count``
    secondaries, in 2010, without Pc."""
    messages = []
    for i in range(count):
        made = B1_TEXT.replace("= 30337", f"= {40000 + i}")
        made = set_keyword(made, "MESSAGE_ID", f"MANY-{i}")
        messages.append(CdmInput.of(keepout.cdm.loads(made), origin="test"))
    with open_store(path) as store:
        assert len(store.ingest(messages).new_events) == count
    return path


def test_the_list_shows_25_events_a_page(
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
    first = [row["Event"] for row in _rows(app)]

    app.button(key="events-next").click().run()
    assert not app.exception
    second = [row["Event"] for row in _rows(app)]
    assert len(second) == 5
    assert not set(first) & set(second)
    assert [caption.value for caption in app.caption] == ["Events 26 to 30 of 30"]
    assert app.button(key="events-next").disabled

    app.button(key="events-previous").click().run()
    assert [row["Event"] for row in _rows(app)] == first


def test_the_event_and_planner_links_are_link_columns(app_test: OpenApp) -> None:
    app = app_test()
    columns = json.loads(app.dataframe[0].proto.columns)
    assert columns["Event"]["type_config"]["type"] == "link"
    assert columns["Manoeuvre"]["type_config"] == {
        "type": "link",
        "display_text": "Planner",
    }


def test_the_upload_starts_collapsed(app_test: OpenApp) -> None:
    (expander,) = app_test().expander
    assert expander.label == "Upload CDMs"
    assert not expander.proto.expanded


def test_adding_no_file_asks_for_one(app_test: OpenApp) -> None:
    app = app_test()
    app.button(key="FormSubmitter:upload-Add to store").click().run()
    assert not app.exception
    assert [item.value for item in app.warning] == ["Choose CDM files to add."]


def _upload(app: AppTest, *files: tuple[str, bytes]) -> AppTest:
    """The app after ``files`` are uploaded and ingested."""
    app.file_uploader[0].set_value([(name, data, "text/plain") for name, data in files])
    app.button(key="FormSubmitter:upload-Add to store").click()
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
        "Added 1 CDM (0 already stored) in 1 event; queued 1 Pc refresh job."
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
