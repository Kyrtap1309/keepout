"""The dashboard's event page: an event's Pc trend, encounter plane and
geometry at TCA, each chart followed by its data, every CDM's Pc with its
provenance, and a refresh job for the Pc not computed."""

import base64
import json
from pathlib import Path
from typing import Any

import astropy.units as u
import dashboard_seed as seed
import numpy as np
import numpy.typing as npt
import pytest
from astropy.time import Time
from builders import EPOCH, LEO_R, LEO_V, cached_pc
from dashboard_builders import OpenApp, shown, use_store, walk
from sqlalchemy import delete
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import Element, PageLink

from keepout._text import pc_text
from keepout.cdm import Cdm, CovarianceMethod, ObjectMetadata, make_cdm
from keepout.conjunction import Conjunction
from keepout.covariance import Covariance
from keepout.dashboard.text import AFTER_TCA, literal, risk_text
from keepout.figures import PROVENANCE_LABELS
from keepout.objects import ObjectState
from keepout.risk import DEFAULT_DECISION_CONFIG, PC_FLOOR, Provenance, RiskLevel
from keepout.settings import Settings, load_settings
from keepout.state import StateVector
from keepout.store import CdmInput, JobKind, JobQueue, models, open_store
from keepout.store.codec import result_key
from keepout.units import HOUR

CRAFTED = "![](http://host/x.png)"
"""Text that Markdown would show as an image, fetched from a host."""

SECTIONS = ("Pc trend", "Encounter plane", "Geometry at TCA")
"""The headings of the page's charts, in order."""

CALCULATED = CovarianceMethod.CALCULATED


def _cdm_rows(app: AppTest) -> dict[str, dict[str, object]]:
    """The rows of the CDMs table, by ``MESSAGE_ID``."""
    (table,) = [frame.value for frame in app.dataframe if "CDM" in frame.value]
    # A missing value is None, as the page gives it (pandas makes it NaN).
    rows = table.astype(object).where(table.notna(), None).to_dict("records")
    return {str(row["CDM"]): row for row in rows}


def _charts(app: AppTest) -> list[dict[str, Any]]:
    """The Plotly figures of the page, as dicts."""
    charts: list[dict[str, Any]] = []
    for node in app.get("plotly_chart"):
        assert isinstance(node, Element)
        charts.append(json.loads(node.proto.spec))
    return charts


def _links_to_the_list(app: AppTest) -> None:
    """Check the page's one link: to the events list, the app's default page
    (an empty path)."""
    (link,) = app.get("page_link")
    assert isinstance(link, PageLink)
    assert link.proto.label == "Events"
    assert link.proto.page == ""
    assert link.proto.page_script_hash


def _values(data: object) -> list[float]:
    """An array of a figure's JSON: a list, or Plotly's typed array (base64
    bytes and their dtype)."""
    if isinstance(data, dict):
        raw = base64.b64decode(data["bdata"])
        return [float(v) for v in np.frombuffer(raw, dtype=np.dtype(data["dtype"]))]
    assert isinstance(data, list)
    return [float(value) for value in data]


def _seed_event(seed_db: Path) -> str:
    """The first upcoming event of the seed."""
    with open_store(seed_db, create=False) as store:
        page = store.summaries(load_settings(seed.SETTINGS), as_of=seed.seed_as_of())
    return page.items[0].event_id


def _forget_pc(path: Path, message_id: str) -> None:
    """Drop a CDM's Pc cached for the seed's settings."""
    with open_store(path, create=False) as store, store.write() as session:
        session.execute(
            delete(models.Result).where(
                models.Result.message_id == message_id,
                models.Result.key == result_key(load_settings(seed.SETTINGS)),
            )
        )


def _object(
    object_id: str,
    r: npt.ArrayLike,
    v: npt.ArrayLike,
    covariance: npt.ArrayLike | None,
    radius: float | None = 2.5,
) -> ObjectState:
    """An object at :data:`builders.EPOCH`, its position covariance [m^2]
    (a small velocity one with it) or none, and its hard-body radius."""
    full = None
    if covariance is not None:
        matrix = np.zeros((6, 6))
        matrix[:3, :3] = covariance
        matrix[3:, 3:] = 1e-4 * np.eye(3)
        full = Covariance(matrix, "GCRF", EPOCH)
    return ObjectState(object_id, StateVector(EPOCH, r, v, "GCRF"), full, radius)


def _crossing(
    miss: float,
    hours: float,
    message_id: str,
    *,
    covariance: bool = True,
    radius: float | None = 2.5,
    relative: npt.ArrayLike | None = None,
) -> Cdm:
    """A LEO head-on crossing ``miss`` m apart along x, issued ``hours``
    before TCA; without covariances or hard-body radii if asked, or with
    the secondary's velocity ``relative`` to the primary's."""
    velocity = -LEO_V if relative is None else LEO_V + np.asarray(relative)
    conjunction = Conjunction(
        _object(
            "PRI",
            LEO_R,
            LEO_V,
            np.diag([100.0, 100.0, 400.0]) if covariance else None,
            radius,
        ),
        _object(
            "SEC",
            LEO_R + np.array([miss, 0.0, 0.0]),
            velocity,
            np.eye(3) if covariance else None,
            radius,
        ),
        5.0,
    )
    method = CALCULATED if covariance else CovarianceMethod.DEFAULT
    return make_cdm(
        conjunction,
        ObjectMetadata("PRI", "2020-001A", covariance_method=method),
        ObjectMetadata("SEC", "2019-001A", covariance_method=method),
        created=EPOCH - hours * HOUR * u.s,
        message_id=message_id,
    )


def _store(path: Path, cdms: list[Cdm], settings: Settings | None) -> str:
    """A store of one event of ``cdms``, its Pc computed with ``settings``
    unless ``None``; the event's id."""
    with open_store(path) as store:
        (event_id,) = store.ingest(
            [CdmInput.of(cdm, origin="test") for cdm in cdms]
        ).events
        if settings is not None:
            store.refresh(settings)
    return event_id


@pytest.fixture
def own_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """The store the test makes, seen early in 2010 with the default
    settings."""
    path = tmp_path / "own.db"
    use_store(monkeypatch, path, as_of=Time("2010-03-01T00:00:00", scale="utc"))
    return path


def test_no_event_param(app_test: OpenApp) -> None:
    app = app_test("event")
    assert not app.exception
    assert not app.error
    assert [item.value for item in app.info] == ["Pick an event from the list."]
    _links_to_the_list(app)
    assert app.get("plotly_chart") == []


def test_an_unknown_event_is_a_warning_with_the_way_back(app_test: OpenApp) -> None:
    app = app_test("event", query={"event": "nope"})
    assert not app.exception
    assert not app.error
    assert app.query_params["event"] == "nope"
    assert [item.value for item in app.title] == ["Event"]
    assert [item.value for item in app.warning] == ["No such event: nope"]
    _links_to_the_list(app)
    assert app.get("plotly_chart") == []


def test_unknown_event_named_verbatim(app_test: OpenApp) -> None:
    app = app_test("event", query={"event": CRAFTED})
    assert not app.exception
    assert [item.value for item in app.warning] == [
        f"No such event: {literal(CRAFTED)}"
    ]
    _links_to_the_list(app)


def test_event_page_sections(app_test: OpenApp, seed_db: Path) -> None:
    event_id = _seed_event(seed_db)
    app = app_test("event", query={"event": event_id})
    assert not app.exception
    assert not app.error
    assert [item.value for item in app.title] == [f"Event {event_id}"]
    # Its links back to the list and on to the event's manoeuvre planner.
    back, on = app.get("page_link")
    assert isinstance(back, PageLink)
    assert isinstance(on, PageLink)
    assert back.proto.label == "Events"
    assert (on.proto.label, on.proto.page) == ("Manoeuvre planner", "planner")
    assert on.proto.query_string == f"event={event_id}"
    headings = [item.value for item in app.subheader]
    assert [h for h in headings if h in SECTIONS] == list(SECTIONS)
    assert len(app.get("plotly_chart")) == 3
    # Every chart is followed by its data table (the trend's: the CDMs).
    kinds = [node.type for node in walk(app.main)]
    charts = [i for i, kind in enumerate(kinds) if kind == "plotly_chart"]
    assert len(charts) == 3
    for index in charts:
        after = [*kinds[index + 1 :], "plotly_chart"]
        assert after.index("dataframe") < after.index("plotly_chart")
    assert "Data as of 2026-10-07T12:00:00 UTC (fixed)" in shown(app)


def test_event_page_badges(app_test: OpenApp, seed_db: Path) -> None:
    event_id = _seed_event(seed_db)
    with open_store(seed_db, create=False) as store:
        event = store.event(event_id)
    app = app_test("event", query={"event": event_id})
    assert not app.exception
    rows = _cdm_rows(app)
    assert len(event.cdms) > 1
    assert list(rows) == [cdm.message_id for cdm in event.cdms]
    for row in rows.values():
        assert row["Provenance"] == PROVENANCE_LABELS[Provenance.GP_AGE_MODEL]


def test_missing_pc_offers_compute(
    app_test: OpenApp, seed_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    event_id = _seed_event(seed_copy)
    settings = load_settings(seed.SETTINGS)
    with open_store(seed_copy, create=False) as store:
        cdms = [cdm.message_id for cdm in store.event(event_id).cdms]
        cached = cached_pc(store, event_id, settings)
    first = cdms[0]
    _forget_pc(seed_copy, first)
    monkeypatch.setenv("KEEPOUT_STORE", str(seed_copy))
    app = app_test("event", query={"event": event_id})
    assert not app.exception
    assert not app.error
    text = shown(app)
    assert f"Pc not computed for 1 of {len(cdms)} CDMs ({first})" in text
    # Each CDM's row: its Pc, its provenance badge and its risk, which
    # waits for every Pc; the cached Pc are shown.
    rows = _cdm_rows(app)
    assert list(rows) == cdms
    shown_pc = [(row["Pc"], row["Provenance"], row["Risk"]) for row in rows.values()]
    assert shown_pc[0] == ("not computed", None, None)
    origin = PROVENANCE_LABELS[Provenance.GP_AGE_MODEL]
    assert shown_pc[1:] == [
        (pc_text(observation.pc), origin, None) for observation in cached[1:]
    ]
    # The latest Pc, as the events list shows it; no risk yet.
    assert f"Latest Pc: {pc_text(cached[-1].pc)}" in text.splitlines()
    assert "not computed</span>" in text
    # No trend without every Pc, but the geometry still shows.
    assert len(app.get("plotly_chart")) == 2
    compute = app.button(key=f"event-compute-{event_id}")
    assert compute.label == "Compute"

    compute.click().run()
    assert not app.exception
    assert not app.error
    with open_store(seed_copy, create=False) as store:
        (job,) = JobQueue(store).jobs()
    assert job.kind is JobKind.REFRESH
    assert job.event_id == event_id
    assert f"Job {job.id} (refresh): **queued**" in shown(app)

    app.button(key=f"event-compute-{event_id}").click().run()
    assert not app.exception
    with open_store(seed_copy, create=False) as store:
        assert [item.id for item in JobQueue(store).jobs()] == [job.id]
    assert shown(app).count(f"Job {job.id} (refresh): **queued**") == 1

    # A new session (a reload) finds the job in the store.
    again = app_test("event", query={"event": event_id})
    assert not again.exception
    assert shown(again).count(f"Job {job.id} (refresh): **queued**") == 1


@pytest.mark.usefixtures("worker_first")
def test_a_refused_cancel_is_a_toast_left_on_its_page(
    app_test: OpenApp, seed_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A refresh job queued elsewhere (keepout CLI, another session), which a
    # worker cancels between the page's read and the click: the page then
    # no longer lists a job, and the refusal must not wait for another page.
    event_id = _seed_event(seed_copy)
    with open_store(seed_copy, create=False) as store:
        _forget_pc(seed_copy, store.event(event_id).cdms[0].message_id)
        job = JobQueue(store).submit_refresh(event_id, load_settings(seed.SETTINGS))
    assert job.job is not None
    job_id = job.job.id
    monkeypatch.setenv("KEEPOUT_STORE", str(seed_copy))
    app = app_test("event", query={"event": event_id})
    app.button(key=f"cancel-job-{job_id}").click().run()
    assert not app.exception
    assert [item.value for item in app.toast] == [
        f"job {job_id} has already ended: cancelled"
    ]
    assert not app.error
    assert f"Job {job_id} " not in shown(app)
    # Nothing of it is left for the next page.
    app.switch_page("views/planner.py")
    app.query_params.update({"event": event_id})
    app.run()
    assert not app.exception
    assert not app.error
    assert not app.toast


def test_trend_of_a_changing_pc(app_test: OpenApp, own_store: Path) -> None:
    cdms = [
        _crossing(200.0, 72.0, "T0"),
        _crossing(40.0, 48.0, "T1"),
        _crossing(15.0, 24.0, "T2"),
    ]
    event_id = _store(own_store, cdms, Settings())
    app = app_test("event", query={"event": event_id})
    assert not app.exception
    assert not app.error
    trend = _charts(app)[0]
    points = [t for t in trend["data"] if t["mode"] == "markers"]
    hours = sorted(x for t in points for x in _values(t["x"]))
    assert hours == pytest.approx([24.0, 48.0, 72.0])
    pc = [y for t in points for y in _values(t["y"])]
    assert len(set(pc)) == 3
    # The decision point the assessment uses, marked on the trend.
    vertical = [s for s in trend["layout"]["shapes"] if s["x0"] == s["x1"]]
    expected = DEFAULT_DECISION_CONFIG.decision_point / HOUR
    assert [s["x0"] for s in vertical] == [pytest.approx(expected)]
    # The trend's data: the CDMs table, with each Pc the store has; the
    # trend draws a Pc below the floor at the floor.
    with open_store(own_store, create=False) as store:
        cached = [item.pc for item in cached_pc(store, event_id, Settings())]
    assert all(value is not None for value in cached)
    assert sorted(pc) == [max(value or 0.0, PC_FLOOR) for value in cached]
    rows = _cdm_rows(app)
    assert list(rows) == ["T0", "T1", "T2"]
    assert [row["Pc"] for row in rows.values()] == [pc_text(v) for v in cached]
    assert {row["Provenance"] for row in rows.values()} == {
        PROVENANCE_LABELS[Provenance.CDM_COVARIANCE]
    }
    # The latest Pc is red, in a coloured mark and words.
    assert rows["T2"]["Risk"] == risk_text(RiskLevel.RED)


def test_max_pc_event_has_no_ellipse(app_test: OpenApp, own_store: Path) -> None:
    event_id = _store(
        own_store, [_crossing(30.0, 24.0, "MAX", covariance=False)], Settings()
    )
    app = app_test("event", query={"event": event_id})
    assert not app.exception
    assert not app.error
    text = shown(app)
    assert "No covariance: the Pc shown is an upper bound (maximum Pc)." in text
    assert PROVENANCE_LABELS[Provenance.MAX_PC] in text
    _, encounter, geometry = _charts(app)
    names = [trace["name"] for trace in encounter["data"]]
    assert not [name for name in names if "sigma" in name]
    assert "hard-body radius" in names
    assert not [t for t in geometry["data"] if "ellipsoid" in t["name"]]


def test_encounter_ellipses_with_a_covariance(
    app_test: OpenApp, own_store: Path
) -> None:
    event_id = _store(own_store, [_crossing(30.0, 24.0, "C")], Settings())
    app = app_test("event", query={"event": event_id})
    assert not app.exception
    _, encounter, geometry = _charts(app)
    names = [trace["name"] for trace in encounter["data"]]
    assert names[:3] == ["1 sigma", "2 sigma", "3 sigma"]
    assert any("ellipsoid" in t["name"] for t in geometry["data"])
    assert "no covariance" not in shown(app)


def test_missing_hard_body_radius(app_test: OpenApp, own_store: Path) -> None:
    event_id = _store(own_store, [_crossing(30.0, 24.0, "N", radius=None)], None)
    app = app_test("event", query={"event": event_id})
    assert not app.exception
    assert not app.error
    assert app.get("plotly_chart") == []
    notes = [item.value for item in app.warning]
    expected = (
        "No hard-body radius: CDM N gives no HBR for OBJECT1 and OBJECT2, "
        "and the settings give none (hbr_m)."
    )
    assert notes.count(expected) == 1
    assert "not computed" in shown(app)


def test_zero_relative_speed(app_test: OpenApp, own_store: Path) -> None:
    cdm = _crossing(30.0, 24.0, "Z", relative=[0.0, 0.0, 0.0])
    event_id = _store(own_store, [cdm], Settings())
    app = app_test("event", query={"event": event_id})
    assert not app.exception
    assert not app.error
    # The Pc is a 3-D one; only the encounter plane is undefined.
    assert len(app.get("plotly_chart")) == 1
    (note,) = (item.value for item in app.warning)
    assert note.startswith(
        "No encounter plane or geometry at TCA: relative velocity is zero"
    )


def test_settings_hard_body_radius_wins(
    app_test: OpenApp,
    own_store: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The CDM gives 2.5 m for each object (5 m combined); the settings 12 m.
    settings = tmp_path / "settings.toml"
    settings.write_text("[pc]\nhbr_m = 12.0\n", encoding="utf-8")
    monkeypatch.setenv("KEEPOUT_SETTINGS", str(settings))
    event_id = _store(own_store, [_crossing(30.0, 24.0, "H")], load_settings(settings))
    app = app_test("event", query={"event": event_id})
    assert not app.exception
    assert not app.error
    table = app.dataframe[1].value
    radius = dict(zip(table["quantity"], table["value"], strict=True))
    assert radius["hard-body radius [m]"] == 12.0
    _, encounter, _ = _charts(app)
    (circle,) = [t for t in encounter["data"] if t["name"] == "hard-body radius"]
    assert max(_values(circle["x"])) == pytest.approx(12.0)


def test_unrated_event_says_why(app_test: OpenApp, own_store: Path) -> None:
    # Issued an hour after TCA: assessed, but no level.
    event_id = _store(own_store, [_crossing(30.0, -1.0, "LATE")], Settings())
    app = app_test("event", query={"event": event_id})
    assert not app.exception
    assert not app.error
    text = shown(app)
    assert f"{AFTER_TCA}</span>" in text
    assert "not computed</span>" not in text
    assert [item.value for item in app.info] == ["No Pc before TCA to draw."]


def test_cdm_text_shown_verbatim(app_test: OpenApp, own_store: Path) -> None:
    # A MESSAGE_ID is text from the CDM: shown as it is, never as Markdown.
    event_id = _store(own_store, [_crossing(30.0, 24.0, CRAFTED)], None)
    app = app_test("event", query={"event": event_id})
    assert not app.exception
    text = shown(app)
    assert literal(CRAFTED) in text
    assert CRAFTED not in text.replace(literal(CRAFTED), "")


def _red(own_store: Path) -> str:
    """An event of three CDMs whose latest Pc is red."""
    cdms = [
        _crossing(200.0, 72.0, "T0"),
        _crossing(40.0, 48.0, "T1"),
        _crossing(15.0, 24.0, "T2"),
    ]
    return _store(own_store, cdms, Settings())


def test_the_reasons_for_the_risk_are_shown_as_text(
    app_test: OpenApp, own_store: Path
) -> None:
    event_id = _red(own_store)
    app = app_test("event", query={"event": event_id})
    assert not app.exception
    (why,) = [item.value for item in app.info]
    head, reasons = why.split("\n\n")
    assert head.startswith("**red · ")
    assert "maneuver" not in head
    assert reasons.startswith("- latest Pc ")
    # Not a grey caption.
    assert all("latest Pc" not in item.value for item in app.caption)


def test_a_red_event_offers_planning_as_its_primary_action(
    app_test: OpenApp, own_store: Path
) -> None:
    event_id = _red(own_store)
    app = app_test("event", query={"event": event_id})
    plan = app.button(key=f"event-plan-{event_id}")
    assert plan.label == "Manoeuvre planner"
    assert plan.proto.type == "primary"
    assert [link.proto.page for link in app.get("page_link")] == [""]
    plan.click().run()
    assert not app.exception
    assert [item.value for item in app.title] == [
        f"Manoeuvre planner: event {event_id}"
    ]


def test_the_numbers_of_the_geometry_have_fixed_decimals(
    app_test: OpenApp, own_store: Path
) -> None:
    event_id = _red(own_store)
    app = app_test("event", query={"event": event_id})
    _, encounter, geometry = (
        json.loads(frame.proto.columns) for frame in app.dataframe
    )
    assert encounter["value"]["type_config"]["format"] == "%.1f"
    assert geometry["secondary at TCA [m]"]["type_config"]["format"] == "%.1f"
