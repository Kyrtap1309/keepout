"""The dashboard's event page: an event's Pc trend, encounter plane and
geometry at TCA, each chart followed by its data, every CDM's Pc with its
provenance, and a refresh job for the Pc not computed."""

import base64
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import astropy.units as u
import dashboard_seed as seed
import numpy as np
import numpy.typing as npt
import pytest
from builders import EPOCH, LEO_R, LEO_V
from dashboard_builders import OpenApp
from sqlalchemy import delete
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import Block, Element, Node, PageLink

from keepout._text import pc_text
from keepout.cdm import Cdm, CovarianceMethod, ObjectMetadata, make_cdm
from keepout.conjunction import Conjunction
from keepout.covariance import Covariance
from keepout.dashboard.text import AFTER_TCA, literal
from keepout.figures import Provenance
from keepout.objects import ObjectState
from keepout.risk import DEFAULT_DECISION_CONFIG
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


def _walk(node: Node) -> Iterator[Node]:
    """The elements under ``node``, depth first, in the page's order."""
    if isinstance(node, Block):
        for child in node.children.values():
            yield from _walk(child)
    else:
        yield node


def _elements(app: AppTest) -> list[Node]:
    """The elements of the page's main area, in order."""
    return list(_walk(app.main))


def _shown(app: AppTest) -> str:
    """Every text of the page's main area, one per line."""
    return "\n".join(str(getattr(node, "value", "")) for node in _elements(app))


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
        return np.frombuffer(raw, dtype=np.dtype(data["dtype"])).tolist()
    assert isinstance(data, list)
    return [float(value) for value in data]


def _seed_event(seed_db: Path) -> str:
    """The first upcoming event of the seed."""
    with open_store(seed_db, create=False) as store:
        page = store.summaries(load_settings(seed.SETTINGS), as_of=seed.seed_as_of())
    return page.items[0].event_id


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
    monkeypatch.setenv("KEEPOUT_STORE", str(path))
    monkeypatch.delenv("KEEPOUT_SETTINGS")
    monkeypatch.setenv("KEEPOUT_AS_OF", "2010-03-01T00:00:00")
    return path


def test_no_event_param(app_test: OpenApp) -> None:
    app = app_test("event")
    assert not app.exception
    assert not app.error
    assert [item.value for item in app.info] == ["Pick an event from the list."]
    _links_to_the_list(app)
    assert app.get("plotly_chart") == []


def test_unknown_event(app_test: OpenApp) -> None:
    app = app_test("event", query={"event": "nope"})
    assert not app.exception
    assert not app.error
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
    # Its link on to the event's manoeuvre planner.
    (link,) = app.get("page_link")
    assert isinstance(link, PageLink)
    assert (link.proto.label, link.proto.page) == ("Manoeuvre planner", "planner")
    assert link.proto.query_string == f"event={event_id}"
    headings = [item.value for item in app.subheader]
    assert [h for h in headings if h in SECTIONS] == list(SECTIONS)
    assert len(app.get("plotly_chart")) == 3
    # Every chart is followed by its data table.
    kinds = [node.type for node in _elements(app)]
    charts = [i for i, kind in enumerate(kinds) if kind == "plotly_chart"]
    assert len(charts) == 3
    for index in charts:
        assert kinds[index + 1] == "dataframe"
    assert "Data as of 2026-10-07T12:00:00 UTC (fixed)" in _shown(app)


def test_event_page_badges(app_test: OpenApp, seed_db: Path) -> None:
    event_id = _seed_event(seed_db)
    with open_store(seed_db, create=False) as store:
        event = store.event(event_id)
    app = app_test("event", query={"event": event_id})
    assert not app.exception
    shown = _shown(app)
    assert len(event.cdms) > 1
    for cdm in event.cdms:
        row = shown.split(cdm.message_id, 1)[1].split("\n")[:12]
        assert any(Provenance.GP_AGE_MODEL.value in line for line in row), cdm
    # The trend's table names each point's provenance too.
    trend = app.dataframe[0].value
    assert set(trend["provenance"]) == {Provenance.GP_AGE_MODEL.value}


def test_missing_pc_offers_compute(
    app_test: OpenApp, seed_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    event_id = _seed_event(seed_copy)
    settings = load_settings(seed.SETTINGS)
    with open_store(seed_copy, create=False) as store:
        cdms = [cdm.message_id for cdm in store.event(event_id).cdms]
        cached = store.observations(event_id, settings)
        first = cdms[0]
        with store.write() as session:
            session.execute(
                delete(models.Result).where(
                    models.Result.message_id == first,
                    models.Result.key == result_key(settings),
                )
            )
    monkeypatch.setenv("KEEPOUT_STORE", str(seed_copy))
    app = app_test("event", query={"event": event_id})
    assert not app.exception
    assert not app.error
    shown = _shown(app)
    assert f"Pc not computed for 1 of {len(cdms)} CDMs ({first})" in shown
    # Each CDM's row: its Pc, its provenance badge and its risk, which
    # waits for every Pc; the cached Pc are shown.
    lines = shown.splitlines()
    rows = [lines[lines.index(message_id) + 4 :][:3] for message_id in cdms]
    assert rows[0][0] == "not computed"
    badge = f":orange-badge[{Provenance.GP_AGE_MODEL.value}]"
    for row, observation in zip(rows[1:], cached[1:], strict=True):
        assert row == [pc_text(observation.pc), badge, "-"]
    # The latest Pc, as the events list shows it; no risk yet.
    assert f"Latest Pc: {pc_text(cached[-1].pc)}" in lines
    assert "not computed</span>" in shown
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
    assert f"Job {job.id} (refresh): **queued**" in _shown(app)

    app.button(key=f"event-compute-{event_id}").click().run()
    assert not app.exception
    with open_store(seed_copy, create=False) as store:
        assert [item.id for item in JobQueue(store).jobs()] == [job.id]
    assert _shown(app).count(f"Job {job.id} (refresh): **queued**") == 1

    # A new session (a reload) finds the job in the store.
    again = app_test("event", query={"event": event_id})
    assert not again.exception
    assert _shown(again).count(f"Job {job.id} (refresh): **queued**") == 1


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
    table = app.dataframe[0].value
    assert list(table["CDM"]) == ["T0", "T1", "T2"]
    assert list(table["Pc"]) == pytest.approx(sorted(pc))
    assert set(table["provenance"]) == {Provenance.CDM_COVARIANCE.value}
    # The latest Pc is red, in colour and words.
    assert "red" in _shown(app)


def test_max_pc_event_has_no_ellipse(app_test: OpenApp, own_store: Path) -> None:
    event_id = _store(
        own_store, [_crossing(30.0, 24.0, "MAX", covariance=False)], Settings()
    )
    app = app_test("event", query={"event": event_id})
    assert not app.exception
    assert not app.error
    shown = _shown(app)
    assert "no covariance: max-Pc" in shown
    assert Provenance.MAX_PC.value in shown
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
    assert "no covariance" not in _shown(app)


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
    assert "not computed" in _shown(app)


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
    shown = _shown(app)
    assert f"{AFTER_TCA}</span>" in shown
    assert "not computed</span>" not in shown
    assert [item.value for item in app.info] == ["No Pc before TCA to draw."]


def test_cdm_text_shown_verbatim(app_test: OpenApp, own_store: Path) -> None:
    # A MESSAGE_ID is text from the CDM: shown as it is, never as Markdown.
    event_id = _store(own_store, [_crossing(30.0, 24.0, CRAFTED)], None)
    app = app_test("event", query={"event": event_id})
    assert not app.exception
    shown = _shown(app)
    assert literal(CRAFTED) in shown
    assert CRAFTED not in shown.replace(literal(CRAFTED), "")
