"""The dashboard's manoeuvre planner page: an event's cached plan (the Pc
grid, the smallest burn each way and its re-screen), the job that plans it,
what-if jobs on another target Pc or hard-body radius, and their status."""

import dataclasses
import json
import shutil
from pathlib import Path
from typing import NamedTuple

import astropy.units as u
import dashboard_seed as seed
import numpy as np
import pytest
from astropy.time import Time
from builders import LEO_R, LEO_V, SMALL_GRID
from cli_builders import operator_kvn
from dashboard_builders import open_app, shown, use_store
from sqlalchemy import update
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import PageLink

import keepout.cdm
from keepout import figures
from keepout._text import ASSUMED_COVARIANCES
from keepout.dashboard.text import merged_notes, risk_text
from keepout.errors import KeepoutError
from keepout.figures import GridFigure
from keepout.maneuver import ManeuverOutcome, NewApproach, RescreenResult
from keepout.maneuver.wording import NO_BURN_NEEDED
from keepout.risk import RiskLevel
from keepout.screening import CloseApproach
from keepout.settings import Settings, load_settings
from keepout.state import StateVector
from keepout.store import (
    CdmInput,
    Job,
    JobKind,
    JobQueue,
    JobStatus,
    models,
    open_store,
)
from keepout.store.maneuver_codec import outcome_to_json
from keepout.store.planning import run_maneuver_job

SETTINGS = """\
[pc]
method = "foster"
policy = "repair"

[risk]
green_below = 1e-7
red_from = 1e-4

[maneuver]
target_pc = 1e-5
"""
"""The seed's settings with Foster's method: the small grid plans in a
fraction of a second."""

REFUSED_TARGET = 1e-12
"""A what-if target no burn of the small grid meets with
:data:`REFUSED_HBR`."""

REFUSED_HBR = 2e5
"""A what-if hard-body radius [m] so large that the event's Pc is 1."""


class Planned(NamedTuple):
    """A copy of the seed with the synthetic operator's crossing, planned
    for :data:`SETTINGS` and for the refused what-if."""

    store: Path
    settings: Path
    event_id: str


def _refused(settings: Settings) -> Settings:
    return dataclasses.replace(
        settings, target_pc=REFUSED_TARGET, combined_hard_body_radius=REFUSED_HBR
    )


def _plan(store_path: Path, event_id: str, settings: Settings) -> None:
    """Cache the event's plan on the small grid, as a worker would."""
    with open_store(store_path) as store:
        queue = JobQueue(store)
        submitted = queue.submit(event_id, settings).job
        job = queue.claim("test")
        assert submitted is not None
        assert job is not None
        assert job.id == submitted.id
        payload = run_maneuver_job(str(store_path), job, 1, grid=SMALL_GRID)
        assert queue.finish(job, payload)


@pytest.fixture(scope="module")
def planned(seed_db: Path, tmp_path_factory: pytest.TempPathFactory) -> Planned:
    folder = tmp_path_factory.mktemp("planned")
    path = Path(shutil.copy2(seed_db, folder / "planned.db"))
    settings_file = folder / "settings.toml"
    settings_file.write_text(SETTINGS)
    settings = load_settings(settings_file)
    crossing = keepout.cdm.loads(operator_kvn()[0])
    with open_store(path) as store:
        (event_id,) = store.ingest([CdmInput.of(crossing, "test")]).events
    _plan(path, event_id, settings)
    _plan(path, event_id, _refused(settings))
    return Planned(path, settings_file, event_id)


@pytest.fixture
def store_path(
    planned: Planned, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """A copy of :func:`planned` the test may write to, which the app opens."""
    path = Path(shutil.copy2(planned.store, tmp_path / "planned.db"))
    use_store(monkeypatch, path, planned.settings, seed.seed_as_of())
    return path


def _unplanned(path: Path) -> str:
    """An upcoming event of the seed, without a cached plan."""
    with open_store(path, create=False) as opened:
        page = opened.summaries(load_settings(seed.SETTINGS), as_of=seed.seed_as_of())
    return page.items[0].event_id


def _jobs(path: Path, event_id: str) -> list[Job]:
    with open_store(path, create=False) as opened:
        return [
            job
            for job in JobQueue(opened).jobs()
            if job.event_id == event_id and job.kind is JobKind.MANEUVER
        ]


def _queue(app: AppTest, event_id: str) -> Job:
    """Click the page's Plan manoeuvre button; the one job it queued."""
    store_path = Path(app.session_state["context"].store_path)
    app.button(key=f"planner-plan-{event_id}").click().run()
    assert not app.exception
    (job,) = _jobs(store_path, event_id)
    return job


def _what_if(
    app: AppTest, event_id: str, target: float | str, hbr: float | None
) -> None:
    """Submit the what-if form."""
    app.text_input(key=f"planner-target-{event_id}").input(
        target if isinstance(target, str) else f"{target:g}"
    )
    app.number_input(key=f"planner-hbr-{event_id}").set_value(hbr)
    app.button(key=f"planner-what-if-{event_id}").click().run()
    assert not app.exception


def test_no_event_param(store_path: Path) -> None:
    app = open_app("planner")
    assert not app.exception
    assert [item.value for item in app.title] == ["Manoeuvre planner"]
    assert "Pick an event" in shown(app)


def test_an_unknown_event_is_a_warning_with_the_way_back(store_path: Path) -> None:
    # As the event page: a warning and the way back to the list.
    app = open_app("planner", query={"event": "nope"})
    assert not app.exception
    assert not app.error
    assert [item.value for item in app.title] == ["Manoeuvre planner"]
    assert [item.value for item in app.warning] == ["No such event: nope"]
    (link,) = app.get("page_link")
    assert isinstance(link, PageLink)
    assert link.proto.label == "Events"
    assert link.proto.page == ""


def test_cached_plan_shown(store_path: Path, planned: Planned) -> None:
    app = open_app("planner", query={"event": planned.event_id})
    assert not app.exception
    assert not app.error
    assert [item.value for item in app.title] == [
        f"Manoeuvre planner: event {planned.event_id}"
    ]
    # The heatmap, followed by its cells; the burns before it.
    assert len(app.get("plotly_chart")) == 1
    burns, cells = (frame.value for frame in app.dataframe)
    assert burns["Option"].tolist() == ["no burn", "prograde", "retrograde"]
    assert burns["Target"].tolist()[1:] == ["met", "met"]
    assert set(cells["direction"]) == {"prograde", "retrograde"}
    assert {"hours before TCA", "|Δv| [m/s]", "Pc"} <= set(cells.columns)
    text = shown(app)
    # Green at the CDM's radii: the burns are shown, none is recommended.
    assert NO_BURN_NEEDED in text
    assert "A burn can be recommended" not in text
    assert "Not re-screened: no catalog stored" in text
    assert "assumed (GP age model)" in text
    assert "target Pc 1.00e-05" in text
    # No job is left for the plan.
    assert all(
        button.key != f"planner-plan-{planned.event_id}" for button in app.button
    )
    assert "Job " not in text


def test_no_plan_offers_queue(store_path: Path) -> None:
    event_id = _unplanned(store_path)
    app = open_app("planner", query={"event": event_id})
    assert not app.exception
    assert not app.get("plotly_chart")
    job = _queue(app, event_id)
    assert job.status is JobStatus.QUEUED
    assert f"Job {job.id} (manoeuvre): **queued**" in shown(app)
    # While it is queued, the page offers no second job.
    assert all(button.key != f"planner-plan-{event_id}" for button in app.button)


def test_a_queued_job_is_found_after_a_reload(
    store_path: Path, planned: Planned
) -> None:
    event_id = _unplanned(store_path)
    with open_store(store_path) as opened:
        job = JobQueue(opened).submit(event_id, load_settings(planned.settings)).job
    assert job is not None
    app = open_app("planner", query={"event": event_id})
    assert not app.exception
    assert f"Job {job.id} (manoeuvre): **queued**" in shown(app)


def test_what_if_dedupes(store_path: Path, planned: Planned) -> None:
    event_id = planned.event_id
    before = {job.id for job in _jobs(store_path, event_id)}
    app = open_app("planner", query={"event": event_id})
    _what_if(app, event_id, 1e-6, None)
    _what_if(app, event_id, 1e-6, None)
    added = [job for job in _jobs(store_path, event_id) if job.id not in before]
    (job,) = added
    assert job.status is JobStatus.QUEUED
    assert job.settings.target_pc == 1e-6
    assert "What-if" in shown(app)
    assert not app.get("plotly_chart")
    # The operator's own values: their plan is cached, no job.
    _what_if(app, event_id, 1e-5, None)
    assert [
        item.id for item in _jobs(store_path, event_id) if item.id not in before
    ] == [job.id]
    assert len(app.get("plotly_chart")) == 1
    assert "What-if" not in shown(app)


def test_a_queued_job_is_cancelled_and_can_be_queued_again(store_path: Path) -> None:
    event_id = _unplanned(store_path)
    app = open_app("planner", query={"event": event_id})
    job = _queue(app, event_id)
    app.button(key=f"cancel-job-{job.id}").click().run()
    assert not app.exception
    assert not app.error
    (cancelled,) = _jobs(store_path, event_id)
    assert cancelled.status is JobStatus.CANCELLED
    assert f"Job {job.id} (manoeuvre): **cancelled**" in shown(app)
    # It can be queued again.
    assert any(button.key == f"planner-plan-{event_id}" for button in app.button)


def test_failed_job_shows_error(store_path: Path, planned: Planned) -> None:
    event_id = _unplanned(store_path)
    with open_store(store_path) as opened:
        queue = JobQueue(opened)
        queue.submit(event_id, load_settings(planned.settings))
        job = queue.claim("test")
        assert job is not None
        assert queue.fail(job, "the burn window closed")
    app = open_app("planner", query={"event": event_id})
    assert not app.exception
    assert [item.value for item in app.error] == ["the burn window closed"]
    assert f"Job {job.id} (manoeuvre): **failed**" in shown(app)


def test_a_refused_what_if_shows_its_refusal_and_queues_nothing(
    store_path: Path, planned: Planned
) -> None:
    event_id = planned.event_id
    before = len(_jobs(store_path, event_id))
    app = open_app("planner", query={"event": event_id})
    _what_if(app, event_id, REFUSED_TARGET, REFUSED_HBR)
    assert len(_jobs(store_path, event_id)) == before
    assert not app.error
    assert [item.value for item in app.warning] == [
        "No burn on the grid meets the target Pc 1.00e-12."
    ]
    burns, _ = (frame.value for frame in app.dataframe)
    assert burns["Option"].tolist() == ["no burn"]
    assert len(app.get("plotly_chart")) == 1


def test_a_bad_what_if_is_an_error(store_path: Path, planned: Planned) -> None:
    event_id = planned.event_id
    app = open_app("planner", query={"event": event_id})
    _what_if(app, event_id, 1e-5, 0.0)
    assert len(app.error) == 1
    assert "combined_hard_body_radius" in str(app.error[0].value)


def test_the_what_if_form_starts_from_the_what_if_shown(
    store_path: Path, planned: Planned
) -> None:
    event_id = planned.event_id
    app = open_app("planner", query={"event": event_id})
    _what_if(app, event_id, 1e-6, 30.0)
    # Away and back: the form's widgets are made anew, the what-if is kept.
    app.switch_page("views/events.py").run()
    app.switch_page("views/planner.py")
    app.query_params.update({"event": event_id})
    app.run()
    assert not app.exception
    assert "**What-if**: target Pc 1.00e-06" in shown(app)
    assert app.text_input(key=f"planner-target-{event_id}").value == repr(1e-6)
    assert app.number_input(key=f"planner-hbr-{event_id}").value == 30.0
    # Changing only the target keeps the what-if's radius.
    app.text_input(key=f"planner-target-{event_id}").input("1e-7")
    app.button(key=f"planner-what-if-{event_id}").click().run()
    assert not app.exception
    (job,) = [
        job
        for job in _jobs(store_path, event_id)
        if job.status.is_active and job.settings.target_pc == 1e-7
    ]
    assert job.settings.combined_hard_body_radius == 30.0


def test_back_to_the_operators_settings(store_path: Path, planned: Planned) -> None:
    event_id = planned.event_id
    app = open_app("planner", query={"event": event_id})
    _what_if(app, event_id, REFUSED_TARGET, REFUSED_HBR)
    assert [item.value for item in app.warning] == [
        "No burn on the grid meets the target Pc 1.00e-12."
    ]
    app.button(key=f"planner-own-{event_id}").click().run()
    assert not app.exception
    assert not app.warning
    assert NO_BURN_NEEDED in shown(app)
    # The form starts from them again.
    assert app.text_input(key=f"planner-target-{event_id}").value == repr(1e-5)
    assert app.number_input(key=f"planner-hbr-{event_id}").value is None


def _approach(secondary: int, tca: Time, pc: float, level: RiskLevel) -> NewApproach:
    """An approach of the primary to ``secondary`` at ``tca``, 300 m apart."""
    primary = StateVector(tca, LEO_R, LEO_V, "GCRF")
    other = StateVector(tca, LEO_R + np.array([0.0, 0.0, 300.0]), -LEO_V, "GCRF")
    return NewApproach(
        approach=CloseApproach(1, secondary, primary, other), pc=pc, level=level
    )


def _rescreen(
    outcome: ManeuverOutcome,
    *,
    new: tuple[NewApproach, ...] = (),
    closer: tuple[NewApproach, ...] = (),
    returns: tuple[NewApproach, ...] = (),
    event_range: float | None = None,
    screened: bool = True,
) -> RescreenResult:
    """A re-screen of the outcome's first burn, as a catalog would give."""
    tca = outcome.event.latest.tca
    return RescreenResult(
        option=outcome.burns[0].option,
        new_approaches=new,
        closer_approaches=closer,
        event_returns=returns,
        event_range=event_range,
        window=(tca - 1 * u.day, tca + 1 * u.day),
        catalog_objects=120,
        secondary_screened=screened,
        skipped_objects=2,
    )


def _outcome(planned: Planned) -> ManeuverOutcome:
    """The plan cached for the planned event."""
    with open_store(planned.store, create=False) as opened:
        outcome = opened.maneuver(planned.event_id, load_settings(planned.settings))
    assert outcome is not None
    return outcome


def _rescreened(
    store_path: Path, planned: Planned, results: tuple[RescreenResult | None, ...]
) -> AppTest:
    """The planner of the planned event, its plan cached in ``store_path`` as if
    re-screened against a catalog with ``results``, a burn each (``None``:
    not re-screened); a real re-screen takes tens of seconds."""
    outcome = _outcome(planned)
    assert len(outcome.burns) == len(results)
    rescreened = dataclasses.replace(
        outcome,
        catalog_label="the test catalog",
        burns=tuple(
            dataclasses.replace(burn, rescreen=result)
            for burn, result in zip(outcome.burns, results, strict=True)
        ),
    )
    with open_store(store_path, create=False) as opened:
        detail = opened.plan_detail(planned.event_id, load_settings(planned.settings))
        assert detail is not None
        with opened.write() as session:
            session.execute(
                update(models.Result)
                .where(
                    models.Result.message_id == detail.message_id,
                    models.Result.key == detail.key,
                )
                .values(payload=outcome_to_json(rescreened))
            )
    app = open_app("planner", query={"event": planned.event_id})
    assert not app.exception
    return app


def test_a_rescreen_with_a_conflict(store_path: Path, planned: Planned) -> None:
    outcome = _outcome(planned)
    tca = outcome.event.latest.tca
    result = _rescreen(
        outcome,
        new=(_approach(40001, tca + 2 * u.hour, 2e-4, RiskLevel.RED),),
        closer=(_approach(40002, tca + 4 * u.hour, 3e-6, RiskLevel.YELLOW),),
        event_range=1234.0,
    )
    app = _rescreened(store_path, planned, (result, None))
    text = shown(app)
    assert "re-screened against 120 objects (the test catalog)" in text
    assert "range 1234 m after the burn" in text
    assert [item.value for item in app.error] == [
        "CONFLICT: a new or closer approach is at red."
    ]
    assert ASSUMED_COVARIANCES in text
    # Each approach once, in a data table, the highest Pc first.
    (table,) = [frame.value for frame in app.dataframe if "risk" in frame.value]
    assert table["approach"].tolist() == ["new", "closer"]
    assert table["secondary"].tolist() == [40001, 40002]
    assert table["risk"].tolist() == [
        risk_text(RiskLevel.RED),
        risk_text(RiskLevel.YELLOW),
    ]
    assert table["assumed Pc"].tolist() == ["2.00e-04", "3.00e-06"]
    # The caption says every Pc rests on assumed covariances: no column.
    assert "provenance" not in table
    assert table["range [m]"].tolist() == pytest.approx([300.0, 300.0])
    # The second burn was not re-screened: one head only.
    assert text.count("re-screened against") == 1


def test_a_rescreen_without_approaches(store_path: Path, planned: Planned) -> None:
    outcome = _outcome(planned)
    result = _rescreen(outcome, screened=False)
    app = _rescreened(store_path, planned, (result, result))
    text = shown(app)
    assert not app.error
    assert (
        text.count("No new or closer approach at red among the objects screened.") == 2
    )
    assert "was NOT screened" in text
    assert not [frame for frame in app.dataframe if "risk" in frame.value]
    assert ASSUMED_COVARIANCES not in text


def test_a_rescreen_bringing_the_pair_back(store_path: Path, planned: Planned) -> None:
    outcome = _outcome(planned)
    back = _approach(40003, outcome.event.latest.tca + 12 * u.hour, 1e-3, RiskLevel.RED)
    result = _rescreen(outcome, returns=(back,))
    app = _rescreened(store_path, planned, (None, result))
    text = shown(app)
    assert not app.error
    assert "no approach within" in text
    (warning,) = [item.value for item in app.warning]
    assert "brings this pair back" in warning
    assert "Pc 1.00e-03 (assumed)" in warning
    # No approach to tabulate, but the Pc shown rests on assumed covariances.
    assert not [frame for frame in app.dataframe if "risk" in frame.value]
    assert ASSUMED_COVARIANCES in text


@pytest.mark.slow
def test_a_rescreened_plan_shows_the_rescreen(
    seed_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A seed event keeps the catalog it was screened with: each burn of its
    # plan is re-screened (tens of seconds).
    path = Path(shutil.copy2(seed_db, tmp_path / seed_db.name))
    catalogs = f"{seed_db.name}.catalogs"
    shutil.copytree(seed_db.with_name(catalogs), tmp_path / catalogs)
    settings_file = tmp_path / "settings.toml"
    settings_file.write_text(SETTINGS)
    settings = load_settings(settings_file)
    with open_store(path, create=False) as opened:
        page = opened.summaries(settings, as_of=seed.seed_as_of())
        reference = opened.catalog(page.items[0].event_id)
    assert reference is not None
    for item in page.items:
        try:
            _plan(path, item.event_id, settings)
        except KeepoutError:
            continue  # its latest CDM is past the burn window
        event_id = item.event_id
        break
    else:
        pytest.fail("no seed event can be planned")
    use_store(monkeypatch, path, settings_file, seed.seed_as_of())
    app = open_app("planner", query={"event": event_id})
    assert not app.exception
    assert not app.error
    text = shown(app)
    assert "Not re-screened" not in text
    assert "re-screened against" in text
    assert "Event secondary" in text
    assert "No new or closer approach at red" in text or "CONFLICT" in text
    # Each approach once, in a data table: its risk in words.
    tables = [frame.value for frame in app.dataframe if "risk" in frame.value]
    assert tables
    for table in tables:
        assert set(table["risk"]) <= {risk_text(level) for level in RiskLevel}
    assert ":orange-badge[assumed (GP age model)]" in text


@pytest.mark.parametrize("typed", ["abc", "0", "1", "nan", "-1e-5"])
def test_a_bad_target_pc_is_an_error(
    store_path: Path, planned: Planned, typed: str
) -> None:
    event_id = planned.event_id
    before = len(_jobs(store_path, event_id))
    app = open_app("planner", query={"event": event_id})
    _what_if(app, event_id, typed, None)
    assert [item.value for item in app.error] == [
        f"The target Pc must be a number in (0, 1), got {typed!r}."
    ]
    assert len(_jobs(store_path, event_id)) == before


def test_each_job_names_its_settings(store_path: Path, planned: Planned) -> None:
    event_id = planned.event_id
    app = open_app("planner", query={"event": event_id})
    _what_if(app, event_id, 1e-6, None)
    _what_if(app, event_id, 1e-7, 30.0)
    jobs = [job for job in _jobs(store_path, event_id) if job.status.is_active]
    assert len(jobs) == 2
    radii = {1e-6: "the CDMs' radii", 1e-7: "30 m"}
    assert {job.settings.target_pc for job in jobs} == set(radii)
    lines = shown(app).splitlines()
    for job in jobs:
        status = lines.index(f"Job {job.id} (manoeuvre): **queued**")
        # The caption right under the job's own status line.
        assert lines[status + 1] == (
            f"target Pc {job.settings.target_pc:.2e}, combined hard-body radius "
            f"{radii[job.settings.target_pc]}."
        )


def test_a_refused_cancel_of_a_job_planned_meanwhile(
    store_path: Path, planned: Planned, monkeypatch: pytest.MonkeyPatch
) -> None:
    event_id = planned.event_id
    app = open_app("planner", query={"event": event_id})
    _what_if(app, event_id, 1e-6, None)
    (job,) = [job for job in _jobs(store_path, event_id) if job.status.is_active]
    cancel = JobQueue.cancel

    def worker_first(queue: JobQueue, job_id: int) -> Job:
        # A worker plans the job between the page's read and the click: the
        # plan is cached, and the page no longer shows the job.
        with open_store(store_path) as opened:
            claimed = JobQueue(opened).claim("test")
            assert claimed is not None
            payload = run_maneuver_job(str(store_path), claimed, 1, grid=SMALL_GRID)
            assert JobQueue(opened).finish(claimed, payload)
        return cancel(queue, job_id)

    monkeypatch.setattr(JobQueue, "cancel", worker_first)
    app.button(key=f"cancel-job-{job.id}").click().run()
    assert not app.exception
    assert f"Job {job.id} " not in shown(app)
    assert len(app.get("plotly_chart")) == 1
    assert [item.value for item in app.toast] == [
        f"job {job.id} has already ended: succeeded"
    ]
    assert not app.error
    # Not kept for the next run.
    app.run()
    assert not app.toast
    assert not app.error


def test_an_unchanged_many_digit_target_is_the_operators(
    store_path: Path, planned: Planned, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The form's default must read back as the operator's very target.
    settings_file = tmp_path / "precise.toml"
    settings_file.write_text(SETTINGS.replace("1e-5", "1.234567891e-5"))
    monkeypatch.setenv("KEEPOUT_SETTINGS", str(settings_file))
    _plan(store_path, planned.event_id, load_settings(settings_file))
    before = len(_jobs(store_path, planned.event_id))
    app = open_app("planner", query={"event": planned.event_id})
    app.button(key=f"planner-what-if-{planned.event_id}").click().run()
    assert not app.exception
    assert not app.error
    assert len(_jobs(store_path, planned.event_id)) == before
    assert "What-if" not in shown(app)
    assert len(app.get("plotly_chart")) == 1


def test_the_plans_notes_are_said_once_and_collapsed(
    store_path: Path, planned: Planned
) -> None:
    notes = merged_notes(_outcome(planned).notes)
    assert notes
    app = open_app("planner", query={"event": planned.event_id})
    (expander,) = [item for item in app.expander if item.label.startswith("Notes")]
    assert expander.label == f"Notes ({len(notes)})"
    assert not expander.proto.expanded
    assert not [item for item in app.caption if item.value.startswith("Note:")]


def test_the_burns_table_gives_signed_dv_and_pc_with_two_exponent_digits(
    store_path: Path, planned: Planned
) -> None:
    app = open_app("planner", query={"event": planned.event_id})
    burns = app.dataframe[0]
    columns = json.loads(burns.proto.columns)
    assert columns["Δv (m/s)"]["type_config"]["format"] == "%+.4f"
    pcs = burns.value["Pc"].tolist()
    assert pcs[0] == f"{_outcome(planned).baseline.pc:.2e}"
    assert all(len(pc.split("e")[1]) == 3 for pc in pcs)


def test_the_grid_cells_are_a_collapsed_table(
    store_path: Path, planned: Planned
) -> None:
    grid = _outcome(planned).grid
    assert grid is not None
    app = open_app("planner", query={"event": planned.event_id})
    (expander,) = [item for item in app.expander if item.label.startswith("Grid")]
    assert expander.label == f"Grid cells ({grid.pc.size})"
    assert not expander.proto.expanded
    (cells,) = [frame.value for frame in expander.dataframe]
    assert len(cells) == grid.pc.size


def test_the_heatmap_is_made_once_per_plan(
    store_path: Path, planned: Planned, monkeypatch: pytest.MonkeyPatch
) -> None:
    made: list[None] = []
    original = figures.grid_figure

    def counted(outcome: ManeuverOutcome) -> GridFigure:
        made.append(None)
        return original(outcome)

    monkeypatch.setattr(figures, "grid_figure", counted)
    app = open_app("planner", query={"event": planned.event_id})
    app.run()
    assert not app.exception
    assert len(app.get("plotly_chart")) == 1
    assert len(made) == 1
