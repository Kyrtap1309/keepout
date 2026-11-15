"""The dashboard's manoeuvre planner page: an event's cached plan (the Pc
grid, the smallest burn each way and its re-screen), the job that plans it,
what-if jobs on another target Pc or hard-body radius, and their status."""

import dataclasses
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import dashboard_seed as seed
import pytest
from builders import SMALL_GRID
from cli_builders import operator_kvn
from dashboard_builders import open_app
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import Block, Node, PageLink

import keepout.cdm
from keepout.errors import KeepoutError
from keepout.figures import RISK_LABELS
from keepout.settings import Settings, load_settings
from keepout.store import CdmInput, Job, JobKind, JobQueue, JobStatus, open_store
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


def _walk(node: Node) -> Iterator[Node]:
    """The elements under ``node``, depth first, in the page's order."""
    if isinstance(node, Block):
        for child in node.children.values():
            yield from _walk(child)
    else:
        yield node


def _shown(app: AppTest) -> str:
    """Every text of the page's main area, one per line."""
    return "\n".join(str(getattr(node, "value", "")) for node in _walk(app.main))


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
def store(planned: Planned, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A copy of :func:`planned` the test may write to, which the app opens."""
    path = Path(shutil.copy2(planned.store, tmp_path / "planned.db"))
    monkeypatch.setenv("KEEPOUT_STORE", str(path))
    monkeypatch.setenv("KEEPOUT_SETTINGS", str(planned.settings))
    monkeypatch.setenv("KEEPOUT_AS_OF", str(seed.seed_as_of().utc.isot))
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
    store = Path(app.session_state["context"].store_path)
    app.button(key=f"planner-plan-{event_id}").click().run()
    assert not app.exception
    (job,) = _jobs(store, event_id)
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


def test_no_event_param(store: Path) -> None:
    app = open_app("planner")
    assert not app.exception
    assert [item.value for item in app.title] == ["Manoeuvre planner"]
    assert "Pick an event" in _shown(app)


def test_unknown_event(store: Path) -> None:
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


def test_cached_plan_shown(store: Path, planned: Planned) -> None:
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
    shown = _shown(app)
    assert "A burn can be recommended" in shown
    assert "Not re-screened: no catalog stored" in shown
    assert "assumed (GP age model)" in shown
    assert "target Pc 1.00e-05" in shown
    # No job is left for the plan.
    assert all(
        button.key != f"planner-plan-{planned.event_id}" for button in app.button
    )
    assert "Job " not in shown


def test_no_plan_offers_queue(store: Path) -> None:
    event_id = _unplanned(store)
    app = open_app("planner", query={"event": event_id})
    assert not app.exception
    assert not app.get("plotly_chart")
    job = _queue(app, event_id)
    assert job.status is JobStatus.QUEUED
    assert f"Job {job.id} (maneuver): **queued**" in _shown(app)
    # While it is queued, the page offers no second job.
    assert all(button.key != f"planner-plan-{event_id}" for button in app.button)


def test_a_queued_job_is_found_after_a_reload(store: Path, planned: Planned) -> None:
    event_id = _unplanned(store)
    with open_store(store) as opened:
        job = JobQueue(opened).submit(event_id, load_settings(planned.settings)).job
    assert job is not None
    app = open_app("planner", query={"event": event_id})
    assert not app.exception
    assert f"Job {job.id} (maneuver): **queued**" in _shown(app)


def test_what_if_dedupes(store: Path, planned: Planned) -> None:
    event_id = planned.event_id
    before = {job.id for job in _jobs(store, event_id)}
    app = open_app("planner", query={"event": event_id})
    _what_if(app, event_id, 1e-6, None)
    _what_if(app, event_id, 1e-6, None)
    added = [job for job in _jobs(store, event_id) if job.id not in before]
    (job,) = added
    assert job.status is JobStatus.QUEUED
    assert job.settings.target_pc == 1e-6
    assert "What-if" in _shown(app)
    assert not app.get("plotly_chart")
    # The operator's own values: their plan is cached, no job.
    _what_if(app, event_id, 1e-5, None)
    assert [item.id for item in _jobs(store, event_id) if item.id not in before] == [
        job.id
    ]
    assert len(app.get("plotly_chart")) == 1
    assert "What-if" not in _shown(app)


def test_cancel_job(store: Path) -> None:
    event_id = _unplanned(store)
    app = open_app("planner", query={"event": event_id})
    job = _queue(app, event_id)
    app.button(key=f"cancel-job-{job.id}").click().run()
    assert not app.exception
    assert not app.error
    (cancelled,) = _jobs(store, event_id)
    assert cancelled.status is JobStatus.CANCELLED
    assert f"Job {job.id} (maneuver): **cancelled**" in _shown(app)
    # It can be queued again.
    assert any(button.key == f"planner-plan-{event_id}" for button in app.button)


def test_failed_job_shows_error(store: Path, planned: Planned) -> None:
    event_id = _unplanned(store)
    with open_store(store) as opened:
        queue = JobQueue(opened)
        queue.submit(event_id, load_settings(planned.settings))
        job = queue.claim("test")
        assert job is not None
        assert queue.fail(job, "the burn window closed")
    app = open_app("planner", query={"event": event_id})
    assert not app.exception
    assert [item.value for item in app.error] == ["the burn window closed"]
    assert f"Job {job.id} (maneuver): **failed**" in _shown(app)


def test_refusal_shown(store: Path, planned: Planned) -> None:
    event_id = planned.event_id
    before = len(_jobs(store, event_id))
    app = open_app("planner", query={"event": event_id})
    _what_if(app, event_id, REFUSED_TARGET, REFUSED_HBR)
    assert len(_jobs(store, event_id)) == before
    assert not app.error
    assert [item.value for item in app.warning] == [
        "No burn on the grid meets the target Pc 1.00e-12."
    ]
    burns, _ = (frame.value for frame in app.dataframe)
    assert burns["Option"].tolist() == ["no burn"]
    assert len(app.get("plotly_chart")) == 1


def test_a_bad_what_if_is_an_error(store: Path, planned: Planned) -> None:
    event_id = planned.event_id
    app = open_app("planner", query={"event": event_id})
    _what_if(app, event_id, 1e-5, 0.0)
    assert len(app.error) == 1
    assert "combined_hard_body_radius" in str(app.error[0].value)


def test_back_to_the_operators_settings(store: Path, planned: Planned) -> None:
    event_id = planned.event_id
    app = open_app("planner", query={"event": event_id})
    _what_if(app, event_id, REFUSED_TARGET, REFUSED_HBR)
    assert app.warning
    app.button(key=f"planner-own-{event_id}").click().run()
    assert not app.exception
    assert not app.warning
    assert "A burn can be recommended" in _shown(app)


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
    monkeypatch.setenv("KEEPOUT_STORE", str(path))
    monkeypatch.setenv("KEEPOUT_SETTINGS", str(settings_file))
    monkeypatch.setenv("KEEPOUT_AS_OF", str(seed.seed_as_of().utc.isot))
    app = open_app("planner", query={"event": event_id})
    assert not app.exception
    assert not app.error
    shown = _shown(app)
    assert "Not re-screened" not in shown
    assert "re-screened against" in shown
    assert "Event secondary" in shown
    assert "No new or closer approach at red" in shown or "CONFLICT" in shown
    # Each approach's risk: a chip of its level's colour and words, and
    # the same words in the data table, its Pc next to its provenance.
    tables = [frame.value for frame in app.dataframe if "risk" in frame.value]
    assert tables
    for table in tables:
        for label in table["risk"]:
            assert label in RISK_LABELS.values()
            assert f'">{label}</span>' in shown
        assert set(table["provenance"]) == {"assumed (GP age model)"}
    assert ":orange-badge[assumed (GP age model)]" in shown


@pytest.mark.parametrize("typed", ["abc", "0", "1", "nan", "-1e-5"])
def test_a_bad_target_pc_is_an_error(store: Path, planned: Planned, typed: str) -> None:
    event_id = planned.event_id
    before = len(_jobs(store, event_id))
    app = open_app("planner", query={"event": event_id})
    _what_if(app, event_id, typed, None)
    assert [item.value for item in app.error] == [
        f"The target Pc must be a number in (0, 1), got {typed!r}."
    ]
    assert len(_jobs(store, event_id)) == before


def test_each_job_names_its_settings(store: Path, planned: Planned) -> None:
    event_id = planned.event_id
    app = open_app("planner", query={"event": event_id})
    _what_if(app, event_id, 1e-6, None)
    _what_if(app, event_id, 1e-7, 30.0)
    jobs = [job for job in _jobs(store, event_id) if job.status.is_active]
    assert len(jobs) == 2
    lines = _shown(app).splitlines()
    for job in jobs:
        status = lines.index(f"Job {job.id} (maneuver): **queued**")
        # The caption right under the job's own status line.
        assert lines[status + 1] == (
            f"target Pc {job.settings.target_pc:.2e}, combined hard-body radius "
            + ("the CDMs' radii." if job.id == jobs[1].id else "30 m.")
        )


def test_a_refused_cancel_outlives_its_job_on_the_page(
    store: Path, planned: Planned, monkeypatch: pytest.MonkeyPatch
) -> None:
    event_id = planned.event_id
    app = open_app("planner", query={"event": event_id})
    _what_if(app, event_id, 1e-6, None)
    (job,) = [job for job in _jobs(store, event_id) if job.status.is_active]
    cancel = JobQueue.cancel

    def worker_first(queue: JobQueue, job_id: int) -> Job:
        # A worker plans the job between the page's read and the click: the
        # plan is cached, and the page no longer shows the job.
        with open_store(store) as opened:
            claimed = JobQueue(opened).claim("test")
            assert claimed is not None
            payload = run_maneuver_job(str(store), claimed, 1, grid=SMALL_GRID)
            assert JobQueue(opened).finish(claimed, payload)
        return cancel(queue, job_id)

    monkeypatch.setattr(JobQueue, "cancel", worker_first)
    app.button(key=f"cancel-job-{job.id}").click().run()
    assert not app.exception
    assert f"Job {job.id} " not in _shown(app)
    assert len(app.get("plotly_chart")) == 1
    assert [item.value for item in app.error] == [
        f"job {job.id} has already ended: succeeded"
    ]
    # The user's next interaction drops it, for good.
    app.run()
    assert not app.error
    app.run()
    assert not app.error


def test_an_unchanged_many_digit_target_is_the_operators(
    store: Path, planned: Planned, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The form's default must read back as the operator's very target.
    settings_file = tmp_path / "precise.toml"
    settings_file.write_text(SETTINGS.replace("1e-5", "1.234567891e-5"))
    monkeypatch.setenv("KEEPOUT_SETTINGS", str(settings_file))
    _plan(store, planned.event_id, load_settings(settings_file))
    before = len(_jobs(store, planned.event_id))
    app = open_app("planner", query={"event": planned.event_id})
    app.button(key=f"planner-what-if-{planned.event_id}").click().run()
    assert not app.exception
    assert not app.error
    assert len(_jobs(store, planned.event_id)) == before
    assert "What-if" not in _shown(app)
    assert len(app.get("plotly_chart")) == 1
