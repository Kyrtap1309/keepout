"""The widgets the dashboard's pages share: risk chips, provenance badges and
the status of jobs."""

from pathlib import Path

import dashboard_seed as seed
import pytest
from dashboard_builders import TIMEOUT, OpenApp
from streamlit.testing.v1 import AppTest

from keepout.dashboard.widgets import ink
from keepout.figures import ACTION_COLOURS, RISK_COLOURS, Provenance
from keepout.risk import Action, RiskLevel
from keepout.settings import load_settings
from keepout.store import Job, JobQueue, JobStatus, open_store


def _chips(level: str | None, action: str | None) -> None:
    from keepout.dashboard.widgets import risk_chip
    from keepout.risk import Action, RiskLevel

    risk_chip(
        None if level is None else RiskLevel[level],
        None if action is None else Action[action],
    )


def _badges() -> None:
    from keepout.dashboard.widgets import provenance_badge
    from keepout.figures import Provenance

    for item in Provenance:
        provenance_badge(item)


def _jobs_after_a_store_error(store: str, job_ids: list[int]) -> None:
    import streamlit as st

    from keepout.dashboard.widgets import job_status
    from keepout.errors import StoreError

    if not st.session_state.get("failed_once"):
        # As the polling fragment leaves a store error for the page.
        st.session_state["failed_once"] = True
        key = "job-status-error-" + ",".join(map(str, job_ids))
        st.session_state[key] = StoreError("database is locked")
    from pathlib import Path

    job_status(Path(store), job_ids)


def _jobs(store: str, job_ids: list[int]) -> None:
    from pathlib import Path

    from keepout.dashboard.widgets import job_status

    job_status(Path(store), job_ids)


@pytest.mark.parametrize(
    ("colour", "text"),
    [
        (RISK_COLOURS[RiskLevel.GREEN], "#000000"),
        (RISK_COLOURS[RiskLevel.YELLOW], "#000000"),
        (RISK_COLOURS[RiskLevel.RED], "#ffffff"),
        (ACTION_COLOURS[Action.PREPARE_MANEUVER], "#000000"),
    ],
)
def test_chip_text_contrasts_with_its_fill(colour: str, text: str) -> None:
    assert ink(colour) == text


def test_risk_chip_shows_colour_and_words() -> None:
    app = AppTest.from_function(
        _chips, args=("RED", "MANEUVER"), default_timeout=TIMEOUT
    ).run()
    [chip] = [item.value for item in app.markdown]
    assert "red · maneuver" in chip
    assert f"background:{RISK_COLOURS[RiskLevel.RED]}" in chip
    assert "color:#ffffff" in chip


def test_risk_chip_without_a_pc() -> None:
    app = AppTest.from_function(_chips, args=(None, None), default_timeout=TIMEOUT)
    [chip] = [item.value for item in app.run().markdown]
    assert ">not computed<" in chip
    assert "background" not in chip


def test_provenance_badges_name_the_provenance() -> None:
    app = AppTest.from_function(_badges, default_timeout=TIMEOUT).run()
    shown = " ".join(item.value for item in app.markdown)
    assert all(item.value in shown for item in Provenance)


def test_job_status_cancels_an_active_job(seed_copy: Path) -> None:
    settings = load_settings(seed.SETTINGS)
    with open_store(seed_copy) as store:
        event = store.events()[0].event_id
        job = JobQueue(store).submit(event, settings).job
    assert job is not None
    app = AppTest.from_function(
        _jobs, args=(str(seed_copy), [job.id]), default_timeout=TIMEOUT
    ).run()
    assert not app.exception
    assert f"Job {job.id} (maneuver): **queued**" in app.markdown[0].value
    app.button(key=f"cancel-job-{job.id}").click().run()
    assert not app.exception
    with open_store(seed_copy) as store:
        assert JobQueue(store).job(job.id).status == JobStatus.CANCELLED
    assert f"Job {job.id} (maneuver): **cancelled**" in app.markdown[0].value
    assert not app.button


def test_job_status_of_a_missing_job(seed_db: Path) -> None:
    app = AppTest.from_function(
        _jobs, args=(str(seed_db), [999]), default_timeout=TIMEOUT
    ).run()
    assert not app.exception
    assert [item.value for item in app.error] == ["no job 999 in the store"]


def test_a_page_opens_with_its_query(app_test: OpenApp) -> None:
    app = app_test("event", query={"event": "abc"})
    assert not app.exception
    assert app.query_params["event"] == "abc"
    assert [item.value for item in app.title] == ["Event"]


def _queued_job(path: Path) -> int:
    settings = load_settings(seed.SETTINGS)
    with open_store(path) as store:
        event = store.events()[0].event_id
        job = JobQueue(store).submit(event, settings).job
    assert job is not None
    return job.id


def test_a_refused_cancel_shows_its_error(
    seed_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = _queued_job(seed_copy)
    app = AppTest.from_function(
        _jobs, args=(str(seed_copy), [job_id]), default_timeout=TIMEOUT
    ).run()
    cancel = JobQueue.cancel

    def worker_first(queue: JobQueue, job: int) -> Job:
        # The worker ends the job between the page's read and the click.
        cancel(queue, job)
        return cancel(queue, job)

    monkeypatch.setattr(JobQueue, "cancel", worker_first)
    app.button(key=f"cancel-job-{job_id}").click().run()
    assert not app.exception
    assert [item.value for item in app.error] == [
        f"job {job_id} has already ended: cancelled"
    ]


def test_a_refused_cancel_stays_through_the_polling_rerun(
    seed_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = _queued_job(seed_copy)
    app = AppTest.from_function(
        _jobs, args=(str(seed_copy), [job_id]), default_timeout=TIMEOUT
    ).run()
    cancel = JobQueue.cancel

    def worker_first(queue: JobQueue, job: int) -> Job:
        cancel(queue, job)
        return cancel(queue, job)

    monkeypatch.setattr(JobQueue, "cancel", worker_first)
    app.button(key=f"cancel-job-{job_id}").click().run()
    assert not app.exception
    # The polling found the job ended and ran the page again: the job is
    # shown as it is now, no longer polled, and the refusal is still there.
    assert f"Job {job_id} (maneuver): **cancelled**" in app.markdown[0].value
    assert not app.button
    assert [item.value for item in app.error] == [
        f"job {job_id} has already ended: cancelled"
    ]
    # The user's next interaction drops it.
    app.run()
    assert not app.exception
    assert not app.error
    assert f"Job {job_id} (maneuver): **cancelled**" in app.markdown[0].value


def test_a_failed_job_shows_its_error(seed_copy: Path) -> None:
    job_id = _queued_job(seed_copy)
    with open_store(seed_copy) as store:
        queue = JobQueue(store)
        job = queue.claim("test")
        assert job is not None
        assert job.id == job_id
        assert queue.fail(job, "no burn")
    app = AppTest.from_function(
        _jobs, args=(str(seed_copy), [job_id]), default_timeout=TIMEOUT
    ).run()
    assert not app.exception
    assert app.markdown[0].value == f"Job {job_id} (maneuver): **failed**"
    assert [item.value for item in app.error] == ["no burn"]


def test_a_store_error_stops_the_polling_until_the_next_run(seed_copy: Path) -> None:
    job_id = _queued_job(seed_copy)
    app = AppTest.from_function(
        _jobs_after_a_store_error,
        args=(str(seed_copy), [job_id]),
        default_timeout=TIMEOUT,
    ).run()
    assert not app.exception
    assert [item.value for item in app.error] == ["database is locked"]
    assert not app.button
    app.run()
    assert not app.error
    assert f"Job {job_id} (maneuver): **queued**" in app.markdown[0].value
