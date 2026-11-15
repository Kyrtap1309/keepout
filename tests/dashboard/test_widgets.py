"""The widgets the dashboard's pages share: risk chips, provenance badges and
the status of jobs."""

from collections.abc import Sequence
from pathlib import Path

import dashboard_seed as seed
import pytest
from dashboard_builders import TIMEOUT
from hypothesis import given
from hypothesis import strategies as st
from streamlit.testing.v1 import AppTest

from keepout.dashboard import widgets
from keepout.dashboard.widgets import ink
from keepout.errors import StoreError
from keepout.figures import PROVENANCE_LABELS, RISK_COLOURS
from keepout.risk import RiskLevel
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
    from keepout.risk import Provenance

    for item in Provenance:
        provenance_badge(item)


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
        ("#d55e00", "#000000"),
    ],
)
def test_chip_text_contrasts_with_its_fill(colour: str, text: str) -> None:
    assert ink(colour) == text


def _contrast(first: str, second: str) -> float:
    """The WCAG 2 contrast ratio of two ``#rrggbb`` colours."""

    def luminance(colour: str) -> float:
        srgb = [int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        r, g, b = (
            c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb
        )
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    lighter, darker = sorted((luminance(first), luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


@given(st.integers(0, 0xFFFFFF).map(lambda rgb: f"#{rgb:06x}"))
def test_chip_text_is_the_ink_of_more_contrast(fill: str) -> None:
    chosen = ink(fill)
    (other,) = {"#000000", "#ffffff"} - {chosen}
    assert _contrast(fill, chosen) >= _contrast(fill, other)


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
    assert all(label in shown for label in PROVENANCE_LABELS.values())


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


def _queued_job(path: Path) -> int:
    settings = load_settings(seed.SETTINGS)
    with open_store(path) as store:
        event = store.events()[0].event_id
        job = JobQueue(store).submit(event, settings).job
    assert job is not None
    return job.id


@pytest.fixture
def worker_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every cancel of a job comes after a worker's: the worker ends the
    job between the page's read and the click."""
    cancel = JobQueue.cancel

    def cancelled_first(queue: JobQueue, job: int) -> Job:
        cancel(queue, job)
        return cancel(queue, job)

    monkeypatch.setattr(JobQueue, "cancel", cancelled_first)


@pytest.mark.usefixtures("worker_first")
def test_a_refused_cancel_is_a_toast(seed_copy: Path) -> None:
    job_id = _queued_job(seed_copy)
    app = AppTest.from_function(
        _jobs, args=(str(seed_copy), [job_id]), default_timeout=TIMEOUT
    ).run()
    app.button(key=f"cancel-job-{job_id}").click().run()
    assert not app.exception
    # The page ran again: the job as it is now, no longer polled.
    assert f"Job {job_id} (maneuver): **cancelled**" in app.markdown[0].value
    assert not app.button
    assert [item.value for item in app.toast] == [
        f"job {job_id} has already ended: cancelled"
    ]
    assert not app.error
    # Not kept: the next run has no toast.
    app.run()
    assert not app.exception
    assert not app.toast
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


def _fails(monkeypatch: pytest.MonkeyPatch, failures: Sequence[bool]) -> None:
    """Make the reads of jobs fail (``database is locked``) or not, in
    turn; those after ``failures`` succeed."""
    read = widgets._read_jobs
    turns = iter(failures)

    def reads(store_path: Path, job_ids: Sequence[int]) -> list[Job]:
        if next(turns, False):
            raise StoreError("database is locked")
        return read(store_path, job_ids)

    monkeypatch.setattr(widgets, "_read_jobs", reads)


def test_a_store_error_is_shown_until_the_next_run(
    seed_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = _queued_job(seed_copy)
    _fails(monkeypatch, [True])
    app = AppTest.from_function(
        _jobs, args=(str(seed_copy), [job_id]), default_timeout=TIMEOUT
    ).run()
    assert not app.exception
    assert [item.value for item in app.error] == ["database is locked"]
    assert not app.button
    app.run()
    assert not app.error
    assert f"Job {job_id} (maneuver): **queued**" in app.markdown[0].value


def test_a_store_error_while_polling_is_shown_in_place_of_the_jobs(
    seed_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The page's read succeeds; the polling's next one (a click reruns the
    # fragment) fails: its error shows instead of the jobs, nothing is done.
    job_id = _queued_job(seed_copy)
    _fails(monkeypatch, [False, True])
    app = AppTest.from_function(
        _jobs, args=(str(seed_copy), [job_id]), default_timeout=TIMEOUT
    ).run()
    app.button(key=f"cancel-job-{job_id}").click().run()
    assert not app.exception
    assert [item.value for item in app.error] == ["database is locked"]
    assert not app.button
    # A later read succeeds: the job is shown again, still queued.
    app.run()
    assert not app.error
    assert f"Job {job_id} (maneuver): **queued**" in app.markdown[0].value
