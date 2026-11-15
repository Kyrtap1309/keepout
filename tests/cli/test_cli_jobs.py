"""``keepout jobs`` and ``keepout worker``: the store's background jobs on the
command line. The worker's computations are those of ``worker_jobs``, run in
child processes as the real one is."""

import json
import os
import re
import signal
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from cli_builders import (
    keepout,
    operator_run,
    queued_job,
    stored_crossing,
    usage_error,
)
from inline_snapshot import snapshot
from sqlalchemy import create_engine, select
from worker_jobs import ready_then_sleep, workers_given

from keepout.cli.output import ExitCode
from keepout.pc import computation
from keepout.settings import Settings
from keepout.store import JobQueue, models, open_store, planning
from keepout.store import db as store_db


@pytest.fixture
def stored(tmp_path: Path) -> tuple[Path, str]:
    """:func:`~cli_builders.stored_crossing` in the test's folder."""
    return stored_crossing(tmp_path)


def _times_apart(value: object) -> object:
    """``value`` with each time of the JSON (to the millisecond) ``<time>``."""
    if isinstance(value, str) and re.fullmatch(r"[\d-]{10}T[\d:]{8}\.\d{3}", value):
        return "<time>"
    return value


def _failed_oldest(db: Path) -> None:
    """Claim the oldest job as worker ``w1`` and fail it: ``no burn``."""
    with open_store(db, create=False) as store:
        queue = JobQueue(store)
        job = queue.claim("w1")
        assert job is not None
        assert queue.fail(job, "no burn")


def test_list_on_an_empty_queue(stored: tuple[Path, str]) -> None:
    db, _ = stored
    empty = keepout("jobs", "list", "--store", str(db))
    assert (empty.exit_code, empty.stdout) == (ExitCode.OK, f"No jobs in {db}.\n")


def test_list_shows_the_jobs_newest_first(stored: tuple[Path, str]) -> None:
    db, event_id = stored
    assert queued_job(db, event_id) == 1
    assert queued_job(db, event_id, "--target-pc", "1e-6") == 2
    _failed_oldest(db)
    # The CDM is the one the job is pinned to: the message, not the event.
    cdm = operator_run().cdms[0].message_id
    assert (event_id, cdm) == snapshot(
        ("2a85ed50c4e6", "52939_47826_20261006T193642.717Z_20261006T000000.000Z")
    )
    listed = keepout("jobs", "list", "--store", str(db), env={"COLUMNS": "200"})
    assert listed.exit_code == ExitCode.OK
    lines = listed.stdout.splitlines()
    assert lines[0] == f"Jobs in {db}"
    minutes = re.compile(r"\d{4}-\d\d-\d\d \d\d:\d\d")
    cells = [
        re.split(r" {2,}", minutes.sub("<minute>", line).strip())
        for line in (lines[1], *lines[3:])
    ]
    assert cells == snapshot(
        [
            [
                "Job",
                "Kind",
                "Event",
                "CDM",
                "Status",
                "Attempt",
                "Created (UTC)",
                "Error",
            ],
            [
                "2",
                "maneuver",
                "2a85ed50c4e6",
                "52939_47826_20261006T193642.717Z_20261006T000000.000Z",
                "queued",
                "0",
                "<minute>",
            ],
            [
                "1",
                "maneuver",
                "2a85ed50c4e6",
                "52939_47826_20261006T193642.717Z_20261006T000000.000Z",
                "failed",
                "1",
                "<minute>",
                "no burn",
            ],
        ]
    )
    payload = json.loads(keepout("jobs", "list", "--store", str(db), "--json").stdout)
    assert payload["path"] == str(db)
    jobs = [
        {key: _times_apart(value) for key, value in job.items()}
        for job in payload["jobs"]
    ]
    assert jobs == snapshot(
        [
            {
                "job": 2,
                "kind": "maneuver",
                "event": "2a85ed50c4e6",
                "cdm": "52939_47826_20261006T193642.717Z_20261006T000000.000Z",
                "status": "queued",
                "attempt": 0,
                "worker": None,
                "created": "<time>",
                "started": None,
                "finished": None,
                "error": None,
            },
            {
                "job": 1,
                "kind": "maneuver",
                "event": "2a85ed50c4e6",
                "cdm": "52939_47826_20261006T193642.717Z_20261006T000000.000Z",
                "status": "failed",
                "attempt": 1,
                "worker": "w1",
                "created": "<time>",
                "started": "<time>",
                "finished": "<time>",
                "error": "no burn",
            },
        ]
    )


def test_cli_jobs_list_shows_kind(stored: tuple[Path, str]) -> None:
    db, event_id = stored
    assert queued_job(db, event_id) == 1
    with open_store(db, create=False) as store:
        # Not the default settings: `store add` cached their Pc.
        foster = Settings(compute=computation("foster"))
        refresh = JobQueue(store).submit_refresh(event_id, foster)
    assert refresh.job is not None
    listed = keepout("jobs", "list", "--store", str(db), env={"COLUMNS": "200"})
    assert listed.exit_code == ExitCode.OK
    rows = [re.split(r" {2,}", line.strip()) for line in listed.stdout.splitlines()]
    assert rows[1][:2] == ["Job", "Kind"]
    assert [row[:2] for row in rows[3:]] == [
        [str(refresh.job.id), "refresh"],
        ["1", "maneuver"],
    ]
    payload = json.loads(keepout("jobs", "list", "--store", str(db), "--json").stdout)
    assert [job["kind"] for job in payload["jobs"]] == ["refresh", "maneuver"]


def test_cancel(stored: tuple[Path, str]) -> None:
    db, event_id = stored
    job = queued_job(db, event_id)
    cancelled = keepout("jobs", "cancel", str(job), "--store", str(db))
    assert cancelled.exit_code == ExitCode.OK
    assert cancelled.stdout == f"Cancelled job {job} (event {event_id}).\n"
    again = keepout("jobs", "cancel", str(job), "--store", str(db))
    assert again.exit_code == ExitCode.ERROR
    assert again.stderr == f"Error: job {job} has already ended: cancelled\n"
    payload = json.loads(keepout("jobs", "list", "--store", str(db), "--json").stdout)
    assert payload["jobs"][0]["status"] == "cancelled"
    # A cancelled job is not one to wait for: queued again, a new job.
    assert queued_job(db, event_id) == job + 1


def test_cancel_json(stored: tuple[Path, str]) -> None:
    db, event_id = stored
    job = queued_job(db, event_id)
    cancelled = keepout("jobs", "cancel", str(job), "--store", str(db), "--json")
    assert json.loads(cancelled.stdout) == {
        "job": job,
        "event": event_id,
        "status": "cancelled",
    }


def test_cancel_unknown_job(stored: tuple[Path, str]) -> None:
    db, _ = stored
    result = keepout("jobs", "cancel", "99", "--store", str(db))
    assert result.exit_code == ExitCode.ERROR
    assert result.stderr == "Error: no job 99 in the store\n"


def test_jobs_need_a_store(tmp_path: Path) -> None:
    missing = tmp_path / "missing.db"
    for args in (("jobs", "list"), ("worker", "--until-idle")):
        result = keepout(*args, "--store", str(missing))
        assert result.exit_code == ExitCode.ERROR
        assert result.stderr == f"Error: no store at {missing}\n"
        assert not missing.exists()
    result = keepout("jobs", "list", env={"KEEPOUT_STORE": None})
    assert result.exit_code == ExitCode.USAGE
    assert "--store" in usage_error(result)


def old_store(path: Path) -> None:
    """Make ``path`` a store at revision ``0001``, before the jobs."""
    engine = create_engine(f"sqlite:///{path}")
    try:
        with engine.begin() as connection:
            command.upgrade(store_db._config(connection), "0001")
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    "args",
    [
        ("worker", "--until-idle"),
        ("jobs", "list"),
        ("jobs", "cancel", "1"),
        ("maneuver", "--event", "deadbeef", "--queue"),
    ],
    ids=["worker", "list", "cancel", "queue"],
)
def test_worker_on_old_schema(tmp_path: Path, args: tuple[str, ...]) -> None:
    db = tmp_path / "keepout.db"
    old_store(db)
    before = db.read_bytes()
    result = keepout(*args, "--store", str(db))
    assert result.exit_code == ExitCode.ERROR
    assert result.stderr.startswith("Error: ")
    assert "keepout store upgrade" in result.stderr
    assert db.read_bytes() == before


@pytest.mark.timeout(60)
def test_worker_on_an_empty_queue(stored: tuple[Path, str]) -> None:
    db, _ = stored
    result = keepout("worker", "--store", str(db), "--until-idle")
    assert (result.exit_code, result.stdout) == (ExitCode.OK, "Ran 0 jobs.\n")


@pytest.mark.timeout(60)
def test_worker_gives_its_workers_to_the_computation(
    stored: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    db, event_id = stored
    job = queued_job(db, event_id)
    monkeypatch.setattr(planning, "run_job", workers_given)
    result = keepout("worker", "--store", str(db), "--until-idle", "--workers", "3")
    assert (result.exit_code, result.stdout) == (ExitCode.OK, "Ran 1 job.\n")
    with open_store(db, create=False) as store:
        key = JobQueue(store).job(job).key
        with store.read() as session:
            cached = session.scalars(
                select(models.Result.payload).where(models.Result.key == key)
            ).one()
    assert cached == {"workers": 3}


@pytest.fixture
def restored_signals() -> Iterator[None]:
    """SIGTERM's and SIGINT's handlers, checked to be as they were after."""
    before = (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT))
    yield
    after = (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT))
    signal.signal(signal.SIGTERM, before[0])
    signal.signal(signal.SIGINT, before[1])
    assert after == before


@pytest.mark.skipif(not hasattr(os, "kill"), reason="POSIX signals")
@pytest.mark.timeout(60)
@pytest.mark.usefixtures("restored_signals")
@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT], ids=["term", "int"])
def test_a_signal_stops_the_worker_and_releases_its_job(
    stored: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
    signum: signal.Signals,
) -> None:
    db, event_id = stored
    job = queued_job(db, event_id)
    monkeypatch.setattr(planning, "run_job", ready_then_sleep)
    ready = db.with_name("ready")

    def signal_when_ready() -> None:
        # Never to the test run itself unless the job is running: without
        # it, the worker runs on until the test's timeout fails it.
        end = time.monotonic() + 10.0
        while not ready.exists():
            if time.monotonic() > end:
                return
            time.sleep(0.02)
        os.kill(os.getpid(), signum)

    sender = threading.Thread(target=signal_when_ready)
    sender.start()
    try:
        result = keepout("worker", "--store", str(db))
    finally:
        sender.join()
    assert ready.exists()
    assert (result.exit_code, result.stdout) == (ExitCode.OK, "Ran 0 jobs.\n")
    assert f"job {job} given back to the queue" in result.stderr
    listed = json.loads(keepout("jobs", "list", "--store", str(db), "--json").stdout)
    assert [(j["job"], j["status"], j["attempt"]) for j in listed["jobs"]] == [
        (job, "queued", 0)
    ]


def test_worker_help_names_its_options() -> None:
    result = keepout("worker", "--help", env={"COLUMNS": "200"})
    assert result.exit_code == ExitCode.OK
    words = " ".join(result.stdout.split())
    for expected in ("--store FILE", "--until-idle", "--workers"):
        assert expected in words
