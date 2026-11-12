"""``keepout jobs`` and ``keepout worker``: the store's background jobs on the
command line. The worker's computations are those of ``worker_jobs``, run in
child processes as the real one is."""

import json
import os
import signal
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from cli_builders import keepout, operator_kvn, usage_error, write
from sqlalchemy import create_engine
from worker_jobs import ready_then_sleep

from keepout._workers import ONE_THREAD
from keepout.cli import worker as worker_cli
from keepout.cli.output import ExitCode
from keepout.store import db as store_db


@pytest.fixture(autouse=True)
def one_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """The worker sets :data:`ONE_THREAD` in this process: undone after."""
    for name, value in ONE_THREAD.items():
        monkeypatch.setenv(name, os.environ.get(name, value))


@pytest.fixture
def stored(tmp_path: Path) -> tuple[Path, str]:
    """A store of the operator's CDM of the crossing event, and its id."""
    cdm = write(tmp_path, "crossing.kvn", operator_kvn()[0])
    db = tmp_path / "keepout.db"
    added = keepout("store", "add", cdm, "--store", str(db), "--json")
    (event_id,) = json.loads(added.stdout)["store"]["events"]
    return db, event_id


def queue(db: Path, event_id: str, *options: str) -> int:
    """The id of the job ``maneuver --queue`` queued."""
    stored = ("--event", event_id, "--store", str(db), "--hbr", "1000")
    result = keepout("maneuver", *stored, "--queue", *options, "--json")
    assert result.exit_code == ExitCode.OK
    job = json.loads(result.stdout)["job"]
    assert isinstance(job, int)
    return job


def test_list_shows_the_jobs_newest_first(stored: tuple[Path, str]) -> None:
    db, event_id = stored
    empty = keepout("jobs", "list", "--store", str(db))
    assert (empty.exit_code, empty.stdout) == (ExitCode.OK, f"No jobs in {db}.\n")
    first = queue(db, event_id)
    second = queue(db, event_id, "--target-pc", "1e-6")
    listed = keepout("jobs", "list", "--store", str(db), env={"COLUMNS": "200"})
    assert listed.exit_code == ExitCode.OK
    header = listed.stdout.splitlines()[1].split()
    assert header == [
        "Job",
        "Event",
        "CDM",
        "Status",
        "Attempt",
        "Created",
        "(UTC)",
        "Error",
    ]
    rows = [line.split() for line in listed.stdout.splitlines()[3:]]
    assert [(row[0], row[1], row[3], row[4]) for row in rows] == [
        (str(second), event_id, "queued", "0"),
        (str(first), event_id, "queued", "0"),
    ]
    payload = json.loads(keepout("jobs", "list", "--store", str(db), "--json").stdout)
    assert payload["path"] == str(db)
    (newest, oldest) = payload["jobs"]
    assert set(newest) == {
        "job",
        "event",
        "cdm",
        "status",
        "attempt",
        "worker",
        "created",
        "started",
        "finished",
        "error",
    }
    assert (newest["job"], oldest["job"]) == (second, first)
    assert newest["started"] is None


def test_cancel(stored: tuple[Path, str]) -> None:
    db, event_id = stored
    job = queue(db, event_id)
    cancelled = keepout("jobs", "cancel", str(job), "--store", str(db))
    assert cancelled.exit_code == ExitCode.OK
    assert cancelled.stdout == f"Cancelled job {job} (event {event_id}).\n"
    again = keepout("jobs", "cancel", str(job), "--store", str(db))
    assert again.exit_code == ExitCode.ERROR
    assert again.stderr == f"Error: job {job} has already ended: cancelled\n"
    payload = json.loads(keepout("jobs", "list", "--store", str(db), "--json").stdout)
    assert payload["jobs"][0]["status"] == "cancelled"
    # A cancelled job is not one to wait for: queued again, a new job.
    assert queue(db, event_id) == job + 1


def test_cancel_json(stored: tuple[Path, str]) -> None:
    db, event_id = stored
    job = queue(db, event_id)
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


def test_worker_on_an_empty_queue(stored: tuple[Path, str]) -> None:
    db, _ = stored
    result = keepout("worker", "--store", str(db), "--until-idle")
    assert (result.exit_code, result.stdout) == (ExitCode.OK, "Ran 0 jobs.\n")


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
@pytest.mark.usefixtures("restored_signals")
@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT], ids=["term", "int"])
def test_a_signal_stops_the_worker_and_releases_its_job(
    stored: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
    signum: signal.Signals,
) -> None:
    db, event_id = stored
    job = queue(db, event_id)
    monkeypatch.setattr(worker_cli, "run_maneuver_job", ready_then_sleep)
    ready = db.with_name("ready")

    def signal_when_ready() -> None:
        end = time.monotonic() + 10.0
        while not ready.exists() and time.monotonic() < end:
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
