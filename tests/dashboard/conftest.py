"""Fixtures of the dashboard's tests: the seed store and the app run by
Streamlit's AppTest on a page of it (helpers in ``dashboard_builders``).

The seed store (``tools/dashboard_seed.py``) is made once per session and is
read-only: a test that writes to a store takes :func:`seed_copy`.
"""

import shutil
import sys
from collections.abc import Iterator
from pathlib import Path

import dashboard_seed as seed
import pytest
import streamlit as st
from dashboard_builders import OpenApp, open_app, use_store

from keepout.store import Job, JobQueue


@pytest.fixture(autouse=True)
def _main_module_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    # AppTest leaves the last script it ran as sys.modules["__main__"]; a
    # later test's spawned process pool would run that script on start and
    # break. Put the test runner's back after each test.
    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])


@pytest.fixture(autouse=True)
def _caches_cleared() -> Iterator[None]:
    # The app's caches are the process's: a store opened, settings read or
    # a figure made by one test is not another's. Clearing closes the
    # stores.
    yield
    st.cache_resource.clear()
    st.cache_data.clear()


@pytest.fixture(scope="session")
def seed_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The seed store, read-only (12 events, 7 of them upcoming at
    :func:`dashboard_seed.seed_as_of`)."""
    return seed.seed_store(tmp_path_factory.mktemp("seed") / "seed.db")


@pytest.fixture
def seed_copy(seed_db: Path, tmp_path: Path) -> Path:
    """A copy of the seed store the test may write to."""
    return Path(shutil.copy2(seed_db, tmp_path / "seed.db"))


@pytest.fixture
def app_test(seed_db: Path, monkeypatch: pytest.MonkeyPatch) -> OpenApp:
    """The app on the seed store, seen at the seed's time
    (:func:`dashboard_builders.open_app`).

    The variables ``keepout dashboard`` passes are set for the test; a test
    may change them (``monkeypatch.setenv``) before opening the app.
    """
    use_store(monkeypatch, seed_db, seed.SETTINGS, seed.seed_as_of())
    return open_app


@pytest.fixture
def worker_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every cancel of a job comes after a worker's: the worker ends the
    job between the page's read and the click."""
    cancel = JobQueue.cancel

    def cancelled_first(queue: JobQueue, job: int) -> Job:
        cancel(queue, job)
        return cancel(queue, job)

    monkeypatch.setattr(JobQueue, "cancel", cancelled_first)
