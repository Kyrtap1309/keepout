"""Fixtures of the dashboard's tests: the seed store and the app run by
Streamlit's AppTest on a page of it (helpers in ``dashboard_builders``).

The seed store (``tools/dashboard_seed.py``) is made once per session and is
read-only: a test that writes to a store takes :func:`seed_copy`.
"""

import shutil
import sys
from pathlib import Path

import dashboard_seed as seed
import pytest
from dashboard_builders import OpenApp, open_app


@pytest.fixture(autouse=True)
def _main_module_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    # AppTest leaves the last script it ran as sys.modules["__main__"]; a
    # later test's spawned process pool would run that script on start and
    # break. Put the test runner's back after each test.
    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])


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
    monkeypatch.setenv("KEEPOUT_STORE", str(seed_db))
    monkeypatch.setenv("KEEPOUT_SETTINGS", str(seed.SETTINGS))
    monkeypatch.setenv("KEEPOUT_AS_OF", str(seed.seed_as_of().utc.isot))
    return open_app
