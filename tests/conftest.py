"""Test-suite configuration: tests must not reach the network.

Only tests marked ``validation`` (which download the NASA CARA reference data
or ESA's Kelvins data on first run) may; the others fail on any
``urllib.request.urlopen`` and on any HTTP or HTTPS request of a
``urllib.request`` opener (the Space-Track client's cookie session), and
astropy never downloads IERS tables (nor refuses the bundled one for its
age).

Hypothesis profiles: ``default`` (random examples, the local database) and
``ci`` (derandomized, so a CI failure reproduces exactly, and printing the
blob to ``@reproduce_failure``); pick one with ``HYPOTHESIS_PROFILE``.

Layout: ``geometry/`` tests the top-level domain modules (state, objects,
covariance, frames, orbits, equinoctial, encounter, conjunction); the other
directories mirror the ``keepout`` subpackages (``cdm``, ``cli``, ``maneuver``,
``omm``, ``pc``, ``propagation``, ``risk``, ``screening``, ``sources``, ``synthetic``,
``validation``) and ``tools``.
"""

import os
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest
from astropy.utils import iers
from hypothesis import settings
from web_builders import Clock

from keepout.sources import _store
from keepout.store import Store, computing, open_store

# The CLI tests read plain text: Typer (on import) and Rich take these to
# mean a colour terminal, and GitHub Actions sets GITHUB_ACTIONS.
for _name in ("GITHUB_ACTIONS", "FORCE_COLOR", "PY_COLORS"):
    os.environ.pop(_name, None)

settings.register_profile("default", deadline=None)
settings.register_profile("ci", derandomize=True, print_blob=True, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))

NETWORKED = ("validation", "e2e")
"""The markers of the tests that may use the network."""

MAX_AUTO_WORKERS = 4


def pytest_xdist_auto_num_workers(config: pytest.Config) -> int | None:
    """Cap ``-n auto`` at ``MAX_AUTO_WORKERS``: a worker per CPU runs out of memory.

    ``PYTEST_XDIST_AUTO_NUM_WORKERS`` still sets the number when given.
    """
    if os.environ.get("PYTEST_XDIST_AUTO_NUM_WORKERS"):
        return None
    return min(MAX_AUTO_WORKERS, os.cpu_count() or 1)


@pytest.fixture(autouse=True)
def _no_store_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # A store exported in the shell must not be read or written by a test;
    # a test that needs $KEEPOUT_STORE sets it.
    monkeypatch.delenv("KEEPOUT_STORE", raising=False)
    monkeypatch.delenv("KEEPOUT_HEARTBEAT_FILE", raising=False)


@pytest.fixture(autouse=True, scope="session")
def _offline_iers() -> Iterator[None]:
    # Without downloads the bundled table ages; refusing its predictions
    # after 30 days would make the suite fail by the calendar.
    with (
        iers.conf.set_temp("auto_download", False),
        iers.conf.set_temp("auto_max_age", None),
    ):
        yield


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Let the validation tests download their data and the e2e tests drive
    a browser: every other test runs with sockets disabled
    (``--disable-socket``, pytest-socket)."""
    for item in items:
        if _networked(item):
            item.add_marker(pytest.mark.enable_socket)


def _networked(item: pytest.Item) -> bool:
    """Whether ``item`` may use the network: a validation or e2e test."""
    return any(item.get_closest_marker(name) is not None for name in NETWORKED)


@pytest.fixture(autouse=True)
def no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail a test that reaches the network; tests that fake it patch again."""
    if _networked(request.node):
        return

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("tests must not access the network; pass download=")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    monkeypatch.setattr(urllib.request.HTTPHandler, "http_open", refuse)
    monkeypatch.setattr(urllib.request.HTTPSHandler, "https_open", refuse)


@pytest.fixture(autouse=True)
def _forget_verified() -> Iterator[None]:
    """Start and end every test without the process-wide memo of verified
    files."""
    _store.forget_verified()
    yield
    _store.forget_verified()


@pytest.fixture(autouse=True)
def _forget_parsed_cdms() -> Iterator[None]:
    """Start and end every test without the process-wide cache of parsed
    store CDMs, so that a test counting parses sees them all."""
    computing.parsed_cdm.cache_clear()
    yield
    computing.parsed_cdm.cache_clear()


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    """A new store, ``keepout.db`` in the test's folder."""
    with open_store(tmp_path / "keepout.db") as opened:
        yield opened


@pytest.fixture
def clock() -> Clock:
    """A settable Unix time, at :data:`web_builders.START` (the sources'
    clients take it as ``now``)."""
    return Clock()
