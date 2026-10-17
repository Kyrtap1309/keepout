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
directories mirror the ``keepout`` subpackages (``cdm``, ``cli``, ``omm``, ``pc``,
``propagation``, ``risk``, ``screening``, ``sources``, ``synthetic``,
``validation``) and ``tools``.
"""

import os
import urllib.request
from collections.abc import Iterator

import pytest
from astropy.utils import iers
from hypothesis import settings

from keepout.sources import cache

settings.register_profile("default")
settings.register_profile("ci", derandomize=True, print_blob=True, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))

MAX_AUTO_WORKERS = 4


def pytest_xdist_auto_num_workers(config: pytest.Config) -> int | None:
    """Cap ``-n auto`` at ``MAX_AUTO_WORKERS``: a worker per CPU runs out of memory.

    ``PYTEST_XDIST_AUTO_NUM_WORKERS`` still sets the number when given.
    """
    if os.environ.get("PYTEST_XDIST_AUTO_NUM_WORKERS"):
        return None
    return min(MAX_AUTO_WORKERS, os.cpu_count() or 1)


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
    """Let the validation tests download their data: every other test runs
    with sockets disabled (``--disable-socket``, pytest-socket)."""
    for item in items:
        if item.get_closest_marker("validation") is not None:
            item.add_marker(pytest.mark.enable_socket)


@pytest.fixture(autouse=True)
def no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail a test that reaches the network; tests that fake it patch again."""
    if request.node.get_closest_marker("validation") is not None:
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
    cache.forget_verified()
    yield
    cache.forget_verified()
