"""Test-suite configuration: tests must not reach the network.

Only tests marked ``validation`` (which download the NASA CARA reference data
or ESA's Kelvins data on first run) may; the others fail on any
``urllib.request.urlopen``, and astropy never downloads IERS tables.

Hypothesis profiles: ``default`` (random examples, the local database) and
``ci`` (derandomized, so a CI failure reproduces exactly, and printing the
blob to ``@reproduce_failure``); pick one with ``HYPOTHESIS_PROFILE``.

Layout: ``geometry/`` tests the top-level domain modules (state, objects,
covariance, frames, orbits, equinoctial, encounter, conjunction); the other
directories mirror the ``keepout`` subpackages (``cdm``, ``omm``, ``pc``,
``propagation``, ``risk``, ``screening``, ``sources``, ``validation``) and
``tools``.
"""

import os
import urllib.request
from collections.abc import Iterator

import pytest
from astropy.utils import iers
from hypothesis import settings

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
    with iers.conf.set_temp("auto_download", False):
        yield


@pytest.fixture(autouse=True)
def no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail a test that reaches the network; tests that fake it patch again."""
    if request.node.get_closest_marker("validation") is not None:
        return

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("tests must not access the network; pass download=")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
