"""Test-suite configuration: astropy must not reach the network.

Layout: ``geometry/`` tests the top-level domain modules (state, objects,
covariance, frames, orbits, equinoctial, encounter, conjunction); the other
directories mirror the ``keepout`` subpackages (``cdm``, ``pc``, ``sources``,
``validation``) and ``tools``.
"""

from collections.abc import Iterator

import pytest
from astropy.utils import iers


@pytest.fixture(autouse=True, scope="session")
def _offline_iers() -> Iterator[None]:
    with iers.conf.set_temp("auto_download", False):
        yield
