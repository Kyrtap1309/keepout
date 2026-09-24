"""Test-suite configuration: astropy must not reach the network."""

from collections.abc import Iterator

import pytest
from astropy.utils import iers


@pytest.fixture(autouse=True, scope="session")
def _offline_iers() -> Iterator[None]:
    with iers.conf.set_temp("auto_download", False):
        yield
