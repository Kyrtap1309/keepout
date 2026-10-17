"""The committed screening data in ``tests/screening/data/``, for tests.

``tools/screening_snapshot.py`` writes these files (see its docstring);
tests read them here, without importing the tool.
"""

import functools
import gzip
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final

from astropy.time import Time

from keepout.omm import Omm, load
from keepout.sources.socrates import SocratesApproach, read_socrates

DATA: Final = Path(__file__).parent / "screening" / "data"
"""The directory of the committed data."""

SNAPSHOT: Final = "celestrak-public.csv.gz"
"""The public catalog, one element set per object, gzipped CelesTrak CSV."""

SOCRATES: Final = "socrates-excerpt.csv.gz"
"""The SOCRATES approaches whose element sets are in the catalog, gzipped."""

FIXTURE: Final = "celestrak-fixture.csv"
"""About 200 objects of the catalog, CelesTrak CSV."""

MANIFEST: Final = "manifest.json"
"""Downloads, counts and digests of the other files."""

FAST_SPEED: Final = 1_000.0
"""Least relative speed of a fast approach [m/s], whose TCA SOCRATES gives
to the millisecond; slower ones (formation flyers) are ill-conditioned."""


FIXTURE_START: Final = Time("2026-10-06T00:00:00", scale="utc")
"""Midnight on the day of the catalog: after most of :data:`FIXTURE`'s element
epochs, the start of the tests' screening windows."""

SGP4_FAILING: Final = (45665, 45711, 270365)
"""The objects of :data:`FIXTURE` SGP4 fails on within a week of
:data:`FIXTURE_START`: two decaying Starlinks and a reentering object."""


@functools.cache
def load_fixture() -> tuple[Omm, ...]:
    """The element sets of :data:`FIXTURE`, by catalog number."""
    return load(DATA / FIXTURE)


@functools.cache
def fixture_by_number() -> Mapping[int, Omm]:
    """The element sets of :data:`FIXTURE` by catalog number."""
    return MappingProxyType({omm.catalog_number: omm for omm in load_fixture()})


def socrates_text() -> str:
    """The SOCRATES excerpt: the report's header and lines, unchanged."""
    return gzip.decompress((DATA / SOCRATES).read_bytes()).decode("utf-8")


def socrates_approaches() -> tuple[SocratesApproach, ...]:
    """The approaches of the SOCRATES excerpt, in report order."""
    return read_socrates(socrates_text())
