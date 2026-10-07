"""The committed screening data in ``tests/screening/data/``, for tests.

``tools/screening_snapshot.py`` writes these files (see its docstring);
tests read them here, without importing the tool.
"""

import gzip
from pathlib import Path
from typing import Final

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


def load_fixture() -> tuple[Omm, ...]:
    """The element sets of :data:`FIXTURE`, by catalog number."""
    return load(DATA / FIXTURE)


def socrates_text() -> str:
    """The SOCRATES excerpt: the report's header and lines, unchanged."""
    return gzip.decompress((DATA / SOCRATES).read_bytes()).decode("utf-8")


def socrates_approaches() -> tuple[SocratesApproach, ...]:
    """The approaches of the SOCRATES excerpt, in report order."""
    return read_socrates(socrates_text())
