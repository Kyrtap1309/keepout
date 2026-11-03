"""Downloads of the two snapshot tools from fake servers, for their tests:
``tools/screening_snapshot.py`` from CelesTrak, ``tools/spacetrack_snapshot.py``
from Space-Track."""

import dataclasses
from collections.abc import Mapping
from pathlib import Path

import astropy.units as u
import screening_snapshot
import spacetrack_snapshot
from celestrak_builders import SAMPLE, SOCRATES_HEADER, socrates_row
from spacetrack_builders import (
    CREDENTIALS,
    FakeSpaceTrack,
    public_cdm,
    space_track_cdm_public,
    space_track_gp,
)
from web_builders import FakeDownloader

from keepout.omm.celestrak import dumps_csv
from keepout.sources.celestrak import SOCRATES_URL, group_url
from keepout.sources.spacetrack import CATALOG, PUBLIC_CDMS

CALSPHERE, ISS, GOES, THEMIS, EXPRESS, SARAMAGO, STARLINK, DRAGON = SAMPLE
OLD_ISS = dataclasses.replace(ISS, epoch=ISS.epoch - 1.0 * u.day)
"""An older element set of the ISS: a duplicate the catalog leaves out."""

# CelesTrak

DOCKED = socrates_row(ISS, DRAGON, ("1.000", "1.000"), "0.004", "0.000")
FAST = socrates_row(CALSPHERE, STARLINK, ("0.838", "0.764"), "2.500", "7.500")
SOCRATES_CSV = (
    SOCRATES_HEADER
    + DOCKED
    + socrates_row(CALSPHERE, GOES, ("2.000", "2.000"), "0.100", "3.000")
    + socrates_row(
        ISS,
        dataclasses.replace(ISS, catalog_number=99999),
        ("1.000", "1.000"),
        "0.200",
        "7.500",
    )
    + FAST
)
"""A report: the ISS and Crew Dragon (docked, with the sample's epochs), a
pair whose epochs are not the sample's, a pair with an object not in the
sample, and Calsphere 1 and Starlink-38128 (the sample's epochs: DSE to 43 s)."""

BODIES: Mapping[str, bytes] = {
    group_url("active"): dumps_csv(SAMPLE[:5]).encode(),
    group_url("last-30-days"): dumps_csv([*SAMPLE[5:], OLD_ISS]).encode(),
    group_url("analyst"): dumps_csv(SAMPLE[5:]).encode(),
    group_url("fengyun-1c-debris"): dumps_csv([]).encode(),
    group_url("cosmos-2251-debris"): dumps_csv([]).encode(),
    group_url("iridium-33-debris"): dumps_csv([]).encode(),
    SOCRATES_URL: SOCRATES_CSV.encode(),
}
"""CelesTrak's answers: the sample in the public groups, and the report."""


def celestrak_raw(directory: Path) -> Path:
    """``directory``, holding a download of :data:`BODIES`."""
    screening_snapshot.fetch(directory, FakeDownloader(BODIES))
    return directory


# Space-Track

CROSSING = dataclasses.replace(
    ISS,
    catalog_number=99999,
    object_name="CROSSING",
    ra_of_asc_node=ISS.ra_of_asc_node + 0.01,
)
"""The ISS's orbit turned 0.01 rad about the Earth's axis: it passes the ISS
at about 42 km every 46 minutes."""
GP = space_track_gp([*SAMPLE, CROSSING, OLD_ISS])
PASSING = public_cdm(
    SAT_1_ID="25544",
    SAT_2_ID="99999",
    TCA="2026-10-06T00:44:04.056973",
    MIN_RNG="42000",
)
"""keepout's first approach of the ISS and CROSSING, 2 s later and 195 m
closer."""
CDMS = space_track_cdm_public(
    PASSING,
    public_cdm(CDM_ID="2", SAT_2_ID="123456"),
    public_cdm(CDM_ID="3", SAT_1_ID="900", SAT_2_ID="25544"),
)
"""The ISS and CROSSING; an object not in the catalog; Calsphere 1 and the
ISS, never within 100 km of each other."""
START = 1_791_000_000.0
"""A Unix time in October 2026."""


def space_track() -> FakeSpaceTrack:
    """Space-Track answering :data:`GP` and :data:`CDMS`."""
    return FakeSpaceTrack({CATALOG.url: GP, PUBLIC_CDMS.url: CDMS})


def spacetrack_download(
    raw: Path, server: FakeSpaceTrack, cache: Path
) -> tuple[spacetrack_snapshot.Saved, ...]:
    """The tool's download from ``server`` into ``raw``, at :data:`START`."""
    return spacetrack_snapshot.download(
        raw,
        cache=cache,
        credentials=CREDENTIALS,
        transport=server,
        clock=lambda: START,
    )


def spacetrack_raw(directory: Path) -> Path:
    """``directory``, holding a download from :func:`space_track`."""
    spacetrack_download(directory, space_track(), directory.with_name("cache"))
    return directory
