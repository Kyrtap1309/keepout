"""Space-Track's answers as ``tools/spacetrack_snapshot.py download`` saved
them, checked offline on the maintainer's disk.

Public CDMs may not be redistributed without Space-Track's approval, so
they are not committed: these tests run only when the environment variable
``KEEPOUT_SPACETRACK_RAW`` names a ``RAW`` directory (and are skipped
otherwise, as in CI), and they write nothing.
"""

import json
import os
from pathlib import Path

import pytest
import spacetrack_snapshot as snapshot

from keepout.omm import Omm
from keepout.sources.spacetrack import (
    CATALOG,
    LOGIN_URL,
    LOGOUT_URL,
    PUBLIC_CDM_COLUMNS,
    PUBLIC_CDMS,
    PublicCdm,
)

RAW = os.environ.get(snapshot.RAW_VARIABLE, "")

pytestmark = [
    pytest.mark.validation,
    pytest.mark.skipif(
        not RAW, reason=f"set {snapshot.RAW_VARIABLE} to a RAW directory to run"
    ),
]


@pytest.fixture(scope="module")
def raw() -> Path:
    return Path(RAW)


@pytest.fixture(scope="module")
def omms(raw: Path) -> tuple[Omm, ...]:
    return snapshot.replay_catalog(raw).omms


@pytest.fixture(scope="module")
def cdms(raw: Path) -> tuple[PublicCdm, ...]:
    return snapshot.replay_public_cdms(raw)


@pytest.fixture(scope="module")
def summary(
    cdms: tuple[PublicCdm, ...], omms: tuple[Omm, ...]
) -> snapshot.ComparisonSummary:
    return snapshot.summarise(
        snapshot.compare(cdms, omms, workers=min(8, os.cpu_count() or 1))
    )


def test_four_requests_were_sent_and_answered(raw: Path) -> None:
    index = snapshot.INDEX.read(raw)
    assert [(saved.url, saved.status) for saved in index] == [
        (LOGIN_URL, 200),
        (CATALOG.url, 200),
        (PUBLIC_CDMS.url, 200),
        (LOGOUT_URL, 200),
    ]
    assert not any(saved.redacted for saved in index)
    # A successful login answers HTTP 200 with an empty JSON string.
    assert snapshot.saved_answer(raw, LOGIN_URL) == b'""'


def test_every_public_cdm_has_the_sixteen_columns(
    raw: Path, cdms: tuple[PublicCdm, ...]
) -> None:
    rows = json.loads(snapshot.saved_answer(raw, PUBLIC_CDMS.url))
    assert len(rows) == len(cdms) > 0
    assert all(set(row) == set(PUBLIC_CDM_COLUMNS) for row in rows)
    assert all(cdm.created < cdm.tca for cdm in cdms)


def test_min_rng_is_in_metres(
    cdms: tuple[PublicCdm, ...], summary: snapshot.ComparisonSummary
) -> None:
    # keepout's ranges on GP data are within a few hundred metres of MIN_RNG
    # read in metres; read in kilometres, MIN_RNG would be 1000 times larger
    # than both keepout's ranges and the 1 to 5 km exclusion volumes.
    assert all(cdm.range_at_tca <= 5e3 for cdm in cdms)
    assert {cdm.object1.exclusion_volume for cdm in cdms} <= {"1.00", "3.00", "5.00"}
    difference = summary["range_difference"]["median"]
    assert difference is not None
    assert difference < 1e3


def test_keepout_finds_every_approach_at_the_same_time(
    summary: snapshot.ComparisonSummary,
) -> None:
    assert summary["compared"] > 0
    assert summary["found"] == summary["compared"]
    late = summary["tca_difference"]["p90"]
    assert late is not None
    assert late < 1.0
