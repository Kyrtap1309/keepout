import gzip
import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from spacetrack_builders import (
    CREDENTIALS,
    HOUR,
    FakeSpaceTrack,
    public_cdm,
    space_track_cdm_public,
)
from web_builders import Clock

from keepout.errors import RateLimited, SourceFormatError
from keepout.sources.cache import read_copy
from keepout.sources.spacetrack import (
    PUBLIC_CDM_COLUMNS,
    PUBLIC_CDMS,
    PublicCdm,
    PublicCdmObject,
    SpaceTrackClient,
    fetch_public_cdms,
    read_public_cdms,
)

ROW = public_cdm()
ANSWER = space_track_cdm_public(ROW, public_cdm(CDM_ID="1000002", PC="0"))


def test_the_query_asks_for_every_approach_to_come() -> None:
    assert PUBLIC_CDMS.url == (
        "https://www.space-track.org/basicspacedata/query/class/cdm_public/"
        "TCA/%3Enow/orderby/TCA%20asc/format/json"
    )
    assert PUBLIC_CDMS.interval == 8 * HOUR


def test_the_columns_are_the_classes_sixteen() -> None:
    assert len(PUBLIC_CDM_COLUMNS) == 16
    assert set(PUBLIC_CDM_COLUMNS) == set(ROW)


def test_a_row_reads_in_si_units() -> None:
    (cdm,) = read_public_cdms(space_track_cdm_public(ROW))
    assert cdm.cdm_id == 1000001
    assert cdm.created.isot == "2026-10-15T06:12:44.000"
    assert cdm.created.scale == "utc"
    assert cdm.emergency_reportable is True
    assert cdm.tca.isot == "2026-10-17T03:25:31.250"
    assert cdm.min_range == 640.0
    assert cdm.probability == 2.5e-5
    assert cdm.object1 == PublicCdmObject(
        900, "CALSPHERE 1", "PAYLOAD", "MEDIUM", "5.00"
    )
    assert cdm.object2 == PublicCdmObject(
        25544, "ISS (ZARYA)", "PAYLOAD", "LARGE", "5.00"
    )


def test_rows_keep_their_order() -> None:
    cdms = read_public_cdms(ANSWER)
    assert [cdm.cdm_id for cdm in cdms] == [1000001, 1000002]
    assert cdms[1].probability == 0.0


def test_an_empty_answer_has_no_rows() -> None:
    assert read_public_cdms(b"[]") == ()


@pytest.mark.parametrize("missing", [None, "", "  "])
def test_optional_values_may_be_missing(missing: str | None) -> None:
    optional = {
        column: missing
        for column in PUBLIC_CDM_COLUMNS
        if column.endswith(("_NAME", "_OBJECT_TYPE", "_RCS", "_EXCL_VOL"))
        or column == "PC"
    }
    assert len(optional) == 9
    (cdm,) = read_public_cdms(space_track_cdm_public(public_cdm(**optional)))
    assert cdm.probability is None
    assert cdm.object1 == PublicCdmObject(900, None, None, None, None)
    assert cdm.object2 == PublicCdmObject(25544, None, None, None, None)


def test_numbers_and_further_columns_are_accepted() -> None:
    row = {**ROW, "CDM_ID": 7, "MIN_RNG": 12.5, "PC": 0.001, "COMMENT": "new"}
    (cdm,) = read_public_cdms(json.dumps([row]).encode())
    assert (cdm.cdm_id, cdm.min_range, cdm.probability) == (7, 12.5, 0.001)


def test_an_approach_not_emergency_reportable_reads_false() -> None:
    row = public_cdm(EMERGENCY_REPORTABLE="N")
    (cdm,) = read_public_cdms(space_track_cdm_public(row))
    assert cdm.emergency_reportable is False


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"\xff[]", "did not send JSON"),
        (b"<html>maintenance</html>", "did not send JSON"),
        (b'{"error": "no"}', "not send an array"),
        (b"[1]", "row 1: not a JSON object"),
        (
            space_track_cdm_public(
                ROW, {k: v for k, v in ROW.items() if k not in {"TCA", "PC"}}
            ),
            "row 2: no TCA, PC$",
        ),
        (space_track_cdm_public(public_cdm(CDM_ID=None)), "CDM_ID is empty"),
        (space_track_cdm_public(public_cdm(TCA="")), "TCA is empty"),
        (space_track_cdm_public(public_cdm(TCA="soon")), "TCA is not an ISO 8601"),
        (
            space_track_cdm_public(public_cdm(CREATED="2026-13-01 00:00:00")),
            "CREATED is not an ISO 8601",
        ),
        (
            space_track_cdm_public(public_cdm(EMERGENCY_REPORTABLE="yes")),
            "must be Y or N",
        ),
        (space_track_cdm_public(public_cdm(MIN_RNG="-1")), "MIN_RNG must be finite"),
        (space_track_cdm_public(public_cdm(MIN_RNG="nan")), "MIN_RNG must be finite"),
        (space_track_cdm_public(public_cdm(MIN_RNG="far")), "could not convert"),
        (space_track_cdm_public(public_cdm(PC="1.5")), r"PC must be in \[0, 1\]"),
        (
            space_track_cdm_public(public_cdm(SAT_2_ID="0")),
            "SAT_2_ID must be at least 1",
        ),
        (space_track_cdm_public(public_cdm(SAT_1_ID="9.5")), "invalid literal"),
        (space_track_cdm_public({**ROW, "SAT_1_NAME": True}), "text or a number"),
        (space_track_cdm_public({**ROW, "SAT_1_NAME": ["A"]}), "text or a number"),
    ],
)
def test_unreadable_answers_are_refused(data: bytes, message: str) -> None:
    with pytest.raises(SourceFormatError, match=message):
        read_public_cdms(data)


@given(
    cdm_id=st.integers(1, 2**32 - 1),
    numbers=st.tuples(st.integers(1, 999_999_999), st.integers(1, 999_999_999)),
    min_range=st.floats(0.0, 1e9),
    probability=st.floats(0.0, 1.0),
)
def test_values_read_back_exactly(
    cdm_id: int, numbers: tuple[int, int], min_range: float, probability: float
) -> None:
    row = public_cdm(
        CDM_ID=str(cdm_id),
        SAT_1_ID=str(numbers[0]),
        SAT_2_ID=str(numbers[1]),
        MIN_RNG=repr(min_range),
        PC=repr(probability),
    )
    (cdm,) = read_public_cdms(space_track_cdm_public(row))
    assert cdm.cdm_id == cdm_id
    assert (cdm.object1.catalog_number, cdm.object2.catalog_number) == numbers
    assert cdm.min_range == min_range
    assert cdm.probability == probability


def test_the_client_fetches_and_caches_the_rows(tmp_path: Path) -> None:
    server = FakeSpaceTrack({PUBLIC_CDMS.url: ANSWER})
    clock = Clock()
    with SpaceTrackClient(
        CREDENTIALS, cache=tmp_path, transport=server, now=clock
    ) as client:
        first = client.fetch_public_cdms()
        clock.now += 8 * HOUR - 1.0
        again = client.fetch_public_cdms()
    assert server.calls == ["login", PUBLIC_CDMS.url, "logout"]
    assert first.cdms == again.cdms == read_public_cdms(ANSWER)
    assert all(isinstance(cdm, PublicCdm) for cdm in first.cdms)
    assert first.retrieval.path == tmp_path / "spacetrack/cdm_public/upcoming.json.gz"
    assert first.retrieval.url == PUBLIC_CDMS.url
    copy = read_copy(first.retrieval.path, PUBLIC_CDMS.url)
    assert copy is not None
    assert gzip.decompress(copy.data) == ANSWER


def test_fetch_public_cdms_waits_eight_hours(tmp_path: Path) -> None:
    server = FakeSpaceTrack({PUBLIC_CDMS.url: ANSWER})
    clock = Clock()
    rows = fetch_public_cdms(
        cache=tmp_path, credentials=CREDENTIALS, transport=server, now=clock
    )
    assert len(rows.cdms) == 2
    rows.retrieval.path.unlink()
    clock.now += 8 * HOUR - 1.0
    with pytest.raises(RateLimited, match="one cdm_public query per 8 h"):
        fetch_public_cdms(
            cache=tmp_path, credentials=CREDENTIALS, transport=server, now=clock
        )
    assert server.calls == ["login", PUBLIC_CDMS.url, "logout"]


def test_an_unreadable_answer_is_not_cached(tmp_path: Path) -> None:
    server = FakeSpaceTrack({PUBLIC_CDMS.url: b'{"error": "no"}'})
    with pytest.raises(SourceFormatError, match="not send an array"):
        fetch_public_cdms(
            cache=tmp_path, credentials=CREDENTIALS, transport=server, now=Clock()
        )
    assert not (tmp_path / "spacetrack/cdm_public/upcoming.json.gz").exists()
