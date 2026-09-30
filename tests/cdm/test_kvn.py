import dataclasses
from pathlib import Path

import numpy as np
import pytest
from builders import EARTH_ROTATION_RATE, known
from cdm_builders import B1_KVN as B1
from cdm_builders import TRACSS, assert_equivalent, cdms, minimal_cdm
from hypothesis import given, settings

from keepout.cdm import Cdm, Field, Section, kvn
from keepout.cdm.keywords import format_epoch
from keepout.encounter import encounter_plane
from keepout.errors import CdmParseError, CdmWriteError, CovarianceError


def mutate(old: str, new: str) -> str:
    text = B1.read_text(encoding="utf-8")
    assert text.count(old) == 1, old
    return text.replace(old, new)


def line_of(text: str, fragment: str) -> int:
    return next(n for n, line in enumerate(text.splitlines(), 1) if fragment in line)


def test_reads_b1_example() -> None:
    cdm = kvn.load(B1)
    assert cdm.version == "1.0"
    assert cdm.originator == "JSPOC"
    assert cdm.message_for == "SATELLITE A"
    assert cdm.message_id == "201113719185"
    assert cdm.header.comments == (
        "keepout test fixture modelled on the CCSDS 508.0-B-1 annex example",
    )
    assert format_epoch(cdm.tca) == "2010-03-13T22:37:52.618"
    assert cdm.miss_distance == 715.0
    assert cdm.collision_probability == pytest.approx(4.835e-05)
    assert cdm.relative.get("RELATIVE_SPEED") == 14762.0
    first = cdm.object1
    assert first.name == "SATELLITE A"
    assert first.section.get("OBS_USED") == 579
    assert first.section.get("RECOMMENDED_OD_SPAN") == pytest.approx(7.88 * 86400)
    assert first.section.get("GRAVITY_MODEL") == "EGM-96: 36D 36O"
    assert first.section.comments == ("Covariance derived from the operator's OD",)
    np.testing.assert_allclose(first.position, [2570097.065, 2244654.904, 6281497.978])
    np.testing.assert_allclose(first.velocity, [4418.769571, 4833.547743, -3526.774282])
    assert first.covariance_rtn[1, 0] == first.covariance_rtn[0, 1] == -8.579
    assert cdm.object2.name == "FENGYUN 1C DEB"


def test_b1_example_geometry() -> None:
    primary, secondary = kvn.load(B1).object_states()
    plane = encounter_plane(primary, secondary)
    relative = secondary.state.r - primary.state.r
    assert plane.miss_distance == pytest.approx(
        float(np.linalg.norm(relative)), abs=0.01
    )
    # Independent check: the message's own RTN relative position (given to
    # 0.1 m; the states are 0.5 m off closest approach).
    reported = np.linalg.norm([27.4, -70.2, 711.8])
    assert plane.miss_distance == pytest.approx(reported, abs=0.1)


def test_reads_tracss_cdm_version_2() -> None:
    cdm = kvn.load(TRACSS)
    assert cdm.version == "2.0"
    assert cdm.header.extras == {"CLASSIFICATION": "UNCLASSIFIED"}
    # TraCSS puts CONJUNCTION_ID in the header, before TCA.
    assert cdm.conjunction_id == "dd8c054b-6bea-48fb-a245-6cb23331b156"
    assert cdm.relative.get("CONJUNCTION_ID") is None
    first = cdm.object1
    assert first.ref_frame == "ITRF"
    assert first.hbr == 5.0
    assert cdm.object2.hbr is None
    assert first.user_defined == {
        "USER_DEFINED_RUN_ID": "8e815679-b659-48df-957e-9c5b4921af52",
        "USER_DEFINED_MEETS_ALERTABLE_CRITERIA": "NO",
    }
    assert (
        first.section.extras["DENSITY_FORECAST_UNCERTAINTY"] == "2.046460670000000E-01"
    )
    assert format_epoch(cdm.tca) == "2025-06-02T08:15:30.123456"


def test_tracss_itrf_states_are_converted_to_gcrf() -> None:
    primary, _ = kvn.load(TRACSS).object_states()
    assert known(primary.covariance).repairs == ()
    r = np.array([6778137.0, 0.0, 0.0])
    v = np.array([0.0, 5900.0, 4400.0])
    inertial = v + np.cross([0.0, 0.0, EARTH_ROTATION_RATE], r)
    assert primary.state.frame == "GCRF"
    assert np.linalg.norm(primary.state.r) == pytest.approx(np.linalg.norm(r), abs=1e-3)
    assert np.linalg.norm(primary.state.v) == pytest.approx(
        np.linalg.norm(inertial), abs=1e-2
    )
    assert primary.hard_body_radius == 5.0


@pytest.mark.parametrize("path", [B1, TRACSS])
def test_fixture_round_trip(path: Path) -> None:
    cdm = kvn.load(path)
    assert_equivalent(kvn.loads(kvn.dumps(cdm)), cdm)


@settings(max_examples=50, deadline=None)
@given(cdms())
def test_round_trip_property(cdm: Cdm) -> None:
    assert_equivalent(kvn.loads(kvn.dumps(cdm)), cdm)


def test_writer_uses_ccsds_units() -> None:
    text = kvn.dumps(kvn.load(B1))
    lines = text.splitlines()
    assert lines[0].startswith("CCSDS_CDM_VERS")
    assert any(
        line.startswith("MISS_DISTANCE") and line.endswith("= 715.0 [m]")
        for line in lines
    )
    assert any(line.startswith("X ") and line.endswith(" [km]") for line in lines)
    assert "COMMENT Covariance derived from the operator's OD" in lines


def test_bad_unit() -> None:
    text = mutate("2570.097065 [km]", "2570.097065 [m]")
    with pytest.raises(
        CdmParseError, match=r"unit \[m\] does not match expected \[km\]"
    ) as info:
        kvn.loads(text)
    assert info.value.keyword == "X"
    assert info.value.line == line_of(text, "2570.097065 [m]")


def test_bad_number() -> None:
    with pytest.raises(CdmParseError, match="invalid number 'seven'") as info:
        kvn.loads(mutate("= 715 [m]", "= seven [m]"))
    assert info.value.keyword == "MISS_DISTANCE"


def test_bad_epoch() -> None:
    with pytest.raises(CdmParseError, match="invalid CCSDS epoch") as info:
        kvn.loads(mutate("2010-03-13T22:37:52.618", "2010-13-40T22:37:52.618"))
    assert info.value.keyword == "TCA"


def test_malformed_line() -> None:
    text = mutate("MISS_DISTANCE ", "MISS DISTANCE ")
    with pytest.raises(CdmParseError, match="expected 'KEYWORD = value'") as info:
        kvn.loads(text)
    assert info.value.line == line_of(text, "MISS DISTANCE")


def test_duplicate_keyword() -> None:
    text = mutate("= 715 [m]\n", "= 715 [m]\nMISS_DISTANCE = 716 [m]\n")
    with pytest.raises(CdmParseError, match="duplicate keyword") as info:
        kvn.loads(text)
    assert info.value.keyword == "MISS_DISTANCE"
    assert info.value.line == line_of(text, "= 716 [m]")


def test_version_must_come_first() -> None:
    text = B1.read_text(encoding="utf-8").split("\n", 1)[1]
    with pytest.raises(
        CdmParseError, match="CCSDS_CDM_VERS must be the first keyword"
    ) as info:
        kvn.loads(text)
    assert info.value.line == 1


def test_missing_required_keyword() -> None:
    lines = B1.read_text(encoding="utf-8").splitlines()
    last = max(i for i, line in enumerate(lines) if line.startswith("CNDOT_NDOT"))
    text = "\n".join(lines[:last] + lines[last + 1 :])
    with pytest.raises(
        CdmParseError,
        match=r"OBJECT2 section is missing required keyword\(s\): CNDOT_NDOT",
    ):
        kvn.loads(text)


def test_empty_required_value_is_rejected() -> None:
    text = mutate("= JSPOC\n", "=\n")
    with pytest.raises(
        CdmParseError,
        match=r"header section is missing required keyword\(s\): ORIGINATOR",
    ):
        kvn.loads(text)


def test_third_object_is_rejected() -> None:
    text = B1.read_text(encoding="utf-8") + "OBJECT = OBJECT3\n"
    with pytest.raises(CdmParseError, match="more than two object sections"):
        kvn.loads(text)


def test_object_before_relative_metadata_is_rejected() -> None:
    with pytest.raises(CdmParseError, match="object section before relative metadata"):
        kvn.loads("CCSDS_CDM_VERS = 1.0\nOBJECT = OBJECT1\n")


@pytest.mark.parametrize("text", ["", "\n   \n"])
def test_empty_message_is_rejected(text: str) -> None:
    with pytest.raises(CdmParseError, match="a CDM needs a header"):
        kvn.loads(text)


def test_objects_must_be_labelled_in_order() -> None:
    with pytest.raises(CdmParseError, match="segment 2 must have OBJECT = OBJECT2"):
        kvn.loads(mutate("= OBJECT2", "= OBJECT1"))


def test_unknown_enumeration_values_are_kept() -> None:
    cdm = kvn.loads(mutate("= PAYLOAD", "= SPACE STATION MODULE"))
    assert cdm.object1.section.get("OBJECT_TYPE") == "SPACE STATION MODULE"


def test_non_psd_covariance_parses_and_is_repaired() -> None:
    cdm = kvn.loads(mutate("-8.579E+00", "-8.579E+04"))
    with pytest.raises(CovarianceError):
        cdm.object1.covariance(policy="raise")
    first, _ = cdm.object_states()
    assert known(first.covariance).repairs[0].startswith("clipped")


def test_file_round_trip_with_bom(tmp_path: Path) -> None:
    source = tmp_path / "bom.kvn"
    source.write_bytes(b"\xef\xbb\xbf" + B1.read_bytes())
    cdm = kvn.load(source)
    target = tmp_path / "out.kvn"
    kvn.dump(cdm, target)
    assert_equivalent(kvn.load(target), cdm)


def test_comments_before_tca_and_object_open_the_new_section() -> None:
    text = mutate("TCA ", "COMMENT screening option A\nTCA ")
    text = text.replace(
        "OBJECT                         = OBJECT2",
        "COMMENT about object 2\nOBJECT                         = OBJECT2",
    )
    cdm = kvn.loads(text)
    assert cdm.relative.comments == ("screening option A",)
    assert cdm.segments[1].comments == ("about object 2",)
    assert "screening option A" not in cdm.header.comments
    assert "about object 2" not in cdm.segments[0].comments


def test_trailing_comments_stay_in_the_last_section() -> None:
    cdm = kvn.loads(B1.read_text(encoding="utf-8") + "COMMENT the end\n")
    assert cdm.segments[1].comments[-1] == "the end"


def test_multi_line_comment_becomes_one_comment_line_each() -> None:
    base = minimal_cdm()
    relative = Section(
        "relative", (Field("COMMENT", "line one\nline two"), *base.relative.fields)
    )
    text = kvn.dumps(dataclasses.replace(base, relative=relative))
    assert "COMMENT line one\nCOMMENT line two\n" in text
    assert kvn.loads(text).relative.comments == ("line one", "line two")


def test_multi_line_value_cannot_be_written() -> None:
    base = minimal_cdm()
    header = Section("header", (*base.header.fields, Field("X_NOTE", "a\nb")))
    with pytest.raises(CdmWriteError, match="X_NOTE: KVN cannot hold a multi-line"):
        kvn.dumps(dataclasses.replace(base, header=header))


def test_keyword_must_start_with_a_letter() -> None:
    text = mutate("MESSAGE_ID ", "1ABC = x\nMESSAGE_ID ")
    with pytest.raises(CdmParseError, match="expected 'KEYWORD = value'") as info:
        kvn.loads(text)
    assert info.value.line == line_of(text, "1ABC")
