from collections.abc import Callable

import numpy as np
import pytest
from builders import DIAGONAL_COVARIANCE, EPOCH, known
from cdm_builders import (
    CREATED,
    TCA,
    minimal_cdm,
    replace,
    segment_fields,
    without,
)

from keepout.cdm import model
from keepout.cdm.keywords import covariance_names
from keepout.cdm.model import (
    Cdm,
    CdmObject,
    CovarianceMethod,
    DuplicateKeyword,
    Field,
    Section,
)
from keepout.conjunction import Conjunction
from keepout.errors import CdmParseError, CovarianceError, UnsupportedFrame
from keepout.frames import teme_to_gcrf


def test_typed_accessors() -> None:
    cdm = minimal_cdm()
    assert cdm.version == "1.0"
    assert cdm.originator == "KEEPOUT"
    assert cdm.message_id == "TEST-1"
    assert cdm.creation_date is CREATED
    assert cdm.message_for is None
    assert cdm.conjunction_id is None
    assert cdm.tca is TCA
    assert cdm.miss_distance == 500.0
    assert cdm.collision_probability is None
    first = cdm.object1
    assert first.object_id == "10001"
    assert first.name == "SAT 1"
    assert first.international_designator == "2000-001A"
    assert first.ref_frame == "EME2000"
    assert first.hard_body_radius is None
    assert first.user_defined == {}
    np.testing.assert_array_equal(first.position, [7000e3, 0.0, 0.0])
    np.testing.assert_array_equal(first.velocity, [0.0, 7500.0, 0.0])
    np.testing.assert_array_equal(first.covariance_rtn, DIAGONAL_COVARIANCE)
    assert cdm.object2.object_id == "10002"


@pytest.mark.parametrize(
    ("value", "method", "assumed"),
    [
        ("CALCULATED", CovarianceMethod.CALCULATED, False),
        ("DEFAULT", CovarianceMethod.DEFAULT, True),
    ],
)
def test_the_covariance_method_is_typed(
    value: str, method: CovarianceMethod, assumed: bool
) -> None:
    cdm = minimal_cdm(segment2=replace(segment_fields(2), "COVARIANCE_METHOD", value))
    assert cdm.object2.covariance_method is method
    assert cdm.object2.assumed_covariance is assumed
    assert cdm.assumed_covariance is assumed
    assert not cdm.object1.assumed_covariance


def test_an_unknown_covariance_method_is_a_parse_error() -> None:
    cdm = minimal_cdm(segment1=replace(segment_fields(1), "COVARIANCE_METHOD", "X"))
    with pytest.raises(CdmParseError, match="must be CALCULATED or DEFAULT, got 'X'"):
        _ = cdm.object1.covariance_method
    with pytest.raises(CdmParseError, match="keyword COVARIANCE_METHOD"):
        _ = cdm.assumed_covariance


def test_a_cdm_has_exactly_two_object_sections() -> None:
    cdm = minimal_cdm()
    with pytest.raises(ValueError, match="a CDM has 2 object sections, got 1"):
        Cdm(cdm.header, cdm.relative, (cdm.object_sections[0],))  # type: ignore[arg-type]


def test_sections_must_have_their_kind() -> None:
    cdm = minimal_cdm()
    with pytest.raises(ValueError, match="header section has kind 'relative'"):
        Cdm(cdm.relative, cdm.relative, cdm.object_sections)


def test_section_rejects_repeated_keywords() -> None:
    fields = (
        Field("TCA", TCA),
        Field("MISS_DISTANCE", 1.0),
        Field("MISS_DISTANCE", 2.0),
    )
    with pytest.raises(DuplicateKeyword, match="repeats keyword MISS_DISTANCE") as info:
        Section("relative", fields)
    assert info.value.keyword == "MISS_DISTANCE"


def test_comments_may_repeat() -> None:
    fields = (Field("COMMENT", "a"), Field("TCA", TCA), Field("COMMENT", "b"))
    assert Section("relative", fields).comments == ("a", "b")


@pytest.mark.parametrize("keyword", ["1ABC", "lower", "HAS SPACE", "", "X-Y"])
def test_section_rejects_malformed_keywords(keyword: str) -> None:
    # Every keyword must be writable as both KVN and an XML element name.
    with pytest.raises(ValueError, match="must be upper-case letters"):
        Section("relative", (Field(keyword, "1"),))


def test_section_rejects_wrongly_typed_values() -> None:
    with pytest.raises(
        ValueError, match="MISS_DISTANCE value '500' is not of kind 'float'"
    ):
        Section("relative", (Field("MISS_DISTANCE", "500"),))


def test_section_rejects_units_on_known_keywords() -> None:
    with pytest.raises(ValueError, match="unit is fixed by the standard"):
        Section("relative", (Field("MISS_DISTANCE", 1.0, "km"),))


def test_unknown_keywords_must_be_text() -> None:
    with pytest.raises(ValueError, match="is not of kind 'str'"):
        Section("relative", (Field("X_FOO", 1.0),))


def test_unknown_section_kind() -> None:
    with pytest.raises(ValueError, match="unknown section kind 'body'"):
        # The wrong literal is the point of the test.
        Section("body", ())  # type: ignore[arg-type]


def test_comments_and_extras() -> None:
    section = Section(
        "relative",
        (
            Field("TCA", TCA),
            Field("COMMENT", "a"),
            Field("MISS_DISTANCE", 1.0),
            Field("X_FOO", "bar", "m"),
            Field("COMMENT", "b"),
        ),
    )
    assert section.comments == ("a", "b")
    assert section.extras == {"X_FOO": "bar"}
    assert section.get("X_FOO") == "bar"
    assert section.get("NOPE") is None
    assert section.missing() == ()


def test_missing_required_keyword() -> None:
    with pytest.raises(
        ValueError, match=r"OBJECT2 section is missing required keyword\(s\): X_DOT"
    ):
        minimal_cdm(segment2=without(segment_fields(2), "X_DOT"))


def test_object_sections_must_be_labelled_in_order() -> None:
    with pytest.raises(ValueError, match="object section 2 must have OBJECT = OBJECT2"):
        minimal_cdm(segment2=replace(segment_fields(2), "OBJECT", "OBJECT1"))


def test_incomplete_extended_covariance() -> None:
    fields = [*segment_fields(1), Field("CDRG_R", 1.0)]
    with pytest.raises(ValueError, match="complete 6x6, 7x7, 8x8 or 9x9"):
        minimal_cdm(segment1=fields)


def test_nine_by_nine_covariance() -> None:
    extra = [Field(name, 0.0) for name in covariance_names(9)[21:]]
    cdm = minimal_cdm(segment1=[*segment_fields(1), *extra])
    assert cdm.object1.covariance_rtn.shape == (9, 9)


def test_unsupported_ref_frame() -> None:
    cdm = minimal_cdm(segment1=replace(segment_fields(1), "REF_FRAME", "TOD"))
    with pytest.raises(UnsupportedFrame, match="unsupported reference frame 'TOD'"):
        cdm.object1.state()
    with pytest.raises(UnsupportedFrame, match="unsupported reference frame 'TOD'"):
        cdm.object_states()


def test_a_teme_state_is_converted_to_gcrf() -> None:
    cdm = minimal_cdm(segment1=replace(segment_fields(1), "REF_FRAME", "TEME"))
    teme = cdm.object1.state()
    assert teme.frame == "TEME"
    primary, _ = cdm.object_states()
    assert primary.state.frame == "GCRF"
    expected_r, expected_v = teme_to_gcrf(teme.epoch, teme.r, teme.v)
    np.testing.assert_array_equal(primary.state.r, expected_r)
    np.testing.assert_array_equal(primary.state.v, expected_v)


def test_covariance_policies() -> None:
    cdm = minimal_cdm(segment1=replace(segment_fields(1), "CT_R", 1e6))
    with pytest.raises(CovarianceError, match="not positive semi-definite"):
        cdm.object1.covariance(policy="raise")
    covariance = cdm.object1.covariance()
    assert covariance.frame == "RTN"
    assert covariance.repairs[0].startswith(
        "clipped 1 negative position-block eigenvalue"
    )


def test_object_states() -> None:
    primary, secondary = minimal_cdm().object_states()
    assert (
        known(primary.covariance).repairs == known(secondary.covariance).repairs == ()
    )
    assert primary.object_id == "10001"
    assert primary.state.frame == "GCRF"
    assert known(primary.covariance).frame == "GCRF"
    np.testing.assert_allclose(
        known(primary.covariance).in_rtn(primary.state).matrix,
        DIAGONAL_COVARIANCE,
        atol=1e-9,
    )
    # r = (0, 0, 500), v = (0, -7500, 7500): half of r lies along v.
    plane = Conjunction(primary, secondary, combined_hard_body_radius=1.0).plane
    assert plane.miss_distance == pytest.approx(500.0 / np.sqrt(2.0), rel=1e-12)


def test_object_state_carries_its_covariance_repairs() -> None:
    cdm = minimal_cdm(segment1=replace(segment_fields(1), "CT_R", 1e6))
    first, second = cdm.object_states()
    (repair,) = known(first.covariance).repairs
    assert repair.startswith("clipped 1 negative position-block eigenvalue")
    assert known(second.covariance).repairs == ()
    assert known(cdm.object1.object_state().covariance).repairs == (repair,)


def test_objects_in_different_frames_share_one_inertial_frame() -> None:
    # The same physical state given in EME2000 and in GCRF: after conversion
    # both objects coincide, so the frame bias is not mistaken for a miss.
    first = minimal_cdm().object1.object_state()
    gcrf = segment_fields(2, frame="GCRF")
    for keyword, value in zip("XYZ", first.state.r, strict=True):
        gcrf = replace(gcrf, keyword, float(value))
    _, second = minimal_cdm(segment2=gcrf).object_states()
    np.testing.assert_allclose(second.state.r, first.state.r, atol=1e-6)


def test_hbr_and_user_defined() -> None:
    fields = [
        *segment_fields(1),
        Field("HBR", 5.0),
        Field("USER_DEFINED_RUN_ID", "abc"),
        Field("X_OTHER", "1"),
    ]
    cdm = minimal_cdm(segment1=fields)
    assert cdm.object1.hard_body_radius == 5.0
    assert cdm.object1.user_defined == {"USER_DEFINED_RUN_ID": "abc"}
    primary, _ = cdm.object_states()
    assert primary.hard_body_radius == 5.0


def test_all_zero_covariance_means_unknown() -> None:
    fields = segment_fields(2)
    for name in covariance_names(6):
        fields = replace(fields, name, 0.0)
    cdm = minimal_cdm(segment2=fields)
    assert cdm.object1.has_covariance
    assert not cdm.object2.has_covariance
    primary, secondary = cdm.object_states()
    assert primary.covariance is not None
    assert secondary.covariance is None


BARE_OBJECT = CdmObject(Section("segment", (Field("OBJECT", "OBJECT1"),)), EPOCH)
"""An object section outside a CDM, which would check its required keywords."""


@pytest.mark.parametrize(
    ("read", "keyword"),
    [
        (lambda: BARE_OBJECT.object_id, "OBJECT_DESIGNATOR"),
        (lambda: BARE_OBJECT.ref_frame, "REF_FRAME"),
        (lambda: BARE_OBJECT.position, "X"),
        (lambda: BARE_OBJECT.velocity, "X_DOT"),
        # A Cdm requires its epochs, so only a bare section lacks one.
        (lambda: model._epoch(Section("relative", ()), "TCA"), "TCA"),
    ],
    ids=["text", "frame", "position", "velocity", "epoch"],
)
def test_reading_an_absent_keyword_is_a_key_error(
    read: Callable[[], object], keyword: str
) -> None:
    with pytest.raises(KeyError, match=f"^'{keyword}'$"):
        read()
