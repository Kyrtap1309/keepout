import numpy as np
import pytest
from cdm_builders import (
    DIAGONAL,
    TCA,
    minimal_cdm,
    replace,
    segment_fields,
    without,
)

from keepout.cdm.keywords import covariance_names
from keepout.cdm.model import Field, Section
from keepout.encounter import encounter_plane
from keepout.errors import CovarianceError, GeometryError


def test_typed_accessors() -> None:
    cdm = minimal_cdm()
    assert cdm.version == "1.0"
    assert cdm.originator == "KEEPOUT"
    assert cdm.message_id == "TEST-1"
    assert cdm.message_for is None
    assert cdm.tca is TCA
    assert cdm.miss_distance == 500.0
    assert cdm.collision_probability is None
    first = cdm.object1
    assert first.designator == "10001"
    assert first.name == "SAT 1"
    assert first.international_designator == "2000-001A"
    assert first.ref_frame == "EME2000"
    assert first.hbr is None
    assert first.user_defined == {}
    np.testing.assert_array_equal(first.position, [7000e3, 0.0, 0.0])
    np.testing.assert_array_equal(first.velocity, [0.0, 7500.0, 0.0])
    np.testing.assert_array_equal(first.covariance_rtn, np.diag(DIAGONAL))
    assert cdm.object2.designator == "10002"


def test_section_rejects_repeated_keywords() -> None:
    fields = (
        Field("TCA", TCA),
        Field("MISS_DISTANCE", 1.0),
        Field("MISS_DISTANCE", 2.0),
    )
    with pytest.raises(ValueError, match="repeats keyword MISS_DISTANCE"):
        Section("relative", fields)


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


def test_segments_must_be_labelled_in_order() -> None:
    with pytest.raises(ValueError, match="segment 2 must have OBJECT = OBJECT2"):
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
    with pytest.raises(GeometryError, match="unsupported REF_FRAME 'TOD'"):
        cdm.object1.state()


def test_covariance_policies() -> None:
    cdm = minimal_cdm(segment1=replace(segment_fields(1), "CT_R", 1e6))
    with pytest.raises(CovarianceError):
        cdm.object1.covariance(policy="raise")
    covariance, diagnostics = cdm.object1.covariance()
    assert covariance.frame == "RTN"
    assert diagnostics[0].startswith("clipped 1 negative eigenvalue")


def test_object_states() -> None:
    primary, secondary, diagnostics = minimal_cdm().object_states()
    assert diagnostics == ()
    assert primary.object_id == "10001"
    assert primary.state.frame == "GCRF"
    assert primary.covariance.frame == "GCRF"
    np.testing.assert_allclose(
        primary.covariance.in_rtn(primary.state).matrix, np.diag(DIAGONAL), atol=1e-9
    )
    # r = (0, 0, 500), v = (0, -7500, 7500): half of r lies along v.
    plane = encounter_plane(primary, secondary)
    assert plane.miss_distance == pytest.approx(500.0 / np.sqrt(2.0), rel=1e-12)


def test_object_state_diagnostics_are_labelled() -> None:
    cdm = minimal_cdm(segment1=replace(segment_fields(1), "CT_R", 1e6))
    _, _, diagnostics = cdm.object_states()
    assert len(diagnostics) == 1
    assert diagnostics[0].startswith("OBJECT1: clipped 1 negative eigenvalue")


def test_hbr_and_user_defined() -> None:
    fields = [
        *segment_fields(1),
        Field("HBR", 5.0),
        Field("USER_DEFINED_RUN_ID", "abc"),
        Field("X_OTHER", "1"),
    ]
    cdm = minimal_cdm(segment1=fields)
    assert cdm.object1.hbr == 5.0
    assert cdm.object1.user_defined == {"USER_DEFINED_RUN_ID": "abc"}
    primary, _, _ = cdm.object_states()
    assert primary.hard_body_radius == 5.0
