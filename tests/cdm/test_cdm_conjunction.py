import pytest
from cdm_builders import minimal_cdm, replace, segment_fields

from keepout.cdm.model import Field
from keepout.errors import CovarianceError


def with_hbr(index: int, radius: float) -> list[Field]:
    return [*segment_fields(index), Field("HBR", radius)]


def test_radius_comes_from_both_objects_hbr() -> None:
    cdm = minimal_cdm(segment1=with_hbr(1, 5.0), segment2=with_hbr(2, 2.5))
    conjunction = cdm.conjunction()
    assert conjunction.hard_body_radius == 7.5
    assert conjunction.primary.object_id == "10001"


def test_explicit_radius_overrides_hbr() -> None:
    cdm = minimal_cdm(segment1=with_hbr(1, 5.0), segment2=with_hbr(2, 2.5))
    assert cdm.conjunction(hard_body_radius=20.0).hard_body_radius == 20.0


def test_cdm_without_hbr_needs_an_explicit_radius() -> None:
    with pytest.raises(ValueError, match="no hard-body radius for 10001, 10002"):
        minimal_cdm().conjunction()


def test_covariance_repairs_become_conjunction_diagnostics() -> None:
    cdm = minimal_cdm(segment1=replace(segment_fields(1), "CT_R", 1e6))
    conjunction = cdm.conjunction(hard_body_radius=10.0)
    (diagnostic,) = conjunction.diagnostics
    assert diagnostic.startswith(
        "primary (10001): clipped 1 negative position-block eigenvalue"
    )


def test_raise_policy_rejects_an_invalid_cdm_covariance() -> None:
    cdm = minimal_cdm(segment2=replace(segment_fields(2), "CT_R", 1e6))
    with pytest.raises(CovarianceError, match="not positive semi-definite"):
        cdm.conjunction(hard_body_radius=10.0, policy="raise")
    valid = minimal_cdm().conjunction(hard_body_radius=10.0, policy="raise")
    assert valid.diagnostics == ()
