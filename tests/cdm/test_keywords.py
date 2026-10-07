import pytest

from keepout.cdm.keywords import (
    REQUIRED,
    SEGMENT,
    covariance_entries,
    covariance_names,
)


def test_covariance_names() -> None:
    six = covariance_names(6)
    assert len(six) == 21
    assert six[:4] == ("CR_R", "CT_R", "CT_T", "CN_R")
    assert six[-1] == "CNDOT_NDOT"
    nine = covariance_names(9)
    assert len(nine) == 45
    assert nine[21] == "CDRG_R"
    assert nine[-1] == "CTHR_THR"
    with pytest.raises(ValueError, match="dimension"):
        covariance_names(10)


def test_covariance_units() -> None:
    assert SEGMENT["CR_R"].unit == "m**2"
    assert SEGMENT["CRDOT_T"].unit == "m**2/s"
    assert SEGMENT["CNDOT_NDOT"].unit == "m**2/s**2"
    assert SEGMENT["CDRG_TDOT"].unit == "m**3/(kg*s)"
    assert SEGMENT["CSRP_DRG"].unit == "m**4/kg**2"
    assert SEGMENT["CTHR_SRP"].unit == "m**3/(kg*s**2)"
    assert SEGMENT["CTHR_THR"].unit == "m**2/s**4"


def test_required_segment_keywords_include_the_six_by_six_covariance() -> None:
    assert set(covariance_names(6)) <= set(REQUIRED["segment"])
    assert {"OBJECT", "REF_FRAME", "X", "Z_DOT"} <= set(REQUIRED["segment"])


def test_covariance_entries_follow_the_lower_triangle() -> None:
    entries = covariance_entries(3)
    assert entries == (
        ("CR_R", 0, 0),
        ("CT_R", 1, 0),
        ("CT_T", 1, 1),
        ("CN_R", 2, 0),
        ("CN_T", 2, 1),
        ("CN_N", 2, 2),
    )
    assert covariance_names(3) == tuple(name for name, _, _ in entries)
