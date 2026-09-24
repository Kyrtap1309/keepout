import pytest

from keepout.cdm.keywords import (
    REQUIRED,
    SEGMENT,
    Keyword,
    covariance_names,
    format_epoch,
    format_value,
    normalize_unit,
    parse_epoch,
    parse_value,
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


def test_normalize_unit() -> None:
    assert normalize_unit(" M**2/S ") == "m2/s"
    assert normalize_unit("m^2") == "m2"


def test_parse_float_converts_to_si() -> None:
    assert parse_value(SEGMENT["X"], "1.5", "km") == 1500.0
    assert parse_value(SEGMENT["X"], "1.5", "KM") == 1500.0
    assert parse_value(SEGMENT["X"], "1.5", None) == 1500.0
    assert parse_value(SEGMENT["RECOMMENDED_OD_SPAN"], "2", "d") == 172800.0
    assert parse_value(SEGMENT["CR_R"], "4.142E+01", "m2") == 41.42


@pytest.mark.parametrize(
    ("keyword", "text", "unit", "message"),
    [
        (SEGMENT["X"], "1.5", "m", r"unit \[m\] does not match expected \[km\]"),
        (SEGMENT["X"], "abc", "km", "invalid number 'abc'"),
        (SEGMENT["X"], "nan", "km", "non-finite number 'nan'"),
        (SEGMENT["X"], "", "km", "empty value"),
        (SEGMENT["OBS_USED"], "1.5", None, "invalid integer '1.5'"),
        (SEGMENT["WEIGHTED_RMS"], "1.0", "m", r"takes no unit"),
    ],
)
def test_parse_value_errors(
    keyword: Keyword, text: str, unit: str | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_value(keyword, text, unit)


def test_unitless_keyword_accepts_placeholder_units() -> None:
    assert parse_value(SEGMENT["WEIGHTED_RMS"], "0.864", "n/a") == 0.864
    assert parse_value(SEGMENT["WEIGHTED_RMS"], "0.864", "") == 0.864


def test_parse_int_and_text() -> None:
    assert parse_value(SEGMENT["OBS_USED"], "579", None) == 579
    assert parse_value(SEGMENT["OBJECT_TYPE"], "SPACE STATION", None) == "SPACE STATION"


@pytest.mark.parametrize(
    "text",
    ["2010-03-13T22:37:52.618", "2010-03-13T22:37:52", "2010-03-13T22:37:52.618123"],
)
def test_epoch_round_trip_keeps_precision(text: str) -> None:
    assert format_epoch(parse_epoch(text)) == text


def test_ordinal_epoch_and_trailing_z() -> None:
    assert (
        format_epoch(parse_epoch("2010-072T22:37:52.618")) == "2010-03-13T22:37:52.618"
    )
    assert (
        format_epoch(parse_epoch("2010-03-13T22:37:52.618Z"))
        == "2010-03-13T22:37:52.618"
    )


@pytest.mark.parametrize("text", ["2010-13-40T00:00:00", "yesterday", "2010-03-13"])
def test_invalid_epochs(text: str) -> None:
    with pytest.raises(ValueError, match="invalid CCSDS epoch"):
        parse_epoch(text)


def test_format_value() -> None:
    assert format_value(SEGMENT["X"], 1500.0) == ("1.5", "km")
    assert format_value(SEGMENT["CR_R"], 4.835e-05) == ("4.835E-05", "m**2")
    assert format_value(SEGMENT["OBS_USED"], 579) == ("579", None)
    assert format_value(SEGMENT["OBJECT_NAME"], "SAT A") == ("SAT A", None)
    epoch = parse_epoch("2010-03-13T22:37:52.618")
    assert format_value(SEGMENT["TIME_LASTOB_START"], epoch) == (
        "2010-03-13T22:37:52.618",
        None,
    )
