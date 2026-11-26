"""CCSDS keyword values: units, numbers and UTC epochs (on CDM keywords)."""

import math
import sys
import warnings

import pytest
from astropy.time import Time
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from keepout.ccsds.epochs import (
    format_epoch,
    parse_epoch,
    preparse_epochs,
    utc_fields,
)
from keepout.ccsds.keywords import (
    Keyword,
    format_value,
    message_number,
    normalize_unit,
    parse_value,
)
from keepout.cdm.keywords import SEGMENT


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
        (SEGMENT["X"], "nan", "km", "invalid number 'nan'"),
        (SEGMENT["X"], "inf", "km", "invalid number 'inf'"),
        (SEGMENT["X"], "1_0.5", "km", "invalid number '1_0.5'"),
        (SEGMENT["X"], "1e999", "km", "number out of range '1e999'"),
        (SEGMENT["OBS_USED"], "1_06", None, "invalid integer '1_06'"),
        (SEGMENT["OBS_USED"], "\u0661\u0662", None, "invalid integer"),
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


@pytest.mark.parametrize(
    ("keyword", "text", "unit", "expected"),
    [
        # A unitless keyword accepts the placeholder units.
        ("WEIGHTED_RMS", "0.864", "n/a", 0.864),
        ("WEIGHTED_RMS", "0.864", "", 0.864),
        ("OBS_USED", "579", None, 579),
        ("OBJECT_TYPE", "SPACE STATION", None, "SPACE STATION"),
    ],
)
def test_a_value_is_parsed_to_its_keywords_kind(
    keyword: str, text: str, unit: str | None, expected: object
) -> None:
    parsed = parse_value(SEGMENT[keyword], text, unit)
    assert parsed == expected
    assert type(parsed) is type(expected)


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


@pytest.mark.parametrize(
    ("text", "isot"),
    [
        ("2020-03-04T10:34:41.9999999996", "2020-03-04T10:34:42.000000000"),
        ("1999-365T23:59:59.9999999999", "2000-01-01T00:00:00.000000000"),
    ],
)
def test_a_fraction_of_over_nine_digits_may_round_up_to_the_next_second(
    text: str, isot: str
) -> None:
    # CCSDS sets no limit on the digits; astropy keeps nine.
    assert format_epoch(parse_epoch(text)) == isot


@pytest.mark.parametrize(
    "text", ["1971-12-31T23:59:59.999", "1964-366T12:00:00", "1957-10-04T19:28:34"]
)
def test_a_utc_epoch_before_1972_is_refused(text: str) -> None:
    with pytest.raises(ValueError, match="UTC epochs before 1972-01-01 are not read"):
        parse_epoch(text)


def test_a_utc_epoch_before_1972_is_not_written() -> None:
    with pytest.raises(ValueError, match="UTC epochs before 1972-01-01 are not"):
        format_epoch(Time("1971-12-31T23:59:59", scale="utc"))


@pytest.mark.parametrize(
    "text",
    [
        "2010-13-40T00:00:00",
        "yesterday",
        "2010-03-13",
        "2021-02-29T00:00:00",
        "2021-366T00:00:00",
        "2020-01-01T23:59:60",
        "2021-365T23:59:60",
        "2020-01-01T23:59:61",
        "2020-01-01T24:00:00",
        "2020-01-01T12:60:00",
        # Outside ERFA's leap-second table a second 60 is still refused.
        "2035-06-30T23:59:60",
    ],
)
def test_invalid_epochs(text: str) -> None:
    with pytest.raises(ValueError, match="invalid CCSDS epoch"):
        parse_epoch(text)


@pytest.mark.parametrize(
    ("text", "isot"),
    [
        ("2016-12-31T23:59:60.500", "2016-12-31T23:59:60.500"),
        ("2016-366T23:59:60", "2016-12-31T23:59:60"),
        ("2020-366T12:00:00", "2020-12-31T12:00:00"),
    ],
)
def test_leap_seconds_and_leap_days_are_accepted(text: str, isot: str) -> None:
    assert format_epoch(parse_epoch(text)) == isot


@pytest.mark.parametrize(
    ("text", "isot"),
    [
        ("2035-06-30T12:00:00.25", "2035-06-30T12:00:00.25"),
        ("2150-001T00:00:00.000001", "2150-01-01T00:00:00.000001"),
    ],
)
def test_epochs_outside_the_leap_second_table_are_accepted(
    text: str, isot: str
) -> None:
    # ERFA calls UTC a few years past its last leap second "dubious": the
    # leap seconds there are unknown, not the time invalid.
    assert format_epoch(parse_epoch(text)) == isot


@pytest.mark.parametrize(
    ("keyword", "text"),
    [("X", "-6535.677249"), ("RECOMMENDED_OD_SPAN", "7.88"), ("X_DOT", "-2.8886125")],
)
def test_scaled_values_are_written_as_given(keyword: str, text: str) -> None:
    entry = SEGMENT[keyword]
    value = parse_value(entry, text, entry.unit)
    assert format_value(entry, value) == (text, entry.unit)


@given(
    st.sampled_from(["X", "X_DOT", "RECOMMENDED_OD_SPAN", "CR_R"]),
    st.integers(min_value=-(10**12), max_value=10**12),
    st.integers(min_value=0, max_value=9),
)
def test_writing_a_value_loses_nothing(
    keyword: str, mantissa: int, places: int
) -> None:
    entry = SEGMENT[keyword]
    text = f"{mantissa / 10**places:.{places}f}"
    value = parse_value(entry, text, entry.unit)
    written, _ = format_value(entry, value)
    assert parse_value(entry, written, entry.unit) == value
    assert float(written) == float(text)


@given(
    st.sampled_from(["X", "X_DOT", "RECOMMENDED_OD_SPAN", "CR_R"]),
    st.floats(allow_nan=False, allow_infinity=False),
)
def test_any_value_written_reads_back_within_one_ulp(
    keyword: str, value: float
) -> None:
    entry = SEGMENT[keyword]
    # Below this the message number is subnormal and has fewer digits.
    assume(value == 0.0 or abs(value) >= entry.scale * sys.float_info.min)
    written, _ = format_value(entry, value)
    back = parse_value(entry, written, entry.unit)
    assert back in (
        math.nextafter(value, -math.inf),
        value,
        math.nextafter(value, math.inf),
    )


def test_the_message_number_of_an_unscaled_value_is_the_value() -> None:
    assert message_number(4.835e-05, 1.0) == 4.835e-05
    assert str(message_number(-0.0, 1.0)) == "-0.0"
    assert message_number(6535677.249, 1000.0) == 6535.677249


def _fields_text(fields: tuple[int, ...], digits: int) -> str:
    year, month, day, hour, minute, second, fraction = fields
    text = f"{year}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}"
    return text + (f".{fraction:0{digits}d}" if digits else "")


@settings(max_examples=300)
@given(
    st.floats(min_value=2441317.5, max_value=5373484.0),
    st.integers(min_value=0, max_value=9),
)
def test_utc_fields_are_those_format_epoch_writes(jd: float, digits: int) -> None:
    epoch = Time(jd, format="jd", scale="utc")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # dubious years
        expected = format_epoch(Time(epoch, precision=digits))
    fields = utc_fields(epoch, digits)
    if fields is not None:
        assert _fields_text(fields, digits) == expected


@pytest.mark.parametrize(
    ("text", "digits", "isot"),
    [
        ("2016-12-31T23:59:60.5", 6, "2016-12-31T23:59:60.500000"),
        ("2016-12-31T23:59:59.9999996", 6, "2016-12-31T23:59:60.000000"),
        ("2026-12-31T23:59:59.9999996", 6, "2027-01-01T00:00:00.000000"),
        ("2100-02-28T23:59:59.5", 0, "2100-03-01T00:00:00"),
    ],
)
def test_utc_fields_round_as_format_epoch(text: str, digits: int, isot: str) -> None:
    fields = utc_fields(parse_epoch(text), digits)
    assert fields is not None
    assert _fields_text(fields, digits) == isot


def test_utc_fields_are_none_for_unusual_epochs() -> None:
    date_only = Time("2026-10-06T00:20:47", scale="utc", out_subfmt="date")
    assert utc_fields(date_only, 6) is None
    assert utc_fields(Time(["2026-10-06T00:20:47"], scale="utc"), 6) is None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # dubious year
        before_1972 = Time(2441317.0, format="jd", scale="utc")
    assert utc_fields(before_1972, 6) is None


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


@pytest.mark.parametrize(
    ("keyword", "value"),
    [("TIME_LASTOB_START", "2010-03-13"), ("X", "1500")],
    ids=["text for an epoch", "text for a number"],
)
def test_format_value_refuses_a_value_of_another_kind(keyword: str, value: str) -> None:
    with pytest.raises(TypeError, match=f"{keyword} is a"):
        format_value(SEGMENT[keyword], value)


@pytest.mark.parametrize(
    ("text", "value"),
    [("+1.5", 1.5), (".5", 0.5), ("5.", 5.0), ("1E+05", 1e5), ("-2.5e-3", -2.5e-3)],
)
def test_ccsds_real_forms_are_accepted(text: str, value: float) -> None:
    assert parse_value(SEGMENT["WEIGHTED_RMS"], text, None) == value


def _epoch_outcome(text: str) -> tuple[object, ...]:
    """What parse_epoch makes of a text: the exact Time, or its error."""
    try:
        epoch = parse_epoch(text)
    except ValueError as exc:
        return ("error", str(exc))
    return _time_state(epoch)


def _time_state(epoch: Time) -> tuple[object, ...]:
    return (
        float(epoch.jd1),
        float(epoch.jd2),
        epoch.scale,
        epoch.format,
        epoch.precision,
        epoch.isscalar,
    )


_LEAP_SECOND_DAYS = ("2016-12-31", "2016-366", "2015-06-30", "1972-06-30", "1990-365")
_STEP_DAYS = ("1964-12-31", "1968-01-31", "1961-07-31", "1971-12-31", "1959-12-31")


@st.composite
def _epoch_texts(draw: st.DrawFn) -> str:
    """CCSDS-like epochs, valid and not: any field may be out of range."""
    if draw(st.integers(0, 9)) == 0:
        return draw(st.text(max_size=30))
    if draw(st.booleans()):
        date = draw(st.sampled_from(_LEAP_SECOND_DAYS + _STEP_DAYS))
    else:
        year = draw(
            st.one_of(
                st.integers(1957, 2040),
                st.sampled_from([0, 999, 1000, 2100, 9999]),
            )
        )
        if draw(st.booleans()):
            date = f"{year:04d}-{draw(st.integers(0, 367)):03d}"
        else:
            month, day = draw(st.integers(0, 13)), draw(st.integers(0, 32))
            date = f"{year:04d}-{month:02d}-{day:02d}"
    hour = draw(st.one_of(st.integers(0, 23), st.just(24)))
    minute = draw(st.one_of(st.integers(0, 59), st.just(60)))
    second = draw(st.one_of(st.integers(0, 59), st.sampled_from([59, 60, 61])))
    fraction = draw(st.text("0123456789", max_size=12))
    if draw(st.integers(0, 3)) == 0:
        fraction = draw(st.sampled_from(["9" * 10, "9" * 9, "0" * 9, "5"]))
    text = f"{date}T{hour:02d}:{minute:02d}:{second:02d}"
    text += f".{fraction}" if fraction else ""
    text += draw(st.sampled_from(["", "Z", " ", "z"]))
    return draw(st.sampled_from(["", " "])) + text


@settings(max_examples=300)
@given(st.lists(_epoch_texts(), max_size=40))
def test_epochs_parsed_together_are_those_parsed_one_by_one(texts: list[str]) -> None:
    expected = [_epoch_outcome(text) for text in texts]
    for text, epoch, outcome in zip(
        texts, preparse_epochs(texts), expected, strict=True
    ):
        if epoch is not None:
            assert _time_state(epoch) == outcome, text


@pytest.mark.parametrize(
    ("texts", "parsed"),
    [
        # Leap seconds, day of year, Z, and every number of fraction digits.
        (
            ["2016-12-31T23:59:60.5", "2016-366T23:59:60Z", "2016-366T12:00:00"]
            + [f"2026-02-28T23:59:59{'.' + '7' * n if n else ''}" for n in range(10)],
            [False, False] + [True] * 11,
        ),
        # A day 30 of February among valid epochs of the same form.
        (
            ["2026-02-27T00:00:00.123456", "2026-02-30T00:00:00.123456"],
            [True, False],
        ),
        # Rounded past the end of its minute; non-ASCII digits; one just before.
        (
            [
                "2026-01-01T00:00:59.9999999999",
                "\u0662026-01-01T00:00:00",
                "2026-01-01T00:00:59.999999999",
            ],
            [False, False, True],
        ),
        # A year past ERFA's leap-second table, and years before 1972.
        (
            ["1955-01-01T00:00:00.5", "2035-06-30T12:00:00", "1971-12-31T12:00:00"],
            [False, True, False],
        ),
    ],
)
def test_unusual_epochs_parsed_together_are_those_parsed_one_by_one(
    texts: list[str], parsed: list[bool]
) -> None:
    epochs = preparse_epochs(texts)
    assert [epoch is not None for epoch in epochs] == parsed
    for text, epoch in zip(texts, epochs, strict=True):
        if epoch is not None:
            assert _time_state(epoch) == _epoch_outcome(text), text


def test_common_epochs_are_parsed_together_as_separate_times() -> None:
    texts = ["2026-10-17T12:00:00.123456", "2026-290T12:00:00.123", "2026-10-17T00:00"]
    first, second, third = preparse_epochs(texts)
    assert first is not None
    assert second is not None
    assert third is None
    assert (first.isscalar, first.precision, first.format) == (True, 6, "isot")
    assert (second.isscalar, second.precision, second.format) == (True, 3, "yday")


def test_epochs_parsed_together_are_independent_times() -> None:
    first, second = preparse_epochs(["2026-01-01T00:00:00.5", "2026-01-02T00:00:00.5"])
    assert first is not None
    assert second is not None
    first.precision = 3
    assert second.precision == 1
