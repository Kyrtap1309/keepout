import datetime
from decimal import Decimal

import pytest
from astropy.time import Time
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from keepout.cdm.keywords import (
    REQUIRED,
    SEGMENT,
    Keyword,
    covariance_entries,
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
        "1955-181T23:59:60",
        # Past the 0.1 s UTC step at the end of 1964-12-31, and in the 0.1 s
        # that 1968-01-31 lacked.
        "1964-12-31T23:59:60.1",
        "1968-01-31T23:59:59.95",
        # astropy writes years before 1000 with fewer digits.
        "0999-365T12:00:00",
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
        ("1957-10-04T19:28:34", "1957-10-04T19:28:34"),
        ("2035-06-30T12:00:00.25", "2035-06-30T12:00:00.25"),
        ("2150-001T00:00:00.000001", "2150-01-01T00:00:00.000001"),
    ],
)
def test_epochs_outside_the_leap_second_table_are_accepted(
    text: str, isot: str
) -> None:
    # ERFA calls UTC before 1960 or a few years past its last leap second
    # "dubious": the leap seconds there are unknown, not the time invalid.
    assert format_epoch(parse_epoch(text)) == isot


UTC_STEP_DAYS = (
    "1960-12-31",
    "1961-07-31",
    "1963-10-31",
    "1964-03-31",
    "1964-08-31",
    "1964-12-31",
    "1965-02-28",
    "1965-06-30",
    "1965-08-31",
    "1968-01-31",
    "1971-12-31",
)
"""The days before 1972 at whose end UTC stepped by a fraction of a second
(-0.1 s to +0.108 s; leap seconds came in 1972)."""


@pytest.mark.parametrize("day", UTC_STEP_DAYS)
def test_epochs_on_a_day_that_ends_in_a_utc_step_are_read_as_written(
    day: str,
) -> None:
    # ERFA's Julian dates stretch such a day by its step; astropy's text
    # (ERFA's d2dtf) undoes that only for steps over 0.5 s, so it was 1 us
    # off at 00:00:00.432864 (a TLE epoch of day .00000501), 0.05 s at noon.
    for clock in ("00:00:00.432864", "12:00:00", "23:59:59.899999"):
        text = f"{day}T{clock}"
        assert format_epoch(parse_epoch(text)) == text


@pytest.mark.parametrize(
    ("text", "isot"),
    [
        ("1964-12-31T23:59:60.05", "1964-12-31T23:59:60.05"),
        ("1964-366T12:00:00.000001", "1964-12-31T12:00:00.000001"),
    ],
)
def test_the_utc_step_ends_its_day(text: str, isot: str) -> None:
    # Like a leap second, the 0.1 s step is written as 23:59:60.0 to 60.1.
    assert format_epoch(parse_epoch(text)) == isot


@pytest.mark.parametrize(
    ("text", "precision", "isot"),
    [
        ("1964-12-31T23:59:60.0999999", 6, "1965-01-01T00:00:00.000000"),
        ("1968-01-31T23:59:59.8999996", 6, "1968-02-01T00:00:00.000000"),
        ("1968-01-31T23:59:59.6", 0, "1968-02-01T00:00:00"),
        ("1964-12-31T23:59:60.06", 1, "1965-01-01T00:00:00.0"),
        ("1964-12-31T23:59:60.04", 1, "1964-12-31T23:59:60.0"),
        ("1964-366T23:59:60.0999999", 6, "1965-01-01T00:00:00.000000"),
    ],
)
def test_a_time_rounded_past_the_end_of_a_step_day_is_the_next_day(
    text: str, precision: int, isot: str
) -> None:
    # The date and the clock round together: the clock never reaches the
    # day's length on the old date.
    assert format_epoch(Time(parse_epoch(text), precision=precision)) == isot


_STEPS = "0.005 -0.05 0.1 0.1 0.1 0.1 0.1 0.1 0.1 -0.1 0.107758"
UTC_STEPS = dict(zip(UTC_STEP_DAYS, map(Decimal, _STEPS.split()), strict=True))
"""Each step [s], from the TAI-UTC table (USNO tai-utc.dat), as ERFA has it."""


def _step_day_reference(day: str, nanoseconds: Decimal, digits: int) -> str:
    """Text of the time ``nanoseconds`` into a UTC step day, rounded to ``digits``.

    A clock rounded to the day's length or past it does not exist on that
    day: the time is the next day's 00:00:00.
    """
    length = 86_400 + UTC_STEPS[day]
    clock = nanoseconds.scaleb(-9).quantize(Decimal(1).scaleb(-digits))
    if clock >= length:
        following = datetime.date.fromisoformat(day) + datetime.timedelta(days=1)
        zeros = "." + "0" * digits if digits else ""
        return f"{following.isoformat()}T00:00:00{zeros}"
    # From 23:59 on, the seconds run on past 59 into a positive step.
    minutes = min(int(clock), 86_340) // 60
    hour, minute = divmod(minutes, 60)
    seconds = clock - 3600 * hour - 60 * minute
    width = 3 + digits if digits else 2
    return f"{day}T{hour:02d}:{minute:02d}:{seconds:0{width}.{digits}f}"


@settings(deadline=None)
@given(
    day=st.sampled_from(UTC_STEP_DAYS),
    before_end=st.one_of(st.integers(0, 2_000_000), st.integers(0, 86_399_999_999_999)),
    digits=st.integers(0, 9),
)
def test_times_on_a_step_day_round_to_the_nearest_written_time(
    day: str, before_end: int, digits: int
) -> None:
    # Times in the last 2 ms of the day or anywhere in it, at every precision,
    # against exact decimal arithmetic on the day's length. A quarter of a
    # nanosecond off the grid keeps the rounding clear of ties.
    length_ns = (86_400 + UTC_STEPS[day]).scaleb(9)
    nanoseconds = length_ns - 1 - before_end + Decimal("0.25")
    assume(nanoseconds >= 0)
    start = Time(f"{day}T00:00:00", scale="utc")
    fraction = float(nanoseconds / length_ns)
    epoch = Time(start.jd1, start.jd2 + fraction, format="jd", scale="utc")
    expected = _step_day_reference(day, nanoseconds, digits)
    assert format_epoch(Time(epoch, precision=digits)) == expected
    # The text reads back to itself, also with the day of year.
    if expected.startswith(day):
        doy = datetime.date.fromisoformat(day).timetuple().tm_yday
        yday = f"{day[:4]}-{doy:03d}{expected[10:]}"
        assert format_epoch(parse_epoch(expected)) == expected
        assert format_epoch(parse_epoch(yday)) == expected


def test_an_epoch_on_a_day_that_ends_in_a_utc_step_keeps_its_time() -> None:
    # TAI-UTC on 1964-12-31: 3.44013 s less 0.001296 s a day before 1965.
    tai = parse_epoch("1964-12-31T12:00:00.000000").tai
    assert tai.isot == "1964-12-31T12:00:03.439482"


@pytest.mark.parametrize(
    "text",
    [
        "1964-12-30T12:00:00",
        "1965-01-01T00:00:00.000001",
        # ERFA's drift arithmetic gives this day a step of -4e-16 s.
        "1961-12-31T23:59:59.999999",
    ],
)
def test_the_other_days_before_1972_are_read_as_written(text: str) -> None:
    assert format_epoch(parse_epoch(text)) == text


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
    ("text", "value"),
    [("+1.5", 1.5), (".5", 0.5), ("5.", 5.0), ("1E+05", 1e5), ("-2.5e-3", -2.5e-3)],
)
def test_ccsds_real_forms_are_accepted(text: str, value: float) -> None:
    assert parse_value(SEGMENT["WEIGHTED_RMS"], text, None) == value


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
