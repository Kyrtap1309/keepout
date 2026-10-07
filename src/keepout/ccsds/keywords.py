"""CCSDS keyword values shared by the message formats: typing, epochs, units.

A :class:`Keyword` says how one keyword of a CCSDS message (CDM, OMM) is
typed, converted to SI and grouped in NDM/XML; :func:`parse_value` and
:func:`format_value` convert its value between message text and Python, so
the message value is recovered exactly when it is written back.
:func:`parse_epoch` and :func:`format_epoch` read and write CCSDS UTC epochs,
also on the days before 1972 that end in a UTC step of a fraction of a
second, where ERFA's text conversion is off.
"""

from __future__ import annotations

import contextlib
import math
import re
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias, assert_never

import erfa
from astropy.time import Time
from erfa import ErfaWarning

from keepout._units import DAY

__all__ = [
    "FieldValue",
    "Keyword",
    "ValueKind",
    "format_epoch",
    "format_value",
    "normalize_unit",
    "parse_epoch",
    "parse_value",
]

ValueKind: TypeAlias = Literal["str", "int", "float", "time"]
FieldValue: TypeAlias = str | int | float | Time


@dataclass(frozen=True)
class Keyword:
    """How one CCSDS message keyword is typed, converted and placed.

    The keyword tables are :mod:`keepout.cdm.keywords` (CDM) and
    :mod:`keepout.omm.keywords` (OMM).

    Parameters
    ----------
    name
        Keyword as written in the message.
    kind
        Python type of the stored value.
    unit
        Canonical CCSDS unit, or None for unitless keywords.
    scale
        Factor converting the file value to SI.
    group
        NDM/XML element path inside the section element.
    """

    name: str
    kind: ValueKind
    unit: str | None = None
    scale: float = 1.0
    group: tuple[str, ...] = ()


_UNITLESS: Final = frozenset({"", "n/a"})
_INTEGER: Final = re.compile(r"^[+-]?[0-9]+$")
_REAL: Final = re.compile(r"^[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")
_EPOCH: Final = re.compile(
    r"^(?P<year>\d{4})-(?:(?P<month>\d{2})-(?P<day>\d{2})|(?P<doy>\d{3}))"
    r"T(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})(?:\.(?P<fraction>\d+))?Z?$"
)


def normalize_unit(unit: str) -> str:
    """Canonical form for comparing units: lower case, no spaces, ``**`` or ``^``.

    Parameters
    ----------
    unit
        Unit text as written in a message.
    """
    return unit.strip().lower().replace(" ", "").replace("**", "").replace("^", "")


@contextlib.contextmanager
def _dubious_years_allowed() -> Iterator[None]:
    """ERFA's "dubious year" warning is ignored, every other ERFA warning raised.

    ERFA calls a UTC year "dubious" outside its leap-second table: before
    1960, when UTC began, or more than about five years past the table's
    release (from 2029 with the tables of 2026). There the UTC-TAI offset is
    extrapolated; the calendar time itself is exact, and an element set or
    CDM epoch there is valid.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", ErfaWarning)
        warnings.filterwarnings("ignore", r".*dubious year", ErfaWarning)
        yield


def parse_epoch(text: str) -> Time:
    """Parse a CCSDS UTC epoch.

    Accepts ``YYYY-MM-DDThh:mm:ss[.f]`` and ``YYYY-DDDThh:mm:ss[.f]`` with an
    optional trailing ``Z``. The returned Time keeps the number of fractional
    digits as its precision, so :func:`format_epoch` writes it back unchanged.
    Every field must be in range: a day past the end of its month or year,
    hour 24, or second 60 outside a UTC leap second is rejected rather than
    rolled over to another time. Years outside ERFA's leap-second table
    (before 1960, or a few years past its last leap second) are accepted.
    The eleven days before 1972 that end in a UTC step of a fraction of a
    second are as long as ERFA counts them: 1964-12-31 runs to 23:59:60.1,
    1968-01-31 ends at 23:59:59.9 (see :func:`format_epoch`).

    Parameters
    ----------
    text
        Epoch as written in the message.

    Raises
    ------
    ValueError
        If the text is not a valid CCSDS epoch.
    """
    match = _EPOCH.match(text.strip())
    if match is None:
        raise ValueError(f"invalid CCSDS epoch {text!r}")
    year, month, day, doy, hour, minute, second, fraction = match.groups()
    clock = f"{hour}:{minute}:{second}" + (f".{fraction}" if fraction else "")
    if doy is None:
        body, time_format = f"{year}-{month}-{day}T{clock}", "isot"
    else:
        body, time_format = f"{year}:{doy}:{clock}", "yday"
    with _dubious_years_allowed():
        try:
            # ERFA warns about a second 60 outside a leap second.
            epoch = Time(
                body,
                format=time_format,
                scale="utc",
                precision=min(len(fraction or ""), 9),
            )
        except (ValueError, ErfaWarning) as exc:
            raise ValueError(f"invalid CCSDS epoch {text!r}: {exc}") from exc
        # astropy rolls some out-of-range fields over (day 366 of a common
        # year becomes 1 January): the parsed fields must survive unchanged.
        date = f"{year}:{doy}" if doy else f"{year}-{month}-{day}"
        isot = _utc_isot(epoch)
        parsed = _day_of_year(isot) if doy else isot[:10]
        if parsed != date or isot[11:19] != f"{hour}:{minute}:{second}":
            raise ValueError(f"invalid CCSDS epoch {text!r}: a field is out of range")
    return epoch


def format_epoch(epoch: Time) -> str:
    """Write an epoch in CCSDS calendar format, UTC.

    Years outside ERFA's leap-second table are written without a warning
    (see :func:`parse_epoch`). On the eleven days before 1972 that end in a
    UTC step of a fraction of a second (1960-12-31 to 1971-12-31) the time
    is written right, where astropy's ``isot`` is off by up to 0.1 s (see
    ``_utc_isot``).

    Parameters
    ----------
    epoch
        Epoch to write; its precision sets the number of fractional digits.
    """
    with _dubious_years_allowed():
        return _utc_isot(epoch)


_UTC_STEP_FLOOR: Final = 1e-6
"""Smallest step [s] taken for one: ERFA's drift arithmetic leaves about
1e-16 s on the other days before 1972; the smallest step is 5 ms."""

_TICK_TOLERANCE: Final = 1e-3
"""Fraction of a tick by which a day's length may come out of ERFA's float
arithmetic too long (a length is a whole number of ticks or at least
0.005 tick off one)."""


def _utc_isot(epoch: Time) -> str:
    """``epoch.utc.isot``, also right on the days that end in a small UTC step.

    ERFA, and astropy with it, counts a UTC day as one unit of its Julian
    date also when the day is longer or shorter: clock time ``t`` on a day
    of ``86400 + s`` seconds is the fraction ``t / (86400 + s)``. Reading
    text (``dtf2d``) and converting to TAI (``utctai``) use that scale for
    any step ``s``, but writing text (``d2dtf``) undoes it only for steps
    over 0.5 s, the leap seconds. On the eleven days before 1972 that end
    in a step of 5 to 108 ms, astropy therefore writes ``t`` as
    ``t 86400 / (86400 + s)``: 1 us off at 00:00:00.43, up to 0.1 s at the
    end of the day. Here the clock is the fraction times the day's length,
    as ``d2dtf`` computes it on a leap-second day. The date is the day's
    own, from the Julian date: a clock that rounds to the day's length or
    past it does not exist on that day and is the next day's 00:00:00, as
    ``d2dtf`` writes a time rounded past the end of a leap second.
    """
    utc = epoch.utc
    year, month, day, fraction = erfa.jd2cal(utc.jd1, utc.jd2)
    if not 1960 <= year < 1972:
        return str(utc.isot)
    step = _utc_step(year, month, day)
    if abs(step) < _UTC_STEP_FLOOR:
        return str(utc.isot)
    digits = utc.precision
    scale = 10**digits
    ticks = round(float(fraction) * (DAY + step) * scale)
    # The first tick at or past the day's end; the tolerance covers the
    # rounding of the step (about 1e-15 s), far below a tick.
    if ticks >= math.ceil((DAY + step) * scale - _TICK_TOLERANCE):
        zero, mjd = erfa.cal2jd(year, month, day)
        year, month, day, _ = erfa.jd2cal(zero, mjd + 1.0)
        ticks = 0
    seconds, part = divmod(ticks, scale)
    # From 23:59 on, the seconds run on past 59 into a positive step.
    minutes = min(seconds, 86_340) // 60
    clock = f"{minutes // 60:02d}:{minutes % 60:02d}:{seconds - 60 * minutes:02d}"
    date = f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    return f"{date}T{clock}" + (f".{part:0{digits}d}" if digits else "")


def _day_of_year(isot: str) -> str:
    """``YYYY:DDD`` of the date of :func:`_utc_isot`'s text.

    astropy writes years before 1000 with fewer digits; those come back
    unchanged, so they never match a CCSDS date.
    """
    if not re.match(r"\d{4}-\d\d-\d\dT", isot):
        return isot
    year, month, day = int(isot[:4]), int(isot[5:7]), int(isot[8:10])
    _, mjd = erfa.cal2jd(year, month, day)
    _, mjd_first = erfa.cal2jd(year, 1, 1)
    return f"{isot[:4]}:{int(mjd - mjd_first) + 1:03d}"


def _utc_step(year: int, month: int, day: int) -> float:
    """Step of TAI-UTC at the end of a UTC day [s], as ERFA reckons it.

    The change from 0h to the next day's 0h beyond the day's drift; ERFA's
    ``dtf2d``, ``d2dtf`` and ``utctai`` compute it alike.
    """
    start = erfa.dat(year, month, day, 0.0)
    noon = erfa.dat(year, month, day, 0.5)
    zero, mjd = erfa.cal2jd(year, month, day)
    next_year, next_month, next_day, _ = erfa.jd2cal(zero, mjd + 1.0)
    end = erfa.dat(next_year, next_month, next_day, 0.0)
    return float(end - (2.0 * noon - start))


def parse_value(keyword: Keyword, text: str, unit: str | None) -> FieldValue:
    """Convert a value from a message into its Python value, in SI.

    Parameters
    ----------
    keyword
        Table entry of the keyword.
    text
        Value text, without the unit.
    unit
        Unit given in the message, if any.

    Raises
    ------
    ValueError
        If the unit does not match, or the text is not a valid value.
    """
    if unit is not None:
        if keyword.unit is None:
            if normalize_unit(unit) not in _UNITLESS:
                raise ValueError(f"{keyword.name} takes no unit, got [{unit}]")
        elif normalize_unit(unit) != normalize_unit(keyword.unit):
            raise ValueError(f"unit [{unit}] does not match expected [{keyword.unit}]")
    if keyword.kind == "str":
        return text
    if not text:
        raise ValueError("empty value")
    # CCSDS numbers only: Python's int()/float() also take "1_0", "inf", ...
    match keyword.kind:
        case "time":
            return parse_epoch(text)
        case "int":
            if not _INTEGER.match(text):
                raise ValueError(f"invalid integer {text!r}")
            return int(text)
        case "float":
            if not _REAL.match(text):
                raise ValueError(f"invalid number {text!r}")
            number = float(text)
            if not math.isfinite(number):
                raise ValueError(f"number out of range {text!r}")
            return number * keyword.scale
        case _:
            assert_never(keyword.kind)


def format_value(keyword: Keyword, value: FieldValue) -> tuple[str, str | None]:
    """Convert a stored value back to message text and unit.

    Parameters
    ----------
    keyword
        Table entry of the keyword.
    value
        Stored (SI) value.
    """
    match keyword.kind:
        case "time":
            return format_epoch(value), None
        case "float":
            return _shortest(float(value), keyword.scale).upper(), keyword.unit
        case "str" | "int":
            return str(value), None
        case _:
            assert_never(keyword.kind)


def _shortest(value: float, scale: float) -> str:
    """Shortest number text that :func:`parse_value` turns back into ``value``.

    ``value / scale`` itself may print with spurious digits: 6535.677249 km
    is 6535677.249 m, and dividing by 1000 gives the neighbouring double
    6535.677248999999. The shortest decimal that scales back exactly is the
    one the message had.
    """
    quotient = value / scale
    for digits in range(1, 18):
        candidate = float(f"{quotient:.{digits}g}")
        if candidate * scale == value:
            return repr(candidate)
    return repr(quotient)
