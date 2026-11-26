"""CCSDS UTC epochs: reading and writing their text.

:func:`parse_epoch` and :func:`format_epoch` read and write CCSDS UTC
epochs, :func:`preparse_epochs` reads many at once, and :func:`utc_fields`
gives the fields :func:`format_epoch` writes without the text. Epochs before
1972-01-01 are refused: CDMs and OMMs never carry them, and until then UTC
stepped by fractions of a second, which ERFA's text conversion gets wrong.
"""

from __future__ import annotations

import calendar
import contextlib
import datetime
import re
import warnings
from typing import TYPE_CHECKING, Final

import erfa.ufunc
from astropy.time import Time
from erfa import ErfaWarning

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

__all__ = ["format_epoch", "parse_epoch", "preparse_epochs", "utc_fields"]

_EPOCH: Final = re.compile(
    r"^(?P<year>\d{4})-(?:(?P<month>\d{2})-(?P<day>\d{2})|(?P<doy>\d{3}))"
    r"T(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})(?:\.(?P<fraction>\d+))?Z?$"
)

_FIRST_YEAR: Final = 1972
"""The first year of UTC read and written: leap seconds since, no
fractional steps."""

_FIRST_JD: Final = 2441317.5
"""Julian date of 1972-01-01 0h UTC."""

_BEFORE_1972: Final = (
    "UTC epochs before 1972-01-01 are not read or written "
    "(CDMs and OMMs never carry them)"
)

_MAX_DIGITS: Final = 9
"""Fractional digits astropy keeps."""


def parse_epoch(text: str) -> Time:
    """The time of a CCSDS UTC epoch.

    Accepts ``YYYY-MM-DDThh:mm:ss[.f]`` and ``YYYY-DDDThh:mm:ss[.f]`` with an
    optional trailing ``Z``, from 1972 on. The returned Time keeps the number
    of fractional digits (at most nine, rounded) as its precision, so
    :func:`format_epoch` writes it back unchanged. Every field must be in
    range: a day past the end of its month or year, hour 24, or second 60
    outside a UTC leap second is rejected rather than rolled over to another
    time. Years a few years past ERFA's last leap second are accepted.

    Parameters
    ----------
    text
        Epoch as written in the message.

    Returns
    -------
    astropy.time.Time
        The epoch, in the UTC scale.

    Raises
    ------
    ValueError
        If the text is not a valid CCSDS epoch, or is before 1972-01-01.
    """
    match = _EPOCH.match(text.strip())
    if match is None:
        raise ValueError(f"invalid CCSDS epoch {text!r}")
    if int(match["year"]) < _FIRST_YEAR:
        raise ValueError(f"invalid CCSDS epoch {text!r}: {_BEFORE_1972}")
    body, time_format, precision = _epoch_body(match)
    with _dubious_years_allowed():
        epoch = _utc(body, time_format, precision, text)
        # astropy rolls some out-of-range fields over (day 366 of a common
        # year becomes 1 January): the parsed fields must survive unchanged.
        # More than nine fraction digits may round up to the next second, so
        # then the whole second is checked.
        checked = epoch
        if len(match["fraction"] or "") > _MAX_DIGITS:
            checked = _utc(body.partition(".")[0], time_format, 0, text)
        if not _fields_kept(match, str(checked.isot)):
            raise ValueError(f"invalid CCSDS epoch {text!r}: a field is out of range")
    return epoch


def preparse_epochs(texts: Sequence[str]) -> list[Time | None]:
    """The common epochs of ``texts`` parsed together, None for the others.

    Each Time is the one :func:`parse_epoch` returns for its text. None is
    left where the text is invalid or unusual: not ASCII, a field out of
    range, a leap second, a year before 1972, or a fraction rounded up to
    the next second, where :func:`parse_epoch` reads it (and raises its
    error).

    The texts are parsed by format and number of fractional digits in one
    astropy call each. Only texts whose fields are all in range go in, so
    astropy's fast parser reads every one of them, as it reads a single
    text; and each result is kept only if it passes :func:`parse_epoch`'s
    check that the fields survive unchanged.

    Parameters
    ----------
    texts
        Epochs as written in a message.

    Returns
    -------
    list[astropy.time.Time | None]
        One entry per text: its time (UTC) if it is a common epoch, else None.
    """
    epochs: list[Time | None] = [None] * len(texts)
    groups: dict[tuple[str, int], list[tuple[int, re.Match[str], str]]] = {}
    for index, text in enumerate(texts):
        match = _EPOCH.match(text.strip())
        if match is None or not _batchable(match):
            continue
        body, time_format, precision = _epoch_body(match)
        groups.setdefault((time_format, precision), []).append((index, match, body))
    for (time_format, precision), members in groups.items():
        with _dubious_years_allowed():
            try:
                batch = Time(
                    [body for _, _, body in members],
                    format=time_format,
                    scale="utc",
                    precision=precision,
                )
                isots = batch.isot
            except (ValueError, ErfaWarning):
                continue  # parse_epoch reads them one by one
        for position, (index, match, _) in enumerate(members):
            if _fields_kept(match, str(isots[position])):
                epochs[index] = batch[position]
    return epochs


def format_epoch(epoch: Time) -> str:
    """An epoch in CCSDS calendar format, UTC.

    Years past ERFA's leap-second table are written without a warning
    (see :func:`parse_epoch`).

    Parameters
    ----------
    epoch
        Epoch to write; its precision sets the number of fractional digits.

    Returns
    -------
    str
        The epoch as ``YYYY-MM-DDThh:mm:ss[.f]``, UTC.

    Raises
    ------
    ValueError
        If the epoch is before 1972-01-01.
    """
    utc = epoch.utc
    if _before_1972(utc):
        raise ValueError(_BEFORE_1972)
    with _dubious_years_allowed():
        return str(utc.isot)


def utc_fields(
    epoch: Time, digits: int
) -> tuple[int, int, int, int, int, int, int] | None:
    """The fields :func:`format_epoch` writes for an epoch, read without text.

    Year, month, day, hour, minute, second and fraction of a second (in
    units of ``10**-digits`` s) of ``format_epoch(Time(epoch,
    precision=digits))``, many times faster. None where that text is not
    plain ``YYYY-MM-DDThh:mm:ss[.f]`` from ERFA alone: a year past 9999, an
    epoch before 1972 (which :func:`format_epoch` refuses), a masked,
    non-scalar or invalid epoch or one with another ``out_subfmt``. A leap
    second is given as second 60.

    Parameters
    ----------
    epoch
        A scalar epoch.
    digits
        Number of fractional-second digits, 0 to 9.

    Returns
    -------
    tuple[int, ...] or None
        Year, month, day, hour, minute, second and fraction, or None.
    """
    utc = epoch.utc
    if utc.masked or not utc.isscalar or utc.out_subfmt != "*" or _before_1972(utc):
        return None
    # The raw ufunc: a dubious year (status 1) is as good as any other.
    year, month, day, clock, status = erfa.ufunc.d2dtf(b"UTC", digits, utc.jd1, utc.jd2)
    if status < 0 or year > 9999:
        return None
    hour, minute, second, fraction = clock.item()
    return int(year), int(month), int(day), hour, minute, second, fraction


def _before_1972(utc: Time) -> bool:
    """Whether a scalar UTC time is before 1972-01-01."""
    return bool((utc.jd1 - _FIRST_JD) + utc.jd2 < 0)


def _utc(body: str, time_format: str, precision: int, text: str) -> Time:
    """The UTC time of an epoch's astropy text.

    Raises
    ------
    ValueError
        If astropy refuses it or ERFA warns (a second 60 outside a leap second).
    """
    try:
        return Time(body, format=time_format, scale="utc", precision=precision)
    except (ValueError, ErfaWarning) as exc:
        raise ValueError(f"invalid CCSDS epoch {text!r}: {exc}") from exc


@contextlib.contextmanager
def _dubious_years_allowed() -> Generator[None]:
    """ERFA's "dubious year" warning is ignored, every other ERFA warning raised.

    ERFA calls a UTC year "dubious" more than about five years past its
    leap-second table's release (from 2029 with the tables of 2026). There
    the UTC-TAI offset is extrapolated; the calendar time itself is exact,
    and an element set or CDM epoch there is valid.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", ErfaWarning)
        warnings.filterwarnings("ignore", r".*dubious year", ErfaWarning)
        yield


def _batchable(match: re.Match[str]) -> bool:
    """Whether an epoch's fields are ASCII and in range, without leap second,
    from 1972 on: astropy's fast parser reads such an epoch without falling
    back to its Python parser, alone as in an array."""
    year = int(match["year"])
    clock = int(match["hour"]) <= 23 and int(match["minute"]) <= 59
    if not (match.group().isascii() and clock and int(match["second"]) <= 59):
        return False
    if match["doy"] is not None:
        days = 1 <= int(match["doy"]) <= 365 + calendar.isleap(year)
    else:
        month, day = int(match["month"]), int(match["day"])
        days = 1 <= month <= 12 and 1 <= day <= calendar.monthrange(year, month)[1]
    return days and year >= _FIRST_YEAR


def _epoch_body(match: re.Match[str]) -> tuple[str, str, int]:
    """Text, format and precision of a matched CCSDS epoch for astropy."""
    year, month, day, doy, hour, minute, second, fraction = match.groups()
    clock = f"{hour}:{minute}:{second}" + (f".{fraction}" if fraction else "")
    precision = min(len(fraction or ""), _MAX_DIGITS)
    if doy is None:
        return f"{year}-{month}-{day}T{clock}", "isot", precision
    return f"{year}:{doy}:{clock}", "yday", precision


def _fields_kept(match: re.Match[str], isot: str) -> bool:
    """Whether the parsed epoch (``isot``) has the date and clock as written."""
    year, month, day, doy, hour, minute, second, _ = match.groups()
    date = f"{year}:{doy}" if doy else f"{year}-{month}-{day}"
    parsed = _day_of_year(isot) if doy else isot[:10]
    return parsed == date and isot[11:19] == f"{hour}:{minute}:{second}"


def _day_of_year(isot: str) -> str:
    """``YYYY:DDD`` of the date of an ``isot`` text."""
    day = datetime.date.fromisoformat(isot[:10])
    return f"{isot[:4]}:{day.timetuple().tm_yday:03d}"
