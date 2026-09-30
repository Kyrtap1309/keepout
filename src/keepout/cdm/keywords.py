"""CCSDS CDM keyword table and value conversion.

Covers every keyword of CCSDS 508.0-B-1 plus ``HBR`` and the header's
``CONJUNCTION_ID`` (CDM 2.0 / TraCSS). For
each keyword the table gives its value type, the canonical CCSDS unit, the
factor converting the file value to SI and the NDM/XML group it lives in.
Keywords not in the table (CDM 2.0 additions, ``USER_DEFINED_*``) are kept as
text by the model.
"""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias, assert_never

import numpy as np
from astropy.time import Time
from erfa import ErfaWarning

from keepout._types import FloatArray

ValueKind: TypeAlias = Literal["str", "int", "float", "time"]
SectionKind: TypeAlias = Literal["header", "relative", "segment"]
FieldValue: TypeAlias = str | int | float | Time


@dataclass(frozen=True)
class Keyword:
    """How one CDM keyword is typed, converted and placed.

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


def _many(
    names: str,
    kind: ValueKind,
    unit: str | None = None,
    scale: float = 1.0,
    group: tuple[str, ...] = (),
) -> list[Keyword]:
    return [Keyword(name, kind, unit, scale, group) for name in names.split()]


def _table(*groups: list[Keyword]) -> dict[str, Keyword]:
    return {keyword.name: keyword for group in groups for keyword in group}


COVARIANCE_AXES = ("R", "T", "N", "RDOT", "TDOT", "NDOT", "DRG", "SRP", "THR")
"""Covariance axes in CCSDS order: position, velocity, drag, SRP, thrust."""

# P position, V velocity, D drag, S solar radiation pressure, T thrust.
_AXIS_CLASS = "PPPVVVDST"
_COVARIANCE_UNITS = {
    ("P", "P"): "m**2",
    ("V", "P"): "m**2/s",
    ("V", "V"): "m**2/s**2",
    ("D", "P"): "m**3/kg",
    ("D", "V"): "m**3/(kg*s)",
    ("D", "D"): "m**4/kg**2",
    ("S", "P"): "m**3/kg",
    ("S", "V"): "m**3/(kg*s)",
    ("S", "D"): "m**4/kg**2",
    ("S", "S"): "m**4/kg**2",
    ("T", "P"): "m**2/s**2",
    ("T", "V"): "m**2/s**3",
    ("T", "D"): "m**3/(kg*s**2)",
    ("T", "S"): "m**3/(kg*s**2)",
    ("T", "T"): "m**2/s**4",
}


def covariance_entries(dimension: int) -> tuple[tuple[str, int, int], ...]:
    """Lower-triangle covariance keywords of the leading block with their
    (row, column), row by row.

    Parameters
    ----------
    dimension
        Block size, 1 to 9 (6 = position-velocity, 9 = with drag/SRP/thrust).
    """
    if not 1 <= dimension <= len(COVARIANCE_AXES):
        raise ValueError(f"covariance dimension must be 1..9, got {dimension}")
    rows, columns = np.tril_indices(dimension)
    return tuple(
        (f"C{COVARIANCE_AXES[row]}_{COVARIANCE_AXES[column]}", int(row), int(column))
        for row, column in zip(rows, columns, strict=True)
    )


def covariance_matrix(dimension: int, value_of: Callable[[str], float]) -> FloatArray:
    """Symmetric covariance of the leading block from its lower-triangle keywords.

    Parameters
    ----------
    dimension
        Block size, as in :func:`covariance_entries`.
    value_of
        Value of a covariance keyword.
    """
    matrix = np.zeros((dimension, dimension))
    for name, row, column in covariance_entries(dimension):
        matrix[row, column] = matrix[column, row] = value_of(name)
    return matrix


def covariance_names(dimension: int) -> tuple[str, ...]:
    """Lower-triangle covariance keywords of the leading block, row by row.

    Parameters
    ----------
    dimension
        Block size, 1 to 9.
    """
    return tuple(name for name, _, _ in covariance_entries(dimension))


_COVARIANCE_GROUP = ("data", "covarianceMatrix")
_COVARIANCE = [
    Keyword(
        name,
        "float",
        _COVARIANCE_UNITS[(_AXIS_CLASS[row], _AXIS_CLASS[column])],
        group=_COVARIANCE_GROUP,
    )
    for name, row, column in covariance_entries(len(COVARIANCE_AXES))
]

_RELATIVE_STATE = ("relativeStateVector",)
_METADATA = ("metadata",)
_OD = ("data", "odParameters")
_ADDITIONAL = ("data", "additionalParameters")
_STATE = ("data", "stateVector")

HEADER = _table(
    _many("CCSDS_CDM_VERS ORIGINATOR MESSAGE_FOR MESSAGE_ID CONJUNCTION_ID", "str"),
    _many("CREATION_DATE", "time"),
)

RELATIVE = _table(
    _many(
        "TCA START_SCREEN_PERIOD STOP_SCREEN_PERIOD SCREEN_ENTRY_TIME SCREEN_EXIT_TIME",
        "time",
    ),
    _many(
        "MISS_DISTANCE SCREEN_VOLUME_X SCREEN_VOLUME_Y SCREEN_VOLUME_Z", "float", "m"
    ),
    _many("RELATIVE_SPEED", "float", "m/s"),
    _many(
        "RELATIVE_POSITION_R RELATIVE_POSITION_T RELATIVE_POSITION_N",
        "float",
        "m",
        group=_RELATIVE_STATE,
    ),
    _many(
        "RELATIVE_VELOCITY_R RELATIVE_VELOCITY_T RELATIVE_VELOCITY_N",
        "float",
        "m/s",
        group=_RELATIVE_STATE,
    ),
    _many(
        "SCREEN_VOLUME_FRAME SCREEN_VOLUME_SHAPE COLLISION_PROBABILITY_METHOD", "str"
    ),
    _many("COLLISION_PROBABILITY", "float"),
)

SEGMENT = _table(
    _many(
        "OBJECT OBJECT_DESIGNATOR CATALOG_NAME OBJECT_NAME INTERNATIONAL_DESIGNATOR "
        "OBJECT_TYPE OPERATOR_CONTACT_POSITION OPERATOR_ORGANIZATION OPERATOR_PHONE "
        "OPERATOR_EMAIL EPHEMERIS_NAME COVARIANCE_METHOD MANEUVERABLE ORBIT_CENTER "
        "REF_FRAME GRAVITY_MODEL ATMOSPHERIC_MODEL N_BODY_PERTURBATIONS "
        "SOLAR_RAD_PRESSURE EARTH_TIDES INTRACK_THRUST",
        "str",
        group=_METADATA,
    ),
    _many("TIME_LASTOB_START TIME_LASTOB_END", "time", group=_OD),
    _many("RECOMMENDED_OD_SPAN ACTUAL_OD_SPAN", "float", "d", 86400.0, _OD),
    _many("OBS_AVAILABLE OBS_USED TRACKS_AVAILABLE TRACKS_USED", "int", group=_OD),
    _many("RESIDUALS_ACCEPTED", "float", "%", group=_OD),
    _many("WEIGHTED_RMS", "float", group=_OD),
    _many("AREA_PC AREA_DRG AREA_SRP", "float", "m**2", group=_ADDITIONAL),
    _many("MASS", "float", "kg", group=_ADDITIONAL),
    _many("CD_AREA_OVER_MASS CR_AREA_OVER_MASS", "float", "m**2/kg", group=_ADDITIONAL),
    _many("THRUST_ACCELERATION", "float", "m/s**2", group=_ADDITIONAL),
    _many("SEDR", "float", "W/kg", group=_ADDITIONAL),
    _many("HBR", "float", "m", group=_ADDITIONAL),
    _many("X Y Z", "float", "km", 1000.0, _STATE),
    _many("X_DOT Y_DOT Z_DOT", "float", "km/s", 1000.0, _STATE),
    _COVARIANCE,
)

KEYWORDS: dict[SectionKind, dict[str, Keyword]] = {
    "header": HEADER,
    "relative": RELATIVE,
    "segment": SEGMENT,
}

REQUIRED: dict[SectionKind, tuple[str, ...]] = {
    "header": ("CCSDS_CDM_VERS", "CREATION_DATE", "ORIGINATOR", "MESSAGE_ID"),
    "relative": ("TCA", "MISS_DISTANCE"),
    "segment": (
        "OBJECT",
        "OBJECT_DESIGNATOR",
        "CATALOG_NAME",
        "OBJECT_NAME",
        "INTERNATIONAL_DESIGNATOR",
        "EPHEMERIS_NAME",
        "COVARIANCE_METHOD",
        "MANEUVERABLE",
        "REF_FRAME",
        "X",
        "Y",
        "Z",
        "X_DOT",
        "Y_DOT",
        "Z_DOT",
        *covariance_names(6),
    ),
}

_UNITLESS = {"", "n/a"}
_INTEGER = re.compile(r"^[+-]?[0-9]+$")
_REAL = re.compile(r"^[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")
_EPOCH = re.compile(
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


def parse_epoch(text: str) -> Time:
    """Parse a CCSDS UTC epoch.

    Accepts ``YYYY-MM-DDThh:mm:ss[.f]`` and ``YYYY-DDDThh:mm:ss[.f]`` with an
    optional trailing ``Z``. The returned Time keeps the number of fractional
    digits as its precision, so :func:`format_epoch` writes it back unchanged.
    Every field must be in range: a day past the end of its month or year,
    hour 24, or second 60 outside a UTC leap second is rejected rather than
    rolled over to another time.

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
    try:
        with warnings.catch_warnings():
            # ERFA only warns about a second 60 outside a leap second.
            warnings.simplefilter("error", ErfaWarning)
            epoch = Time(
                body,
                format=time_format,
                scale="utc",
                precision=min(len(fraction or ""), 9),
            )
    except (ValueError, ErfaWarning) as exc:
        raise ValueError(f"invalid CCSDS epoch {text!r}: {exc}") from exc
    # astropy rolls some out-of-range fields over (day 366 of a common year
    # becomes 1 January): the parsed fields must survive unchanged.
    date = f"{year}:{doy}" if doy else f"{year}-{month}-{day}"
    parsed = epoch.yday[:8] if doy else epoch.isot[:10]
    if parsed != date or epoch.isot[11:19] != f"{hour}:{minute}:{second}":
        raise ValueError(f"invalid CCSDS epoch {text!r}: a field is out of range")
    return epoch


def format_epoch(epoch: Time) -> str:
    """Write an epoch in CCSDS calendar format, UTC.

    Parameters
    ----------
    epoch
        Epoch to write; its precision sets the number of fractional digits.
    """
    return str(epoch.utc.isot)


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
