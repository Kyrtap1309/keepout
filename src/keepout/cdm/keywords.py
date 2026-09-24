"""CCSDS CDM keyword table and value conversion.

Covers every keyword of CCSDS 508.0-B-1 plus ``HBR`` (CDM 2.0 / TraCSS). For
each keyword the table gives its value type, the canonical CCSDS unit, the
factor converting the file value to SI and the NDM/XML group it lives in.
Keywords not in the table (CDM 2.0 additions, ``USER_DEFINED_*``) are kept as
text by the model.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal, TypeAlias

from astropy.time import Time

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


def covariance_names(dimension: int) -> tuple[str, ...]:
    """Lower-triangle covariance keywords of the leading block, row by row.

    Parameters
    ----------
    dimension
        Block size, 1 to 9 (6 = position-velocity, 9 = with drag/SRP/thrust).
    """
    if not 1 <= dimension <= len(COVARIANCE_AXES):
        raise ValueError(f"covariance dimension must be 1..9, got {dimension}")
    return tuple(
        f"C{COVARIANCE_AXES[row]}_{COVARIANCE_AXES[column]}"
        for row in range(dimension)
        for column in range(row + 1)
    )


_COVARIANCE_GROUP = ("data", "covarianceMatrix")
_COVARIANCE = [
    Keyword(
        f"C{COVARIANCE_AXES[row]}_{COVARIANCE_AXES[column]}",
        "float",
        _COVARIANCE_UNITS[(_AXIS_CLASS[row], _AXIS_CLASS[column])],
        group=_COVARIANCE_GROUP,
    )
    for row in range(len(COVARIANCE_AXES))
    for column in range(row + 1)
]

_RELATIVE_STATE = ("relativeStateVector",)
_METADATA = ("metadata",)
_OD = ("data", "odParameters")
_ADDITIONAL = ("data", "additionalParameters")
_STATE = ("data", "stateVector")

HEADER = _table(
    _many("CCSDS_CDM_VERS ORIGINATOR MESSAGE_FOR MESSAGE_ID", "str"),
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
_CALENDAR = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.(\d+))?Z?$")
_ORDINAL = re.compile(r"^(\d{4})-(\d{3})T(\d{2}:\d{2}:\d{2}(?:\.(\d+))?)Z?$")


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

    Parameters
    ----------
    text
        Epoch as written in the message.

    Raises
    ------
    ValueError
        If the text is not a valid CCSDS epoch.
    """
    value = text.strip()
    if match := _CALENDAR.match(value):
        body, time_format, fraction = value.removesuffix("Z"), "isot", match.group(1)
    elif match := _ORDINAL.match(value):
        year, day, clock, fraction = match.groups()
        body, time_format = f"{year}:{day}:{clock}", "yday"
    else:
        raise ValueError(f"invalid CCSDS epoch {text!r}")
    try:
        return Time(
            body,
            format=time_format,
            scale="utc",
            precision=min(len(fraction or ""), 9),
        )
    except ValueError as exc:
        raise ValueError(f"invalid CCSDS epoch {text!r}: {exc}") from exc


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
    if keyword.kind == "time":
        return parse_epoch(text)
    if keyword.kind == "int":
        try:
            return int(text)
        except ValueError:
            raise ValueError(f"invalid integer {text!r}") from None
    try:
        number = float(text)
    except ValueError:
        raise ValueError(f"invalid number {text!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"non-finite number {text!r}")
    return number * keyword.scale


def format_value(keyword: Keyword, value: FieldValue) -> tuple[str, str | None]:
    """Convert a stored value back to message text and unit.

    Parameters
    ----------
    keyword
        Table entry of the keyword.
    value
        Stored (SI) value.
    """
    if keyword.kind == "time":
        return format_epoch(value), None
    if keyword.kind == "float":
        return repr(float(value) / keyword.scale).upper(), keyword.unit
    return str(value), None
