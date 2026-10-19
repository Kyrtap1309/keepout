"""CCSDS CDM keyword table.

Covers every keyword of CCSDS 508.0-B-1 plus ``HBR`` and the header's
``CONJUNCTION_ID`` (CDM 2.0 / TraCSS). For
each keyword the table gives its value type, the canonical CCSDS unit, the
factor converting the file value to SI and the NDM/XML group it lives in;
values are converted by :func:`keepout.ccsds.keywords.parse_value` and
:func:`~keepout.ccsds.keywords.format_value`. Keywords not in the table (CDM
2.0 additions, ``USER_DEFINED_*``) are kept as text by the model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Final, Literal, TypeAlias

import numpy as np

from keepout._types import FloatArray
from keepout.ccsds.keywords import Keyword, ValueKind
from keepout.units import DAY, KM

__all__ = [
    "COVARIANCE_AXES",
    "HEADER",
    "KEYWORDS",
    "RELATIVE",
    "REQUIRED",
    "SEGMENT",
    "SectionKind",
    "covariance_entries",
    "covariance_matrix",
    "covariance_names",
]

SectionKind: TypeAlias = Literal["header", "relative", "segment"]


def _many(
    names: str,
    kind: ValueKind,
    unit: str | None = None,
    scale: float = 1.0,
    group: tuple[str, ...] = (),
) -> list[Keyword]:
    return [Keyword(name, kind, unit, scale, group) for name in names.split()]


def _table(*groups: Sequence[Keyword]) -> Mapping[str, Keyword]:
    return {keyword.name: keyword for group in groups for keyword in group}


COVARIANCE_AXES: Final[tuple[str, ...]] = (
    "R",
    "T",
    "N",
    "RDOT",
    "TDOT",
    "NDOT",
    "DRG",
    "SRP",
    "THR",
)
"""Covariance axes in CCSDS order: position, velocity, drag, SRP, thrust."""

# P position, V velocity, D drag, S solar radiation pressure, T thrust.
_AXIS_CLASS: Final = "PPPVVVDST"
_COVARIANCE_UNITS: Final[Mapping[tuple[str, str], str]] = {
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

    Raises
    ------
    ValueError
        If ``dimension`` is not between 1 and 9.
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


_COVARIANCE_GROUP: Final = ("data", "covarianceMatrix")
_COVARIANCE: Final = tuple(
    Keyword(
        name,
        "float",
        _COVARIANCE_UNITS[(_AXIS_CLASS[row], _AXIS_CLASS[column])],
        group=_COVARIANCE_GROUP,
    )
    for name, row, column in covariance_entries(len(COVARIANCE_AXES))
)

_RELATIVE_STATE: Final = ("relativeStateVector",)
_METADATA: Final = ("metadata",)
_OD: Final = ("data", "odParameters")
_ADDITIONAL: Final = ("data", "additionalParameters")
_STATE: Final = ("data", "stateVector")

HEADER: Final = _table(
    _many("CCSDS_CDM_VERS ORIGINATOR MESSAGE_FOR MESSAGE_ID CONJUNCTION_ID", "str"),
    _many("CREATION_DATE", "time"),
)

RELATIVE: Final = _table(
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

SEGMENT: Final = _table(
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
    _many("RECOMMENDED_OD_SPAN ACTUAL_OD_SPAN", "float", "d", DAY, _OD),
    _many("OBS_AVAILABLE OBS_USED TRACKS_AVAILABLE TRACKS_USED", "int", group=_OD),
    _many("RESIDUALS_ACCEPTED", "float", "%", group=_OD),
    _many("WEIGHTED_RMS", "float", group=_OD),
    _many("AREA_PC AREA_DRG AREA_SRP", "float", "m**2", group=_ADDITIONAL),
    _many("MASS", "float", "kg", group=_ADDITIONAL),
    _many("CD_AREA_OVER_MASS CR_AREA_OVER_MASS", "float", "m**2/kg", group=_ADDITIONAL),
    _many("THRUST_ACCELERATION", "float", "m/s**2", group=_ADDITIONAL),
    _many("SEDR", "float", "W/kg", group=_ADDITIONAL),
    _many("HBR", "float", "m", group=_ADDITIONAL),
    _many("X Y Z", "float", "km", KM, _STATE),
    _many("X_DOT Y_DOT Z_DOT", "float", "km/s", KM, _STATE),
    _COVARIANCE,
)

KEYWORDS: Final[Mapping[SectionKind, Mapping[str, Keyword]]] = {
    "header": HEADER,
    "relative": RELATIVE,
    "segment": SEGMENT,
}

REQUIRED: Final[Mapping[SectionKind, tuple[str, ...]]] = {
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
