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

from typing import TYPE_CHECKING, Final, Literal, TypeAlias

import numpy as np

from keepout.ccsds.keywords import Keyword, ValueKind
from keepout.units import DAY, KM

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from keepout._types import FloatArray

__all__ = [
    "ADDITIONAL_PARAMETERS",
    "COVARIANCE_AXES",
    "HEADER",
    "KEYWORDS",
    "METADATA_GROUP",
    "PHYSICAL_PARAMETERS",
    "POSITION_KEYWORDS",
    "RELATIVE",
    "REQUIRED",
    "SEGMENT",
    "STATE_KEYWORDS",
    "VELOCITY_KEYWORDS",
    "VERSION_1_0",
    "VERSION_2_0",
    "VERSION_KEYWORD",
    "SectionKind",
    "covariance_entries",
    "covariance_matrix",
    "covariance_names",
]

SectionKind: TypeAlias = Literal["header", "relative", "segment"]
"""The kind of a CDM section: the header, the relative metadata/data, or an
object section (``"segment"``, the NDM/XML element name of an object section)."""

VERSION_KEYWORD: Final = "CCSDS_CDM_VERS"
"""Keyword of the CDM version, the first of every message."""

VERSION_1_0: Final = "1.0"
"""``CCSDS_CDM_VERS`` of CCSDS 508.0-B-1 messages."""

VERSION_2_0: Final = "2.0"
"""``CCSDS_CDM_VERS`` of CDM 2.0 messages (TraCSS), the version with ``HBR``."""

POSITION_KEYWORDS: Final[tuple[str, str, str]] = ("X", "Y", "Z")
"""Position keywords of an object section, in order."""

VELOCITY_KEYWORDS: Final[tuple[str, str, str]] = ("X_DOT", "Y_DOT", "Z_DOT")
"""Velocity keywords of an object section, in order."""

STATE_KEYWORDS: Final[tuple[str, ...]] = (*POSITION_KEYWORDS, *VELOCITY_KEYWORDS)
"""State-vector keywords of an object section: position, then velocity."""


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

    Returns
    -------
    tuple[tuple[str, int, int], ...]
        Each keyword with its row and column.

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

    Returns
    -------
    FloatArray
        The symmetric matrix, in the units of the values given.
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

    Returns
    -------
    tuple of str
        The keywords, row by row.
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

METADATA_GROUP: Final = ("metadata",)
"""NDM/XML group of an object section's metadata."""

ADDITIONAL_PARAMETERS: Final = "additionalParameters"
"""NDM/XML group of an object's physical parameters in CDM 1.0 (CCSDS
508.0-B-1), inside its ``data``."""

PHYSICAL_PARAMETERS: Final = "physicalParameters"
"""The CDM 2.0 name of :data:`ADDITIONAL_PARAMETERS`."""

_RELATIVE_STATE: Final = ("relativeStateVector",)
_METADATA: Final = METADATA_GROUP
_OD: Final = ("data", "odParameters")
_ADDITIONAL: Final = ("data", ADDITIONAL_PARAMETERS)
_STATE: Final = ("data", "stateVector")

HEADER: Final = _table(
    _many("CCSDS_CDM_VERS ORIGINATOR MESSAGE_FOR MESSAGE_ID CONJUNCTION_ID", "str"),
    _many("CREATION_DATE", "time"),
)
"""The keywords of a CDM's header, by name: each with its type (and so its
parser)."""

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
"""The keywords of a CDM's relative metadata and data (TCA, miss distance,
screening volume, relative state, Pc), by name: each with its type, unit and
NDM/XML group."""

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
"""The keywords of an object's segment (metadata, OD parameters, physical
parameters, state vector and covariance), by name: each with its type, unit
and NDM/XML group. States are in km and km/s in the message, kept in SI
units."""

KEYWORDS: Final[Mapping[SectionKind, Mapping[str, Keyword]]] = {
    "header": HEADER,
    "relative": RELATIVE,
    "segment": SEGMENT,
}
"""Every keyword a CDM section can hold, by section: :data:`HEADER`,
:data:`RELATIVE` and :data:`SEGMENT` (each object has one)."""

REQUIRED: Final[Mapping[SectionKind, tuple[str, ...]]] = {
    "header": (VERSION_KEYWORD, "CREATION_DATE", "ORIGINATOR", "MESSAGE_ID"),
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
        *STATE_KEYWORDS,
        *covariance_names(6),
    ),
}
"""The keywords each section must give a value, by section: a
:class:`~keepout.cdm.Cdm` without one is refused."""
