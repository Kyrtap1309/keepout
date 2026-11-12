"""CCSDS OMM keywords of SGP4 mean elements.

The keywords are those of CCSDS 502.0-B-3 (Orbit Data Messages) that hold the
SGP4/SDP4 mean elements and TLE-related parameters, the ones CelesTrak and
Space-Track publish as CSV columns and JSON keys. Each entry gives the value
type, the CCSDS unit, the factor converting the message value to keepout's
unit (SI where one exists) and the OMM XML element group. Values are converted
by :func:`keepout.ccsds.keywords.parse_value` and
:func:`~keepout.ccsds.keywords.format_value`, so the message value is recovered
exactly when it is written back.
"""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from keepout.ccsds.keywords import Keyword
from keepout.units import DAY

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "ATTRIBUTES",
    "ATTRIBUTE_KEYWORDS",
    "DEFAULTS",
    "DEGREE",
    "FIXED_METADATA",
    "KEYWORDS",
    "REVOLUTION_PER_DAY",
    "SGP4_EPHEMERIS_TYPE",
    "VERSION",
    "VERSIONS",
    "VERSION_KEYWORD",
]

DEGREE: Final = math.pi / 180.0
"""Radians per degree, computed as :mod:`sgp4.omm` does, so a converted angle
is bit-identical to the one SGP4 is initialised with."""

REVOLUTION_PER_DAY: Final = math.tau / DAY
"""Radians per second in one revolution per day."""

SGP4_EPHEMERIS_TYPE: Final = 0
"""``EPHEMERIS_TYPE`` of SGP4/SDP4 elements (4 is SGP4-XP, not supported)."""

_METADATA: Final = ("metadata",)
_MEAN_ELEMENTS: Final = ("data", "meanElements")
_TLE_PARAMETERS: Final = ("data", "tleParameters")

KEYWORDS: Final[Mapping[str, Keyword]] = MappingProxyType(
    {
        keyword.name: keyword
        for keyword in (
            Keyword("OBJECT_NAME", "str", group=_METADATA),
            Keyword("OBJECT_ID", "str", group=_METADATA),
            Keyword("EPOCH", "time", group=_MEAN_ELEMENTS),
            Keyword(
                "MEAN_MOTION", "float", "rev/day", REVOLUTION_PER_DAY, _MEAN_ELEMENTS
            ),
            Keyword("ECCENTRICITY", "float", group=_MEAN_ELEMENTS),
            Keyword("INCLINATION", "float", "deg", DEGREE, _MEAN_ELEMENTS),
            Keyword("RA_OF_ASC_NODE", "float", "deg", DEGREE, _MEAN_ELEMENTS),
            Keyword("ARG_OF_PERICENTER", "float", "deg", DEGREE, _MEAN_ELEMENTS),
            Keyword("MEAN_ANOMALY", "float", "deg", DEGREE, _MEAN_ELEMENTS),
            Keyword("EPHEMERIS_TYPE", "int", group=_TLE_PARAMETERS),
            Keyword("CLASSIFICATION_TYPE", "str", group=_TLE_PARAMETERS),
            Keyword("NORAD_CAT_ID", "int", group=_TLE_PARAMETERS),
            Keyword("ELEMENT_SET_NO", "int", group=_TLE_PARAMETERS),
            Keyword("REV_AT_EPOCH", "int", group=_TLE_PARAMETERS),
            Keyword("BSTAR", "float", "1/ER", group=_TLE_PARAMETERS),
            Keyword(
                "MEAN_MOTION_DOT",
                "float",
                "rev/day**2",
                REVOLUTION_PER_DAY / DAY,
                _TLE_PARAMETERS,
            ),
            Keyword(
                "MEAN_MOTION_DDOT",
                "float",
                "rev/day**3",
                REVOLUTION_PER_DAY / DAY**2,
                _TLE_PARAMETERS,
            ),
        )
    }
)
"""The keywords of one object's SGP4 elements, in CelesTrak's CSV column order."""

ATTRIBUTES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "OBJECT_NAME": "object_name",
        "OBJECT_ID": "international_designator",
        "EPOCH": "epoch",
        "MEAN_MOTION": "mean_motion",
        "ECCENTRICITY": "eccentricity",
        "INCLINATION": "inclination",
        "RA_OF_ASC_NODE": "ra_of_asc_node",
        "ARG_OF_PERICENTER": "arg_of_pericenter",
        "MEAN_ANOMALY": "mean_anomaly",
        "CLASSIFICATION_TYPE": "classification",
        "NORAD_CAT_ID": "catalog_number",
        "ELEMENT_SET_NO": "element_set_no",
        "REV_AT_EPOCH": "rev_at_epoch",
        "BSTAR": "bstar",
        "MEAN_MOTION_DOT": "mean_motion_dot",
        "MEAN_MOTION_DDOT": "mean_motion_ddot",
    }
)
"""The :class:`~keepout.omm.model.Omm` attribute of each keyword but
``EPHEMERIS_TYPE``, which is always :data:`SGP4_EPHEMERIS_TYPE`."""

ATTRIBUTE_KEYWORDS: Final[Mapping[str, Keyword]] = MappingProxyType(
    {attribute: KEYWORDS[keyword] for keyword, attribute in ATTRIBUTES.items()}
)
"""The keyword of each :class:`~keepout.omm.model.Omm` attribute: its name
and how its message value converts."""

FIXED_METADATA: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "CENTER_NAME": ("EARTH",),
        "REF_FRAME": ("TEME",),
        "TIME_SYSTEM": ("UTC",),
        "MEAN_ELEMENT_THEORY": ("SGP4", "SGP/SGP4"),
    }
)
"""Metadata every SGP4 element set has, with the values accepted for each; a
message that states one must state one of them (CelesTrak's CSV and JSON
state none), and writers write the first. CCSDS 502.0-B-3's KVN examples
(annex G) write the theory as ``SGP/SGP4``, its XML example and CelesTrak as
``SGP4``."""

DEFAULTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "EPHEMERIS_TYPE": str(SGP4_EPHEMERIS_TYPE),
        "CLASSIFICATION_TYPE": "U",
    }
)
"""Values of the TLE parameters CCSDS 502.0-B-3 (table 4-3) lets an OMM omit."""

VERSION_KEYWORD: Final = "CCSDS_OMM_VERS"
"""Keyword of the OMM version, the first of every KVN message."""

VERSIONS: Final = ("2.0", "3.0")
"""``CCSDS_OMM_VERS`` values read: CCSDS 502.0-B-2 and 502.0-B-3."""

VERSION: Final = "3.0"
"""``CCSDS_OMM_VERS`` written (CCSDS 502.0-B-3)."""
