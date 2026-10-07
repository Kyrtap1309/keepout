"""In-memory Orbit Mean-Elements Message: one object's SGP4 mean elements.

An :class:`Omm` holds what the 18th Space Defense Squadron publishes for one
catalogued object as general perturbations (GP) data, read from CCSDS OMM
(CCSDS 502.0-B-3) or CelesTrak's CSV and JSON: the object's identity and its
SGP4/SDP4 mean elements in SI units. :meth:`Omm.satrec` initialises SGP4 from
it exactly as :func:`sgp4.omm.initialize` does from the message.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Final

from astropy.time import Time
from sgp4 import omm as sgp4_omm
from sgp4.api import Satrec

from keepout.cdm.keywords import format_epoch, format_value
from keepout.omm.keywords import ATTRIBUTES, KEYWORDS, SGP4_EPHEMERIS_TYPE

__all__ = [
    "MAX_CATALOG_NUMBER",
    "MAX_ELEMENT_SET_NO",
    "MAX_SATREC_NUMBER",
    "InvalidElement",
    "Omm",
]

MAX_CATALOG_NUMBER: Final = 999_999_999
"""Largest ``NORAD_CAT_ID`` of an OMM (nine digits)."""

MAX_ELEMENT_SET_NO: Final = 9999
"""Largest ``ELEMENT_SET_NO`` (four digits, as in a TLE and the OMM schema)."""

MAX_SATREC_NUMBER: Final = 339_999
"""Largest catalog number SGP4's satellite record holds (Alpha-5 ``Z9999``)."""

_REALS: Final = (
    "mean_motion",
    "eccentricity",
    "inclination",
    "ra_of_asc_node",
    "arg_of_pericenter",
    "mean_anomaly",
    "bstar",
    "mean_motion_dot",
    "mean_motion_ddot",
)

_SGP4_EPOCH_DIGITS: Final = 6
"""Fractional-second digits of the epoch :func:`sgp4.omm.initialize` reads."""


class InvalidElement(ValueError):
    """An :class:`Omm` attribute has an invalid value.

    Parameters
    ----------
    attribute
        Name of the attribute.
    message
        What is wrong.
    """

    def __init__(self, attribute: str, message: str) -> None:
        super().__init__(f"{attribute} {message}")
        self.attribute = attribute


@dataclass(frozen=True)
class Omm:
    """The SGP4 mean elements of one catalogued object (CCSDS OMM).

    Units are SI except for ``bstar``, which SGP4 defines in inverse Earth
    radii. Mean elements belong to the SGP4/SDP4 theory and must be
    propagated with it (:meth:`satrec`); they are not osculating elements.

    Parameters
    ----------
    object_name
        Name of the object (``OBJECT_NAME``).
    international_designator
        COSPAR international designator, e.g. ``1998-067A`` (``OBJECT_ID``).
    catalog_number
        Satellite catalog number (``NORAD_CAT_ID``), six digits since 2026.
    epoch
        Epoch of the elements, a scalar time; stored in UTC.
    mean_motion
        Kozai mean motion [rad/s] (``MEAN_MOTION``, in rev/day in a message).
    eccentricity
        Mean eccentricity, in [0, 1).
    inclination
        Mean inclination [rad], in [0, pi].
    ra_of_asc_node
        Mean right ascension of the ascending node [rad], TEME.
    arg_of_pericenter
        Mean argument of perigee [rad].
    mean_anomaly
        Mean anomaly at epoch [rad].
    bstar
        SGP4 drag term B* [1/Earth radii].
    mean_motion_dot
        ``MEAN_MOTION_DOT`` [rad/s^2]: the TLE's first derivative of mean
        motion divided by two. SGP4 does not use it.
    mean_motion_ddot
        ``MEAN_MOTION_DDOT`` [rad/s^3]: the TLE's second derivative of mean
        motion divided by six. SGP4 does not use it.
    classification
        Security classification, one character: ``U`` (unclassified), ``C``
        or ``S`` (``CLASSIFICATION_TYPE``).
    element_set_no
        Element set number (``ELEMENT_SET_NO``), 0 to 9999.
    rev_at_epoch
        Revolution number at epoch (``REV_AT_EPOCH``).

    Raises
    ------
    InvalidElement
        If a value has the wrong type, is not finite or is out of range.
    """

    object_name: str
    international_designator: str
    catalog_number: int
    epoch: Time
    mean_motion: float
    eccentricity: float
    inclination: float
    ra_of_asc_node: float
    arg_of_pericenter: float
    mean_anomaly: float
    bstar: float
    mean_motion_dot: float
    mean_motion_ddot: float
    classification: str
    element_set_no: int
    rev_at_epoch: int

    def __post_init__(self) -> None:
        for name in ("object_name", "international_designator", "classification"):
            if not isinstance(getattr(self, name), str):
                raise InvalidElement(name, "must be a string")
        if len(self.classification) != 1:
            raise InvalidElement(
                "classification", f"must be one character, got {self.classification!r}"
            )
        _check_integer(self, "catalog_number", MAX_CATALOG_NUMBER)
        _check_integer(self, "element_set_no", MAX_ELEMENT_SET_NO)
        _check_integer(self, "rev_at_epoch")
        if not (isinstance(self.epoch, Time) and self.epoch.isscalar):
            raise InvalidElement("epoch", "must be a scalar astropy Time")
        object.__setattr__(self, "epoch", self.epoch.utc)
        for name in _REALS:
            _check_real(self, name)
        if self.mean_motion <= 0.0:
            raise InvalidElement("mean_motion", f"must be > 0, got {self.mean_motion}")
        if not 0.0 <= self.eccentricity < 1.0:
            raise InvalidElement(
                "eccentricity", f"must be in [0, 1), got {self.eccentricity}"
            )
        if not 0.0 <= self.inclination <= math.pi:
            raise InvalidElement(
                "inclination", f"must be in [0, pi], got {self.inclination}"
            )

    def keywords(self) -> dict[str, str]:
        """The message text of every keyword, in CCSDS units.

        Keys are in CelesTrak's CSV column order (see
        :data:`~keepout.omm.keywords.KEYWORDS`); each number is the shortest
        text that reads back as the stored value, so the values of a message
        read by keepout are given back as written, up to formatting
        (``.00245948`` becomes ``0.00245948``).
        """
        values = {
            keyword: format_value(KEYWORDS[keyword], getattr(self, attribute))[0]
            for keyword, attribute in ATTRIBUTES.items()
        }
        values["EPHEMERIS_TYPE"] = str(SGP4_EPHEMERIS_TYPE)
        return {keyword: values[keyword] for keyword in KEYWORDS}

    def satrec(self) -> Satrec:
        """A new SGP4 satellite record of these elements (WGS-72 constants).

        It is the record :func:`sgp4.omm.initialize` makes from the message
        the elements were read from, bit for bit: every number is converted
        back to the message's value exactly (see :meth:`keywords`; this holds
        for numbers of up to 15 significant digits, and CelesTrak writes at
        most 10). Only the epoch is rounded to whole microseconds, the
        resolution :func:`sgp4.omm.initialize` reads; CelesTrak's epochs have
        exactly that resolution.

        Raises
        ------
        ValueError
            If the catalog number is above :data:`MAX_SATREC_NUMBER`, or
            the epoch falls in a leap second (SGP4 counts days of 86400 s).
        """
        if self.catalog_number > MAX_SATREC_NUMBER:
            raise ValueError(
                f"catalog number {self.catalog_number} is above "
                f"{MAX_SATREC_NUMBER}, the largest an SGP4 record holds"
            )
        values = self.keywords()
        epoch = format_epoch(Time(self.epoch, precision=_SGP4_EPOCH_DIGITS))
        if epoch[17:19] == "60":
            raise ValueError(f"epoch {epoch} falls in a leap second")
        values["EPOCH"] = epoch
        satrec = Satrec()
        sgp4_omm.initialize(satrec, values)
        return satrec


def _check_integer(omm: Omm, name: str, maximum: int | None = None) -> None:
    value = getattr(omm, name)
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise InvalidElement(name, f"must be an integer, got {value!r}")
    value = int(value)
    object.__setattr__(omm, name, value)
    if value < 0 or (maximum is not None and value > maximum):
        bound = f"be in [0, {maximum}]" if maximum is not None else "be >= 0"
        raise InvalidElement(name, f"must {bound}, got {value}")


def _check_real(omm: Omm, name: str) -> None:
    value = getattr(omm, name)
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise InvalidElement(name, f"must be a real number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise InvalidElement(name, f"must be finite, got {value}")
    object.__setattr__(omm, name, value)
