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
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final, NamedTuple

from astropy.time import Time
from sgp4.api import WGS72, Satrec

from keepout.ccsds.keywords import (
    format_epoch,
    format_value,
    message_number,
    parse_epoch,
    utc_fields,
)
from keepout.errors import UnsupportedElements
from keepout.omm.keywords import ATTRIBUTES, DEGREE, KEYWORDS, SGP4_EPHEMERIS_TYPE
from keepout.units import DAY

__all__ = [
    "MAX_CATALOG_NUMBER",
    "MAX_ELEMENT_SET_NO",
    "MAX_SATREC_NUMBER",
    "SGP4_EPOCH_DIGITS",
    "Deduplicated",
    "InvalidElement",
    "Omm",
    "deduplicate",
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

SGP4_EPOCH_DIGITS: Final = 6
"""Fractional-second digits of the epoch :func:`sgp4.omm.initialize` reads:
microseconds, the resolution of CelesTrak's epochs."""

_EPOCH_SECONDS: Final = slice(17, 19)
"""The whole seconds of epoch text ``YYYY-MM-DDTHH:MM:SS.ffffff``."""

_MAX_EPOCH_DIGITS: Final = 9
"""Most fractional-second digits astropy writes (nanoseconds)."""

_SGP4_EPOCH_FORMAT: Final = "%Y-%m-%dT%H:%M:%S.%f"
"""How :func:`sgp4.omm.initialize` reads the epoch text."""

# The constants of sgp4.omm, computed as it does.
_SGP4_EPOCH0: Final = datetime(1949, 12, 31, tzinfo=UTC)
_NDOT_UNITS: Final = 1036800.0 / math.pi
_NDDOT_UNITS: Final = 2985984000.0 / 2.0 / math.pi


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


@dataclass(frozen=True, slots=True)
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
        epoch: object = self.epoch  # checked: callers need not be type-checked
        if not (isinstance(epoch, Time) and epoch.isscalar):
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

    def keywords(self, *, epoch_digits: int | None = None) -> dict[str, str]:
        """The message text of every keyword, in CCSDS units.

        Keys are in CelesTrak's CSV column order (see
        :data:`~keepout.omm.keywords.KEYWORDS`); each number is the shortest
        text that reads back as the stored value, so the values of a message
        read by keepout are given back as written, up to formatting
        (``.00245948`` becomes ``0.00245948``).

        The ``EPOCH`` text does not depend on the ``precision`` of the
        stored :class:`~astropy.time.Time`. By default it has the fewest
        fractional-second digits, at least six (microseconds), that read
        back as the stored epoch exactly; an epoch that no text of up to
        nine digits (nanoseconds, the most astropy writes) holds exactly is
        written to the nanosecond.

        Parameters
        ----------
        epoch_digits
            Write the epoch with exactly this many fractional-second digits
            (0 to 9) instead, rounded: :data:`SGP4_EPOCH_DIGITS` for the
            microseconds :func:`sgp4.omm.initialize` and CelesTrak's formats
            take.
        """
        values = {
            keyword: format_value(KEYWORDS[keyword], getattr(self, attribute))[0]
            for keyword, attribute in ATTRIBUTES.items()
            if keyword != "EPOCH"
        }
        values["EPOCH"] = _epoch_text(self.epoch, epoch_digits)
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
        UnsupportedElements
            If the catalog number is above :data:`MAX_SATREC_NUMBER`, or
            the epoch falls in a leap second (SGP4 counts days of 86400 s).
        """
        if self.catalog_number > MAX_SATREC_NUMBER:
            raise UnsupportedElements(
                f"catalog number {self.catalog_number} is above "
                f"{MAX_SATREC_NUMBER}, the largest an SGP4 record holds"
            )
        epoch = _sgp4_epoch(self.epoch)
        # What sgp4.omm.initialize does with the text of keywords(), without
        # the text: each number is the one the text reads back as.
        satrec = Satrec()
        satrec.classification = self.classification
        satrec.intldesg = self.international_designator[2:].replace("-", "")
        satrec.ephtype = SGP4_EPHEMERIS_TYPE
        satrec.elnum = self.element_set_no
        satrec.revnum = self.rev_at_epoch
        satrec.sgp4init(
            WGS72,
            "i",
            self.catalog_number,
            epoch,
            self._message_number("bstar"),
            self._message_number("mean_motion_dot") / _NDOT_UNITS,
            self._message_number("mean_motion_ddot") / _NDDOT_UNITS,
            self._message_number("eccentricity"),
            self._message_number("arg_of_pericenter") * DEGREE,
            self._message_number("inclination") * DEGREE,
            self._message_number("mean_anomaly") * DEGREE,
            self._message_number("mean_motion") / 720.0 * math.pi,
            self._message_number("ra_of_asc_node") * DEGREE,
        )
        return satrec

    def _message_number(self, attribute: str) -> float:
        """The value of an attribute in the message's units, as written."""
        return message_number(getattr(self, attribute), _SCALES[attribute])


class Deduplicated(NamedTuple):
    """Element sets split by :func:`deduplicate`."""

    kept: tuple[Omm, ...]
    """The newest element set of each object, in the order the objects first
    appear."""
    dropped: tuple[Omm, ...]
    """The other element sets, in input order; an exact copy of a set
    already kept or dropped is not listed again."""


def deduplicate(omms: Iterable[Omm]) -> Deduplicated:
    """One element set per catalog number: the one with the newest epoch.

    Catalogs put together from several sources (e.g. CelesTrak groups) hold
    some objects more than once. Of two different sets with the same epoch,
    the first given is kept. Exact copies (equal in every element) are the
    same set and not reported.

    Parameters
    ----------
    omms
        Element sets, of any objects, in any order.

    Returns
    -------
    Deduplicated
        ``kept``, one set per object, and ``dropped``, the older (or tied
        later) sets.
    """
    newest: dict[int, Omm] = {}
    distinct: dict[int, list[Omm]] = {}
    order: list[Omm] = []
    for omm in omms:
        seen = distinct.setdefault(omm.catalog_number, [])
        if any(omm == other for other in seen):
            continue
        seen.append(omm)
        order.append(omm)
        best = newest.get(omm.catalog_number)
        if best is None or omm.epoch > best.epoch:
            newest[omm.catalog_number] = omm
    kept = tuple(newest.values())
    kept_ids = {id(omm) for omm in kept}
    return Deduplicated(kept, tuple(omm for omm in order if id(omm) not in kept_ids))


_SCALES: Final = {
    attribute: KEYWORDS[keyword].scale for keyword, attribute in ATTRIBUTES.items()
}
"""Factor from message units to the attribute's, by attribute."""


def _sgp4_epoch(epoch: Time) -> float:
    """Days from 1949-12-31 0h UTC that :func:`sgp4.omm.initialize` reads.

    That is, of the epoch's text rounded to microseconds (see
    :meth:`Omm.keywords`); read from the text itself only where it is not
    plain ``YYYY-MM-DDThh:mm:ss.ffffff`` from ERFA (see
    :func:`~keepout.ccsds.keywords.utc_fields`).

    Raises
    ------
    UnsupportedElements
        If the epoch falls in a leap second.
    """
    fields = utc_fields(epoch, SGP4_EPOCH_DIGITS)
    if fields is None:
        text = _epoch_text(epoch, SGP4_EPOCH_DIGITS)
        if text[_EPOCH_SECONDS] == "60":
            raise UnsupportedElements(f"epoch {text} falls in a leap second")
        when = datetime.strptime(text, _SGP4_EPOCH_FORMAT).replace(tzinfo=UTC)
    else:
        year, month, day, hour, minute, second, microsecond = fields
        if second == 60:
            text = (
                f"{year}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:"
                f"{second:02d}.{microsecond:06d}"
            )
            raise UnsupportedElements(f"epoch {text} falls in a leap second")
        when = datetime(year, month, day, hour, minute, second, microsecond, UTC)
    return (when - _SGP4_EPOCH0).total_seconds() / DAY


def _epoch_text(epoch: Time, digits: int | None) -> str:
    """CCSDS text of an epoch, whatever its ``precision`` (see :meth:`Omm.keywords`)."""
    if digits is not None:
        return format_epoch(Time(epoch, precision=digits))
    for fewest in range(SGP4_EPOCH_DIGITS, _MAX_EPOCH_DIGITS):
        text = format_epoch(Time(epoch, precision=fewest))
        if parse_epoch(text) == epoch:
            return text
    return format_epoch(Time(epoch, precision=_MAX_EPOCH_DIGITS))


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
