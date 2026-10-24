"""In-memory Orbit Mean-Elements Message: one object's SGP4 mean elements.

An :class:`Omm` holds what the 18th Space Defense Squadron publishes for one
cataloged object as general perturbations (GP) data, read from CCSDS OMM
(CCSDS 502.0-B-3) or CelesTrak's CSV and JSON: the object's identity and its
SGP4/SDP4 mean elements in SI units.
:func:`keepout.propagation.satrec_of` initialises SGP4 from it exactly as
:func:`sgp4.omm.initialize` does from the message.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final, NamedTuple

from astropy.time import Time

from keepout.ccsds.epochs import format_epoch, parse_epoch
from keepout.ccsds.keywords import format_value
from keepout.omm.keywords import ATTRIBUTE_KEYWORDS, KEYWORDS, SGP4_EPHEMERIS_TYPE

__all__ = [
    "MAX_CATALOG_NUMBER",
    "MAX_ELEMENT_SET_NO",
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

_REALS: Final = tuple(
    name for name, keyword in ATTRIBUTE_KEYWORDS.items() if keyword.kind == "float"
)
"""The real-valued attributes, in table order."""

_TEXTS: Final = tuple(
    name for name, keyword in ATTRIBUTE_KEYWORDS.items() if keyword.kind == "str"
)
"""The text attributes, in table order."""

SGP4_EPOCH_DIGITS: Final = 6
"""Fractional-second digits of the epoch :func:`sgp4.omm.initialize` reads:
microseconds, the resolution of CelesTrak's epochs."""

_MAX_EPOCH_DIGITS: Final = 9
"""Most fractional-second digits astropy writes (nanoseconds)."""


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
    """The SGP4 mean elements of one cataloged object (CCSDS OMM).

    Units are SI except for ``bstar``, which SGP4 defines in inverse Earth
    radii. Mean elements belong to the SGP4/SDP4 theory and must be
    propagated with it (:func:`keepout.propagation.satrec_of`); they are not
    osculating elements.

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
        for name in _TEXTS:
            if not isinstance(getattr(self, name), str):
                raise InvalidElement(name, "must be a string")
        if len(self.classification) != 1:
            raise InvalidElement(
                "classification", f"must be one character, got {self.classification!r}"
            )
        _coerce_integer(self, "catalog_number", MAX_CATALOG_NUMBER)
        _coerce_integer(self, "element_set_no", MAX_ELEMENT_SET_NO)
        _coerce_integer(self, "rev_at_epoch")
        epoch: object = self.epoch  # checked: callers need not be type-checked
        if not (isinstance(epoch, Time) and epoch.isscalar):
            raise InvalidElement("epoch", "must be a scalar astropy Time")
        object.__setattr__(self, "epoch", self.epoch.utc)
        for name in _REALS:
            _coerce_real(self, name)
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

    def keyword_texts(self, *, epoch_digits: int | None = None) -> dict[str, str]:
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
            keyword.name: format_value(keyword, getattr(self, attribute))[0]
            for attribute, keyword in ATTRIBUTE_KEYWORDS.items()
            if keyword.name != "EPOCH"
        }
        values["EPOCH"] = _epoch_text(self.epoch, epoch_digits)
        values["EPHEMERIS_TYPE"] = str(SGP4_EPHEMERIS_TYPE)
        return {keyword: values[keyword] for keyword in KEYWORDS}


class Deduplicated(NamedTuple):
    """Element sets split by :func:`deduplicate`."""

    kept: tuple[Omm, ...]
    """The newest element set of each object, in the order the objects first
    appear."""
    dropped: tuple[Omm, ...]
    """The other element sets, in input order; an exact copy of a set
    already kept or dropped is not listed again."""

    @property
    def by_number(self) -> dict[int, Omm]:
        """The kept element sets by catalog number, in :attr:`kept` order."""
        return {omm.catalog_number: omm for omm in self.kept}


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
        if omm in seen:
            continue
        seen.append(omm)
        order.append(omm)
        best = newest.get(omm.catalog_number)
        if best is None or omm.epoch > best.epoch:
            newest[omm.catalog_number] = omm
    kept = tuple(newest.values())
    kept_ids = {id(omm) for omm in kept}
    return Deduplicated(kept, tuple(omm for omm in order if id(omm) not in kept_ids))


def _epoch_text(epoch: Time, digits: int | None) -> str:
    """CCSDS text of an epoch, whatever its ``precision`` (see :meth:`Omm.keywords`)."""
    if digits is not None:
        return format_epoch(Time(epoch, precision=digits))
    for fewest in range(SGP4_EPOCH_DIGITS, _MAX_EPOCH_DIGITS):
        text = format_epoch(Time(epoch, precision=fewest))
        if parse_epoch(text) == epoch:
            return text
    return format_epoch(Time(epoch, precision=_MAX_EPOCH_DIGITS))


def _coerce_integer(omm: Omm, name: str, maximum: int | None = None) -> None:
    value = getattr(omm, name)
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise InvalidElement(name, f"must be an integer, got {value!r}")
    value = int(value)
    object.__setattr__(omm, name, value)
    if value < 0 or (maximum is not None and value > maximum):
        bound = f"be in [0, {maximum}]" if maximum is not None else "be >= 0"
        raise InvalidElement(name, f"must {bound}, got {value}")


def _coerce_real(omm: Omm, name: str) -> None:
    value = getattr(omm, name)
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise InvalidElement(name, f"must be a real number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise InvalidElement(name, f"must be finite, got {value}")
    object.__setattr__(omm, name, value)
