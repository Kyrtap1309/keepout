"""SGP4 for many objects at once, behind the
:class:`~keepout.propagation.base.Propagator` interface, and the SGP4 record
of one element set (:func:`satrec_of`)."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from typing import Final, Literal, NamedTuple, Self, TypeAlias

import erfa
import numpy as np
import numpy.typing as npt
from astropy.time import Time
from sgp4.api import WGS72, Satrec, SatrecArray

from keepout._types import FloatArray, IndexArray, override
from keepout._validation import require_non_negative
from keepout.ccsds.epochs import format_epoch, utc_fields
from keepout.ccsds.keywords import message_number
from keepout.errors import UnsupportedElements
from keepout.omm.keywords import ATTRIBUTE_KEYWORDS, DEGREE, SGP4_EPHEMERIS_TYPE
from keepout.omm.model import SGP4_EPOCH_DIGITS, Omm
from keepout.propagation._sgp4_radius import brouwer_mean_motion, radius_bounds
from keepout.propagation.base import (
    Ephemeris,
    ErrorArray,
    Propagator,
    RadiusBounds,
    States,
    object_indices,
    pair_sampler,
    time_offsets,
)
from keepout.units import DAY, KM, MINUTE

__all__ = [
    "MAX_SATREC_NUMBER",
    "PairStates",
    "Sgp4Propagator",
    "Sgp4Sampler",
    "Sgp4Shape",
    "pair_states",
    "satrec_of",
    "sgp4_shape",
]

MAX_SATREC_NUMBER: Final = 339_999
"""Largest catalog number SGP4's satellite record holds (Alpha-5 ``Z9999``)."""

_MINUTES_PER_DAY: Final = DAY / MINUTE

_EPOCH_SECONDS: Final = slice(17, 19)
"""The whole seconds of epoch text ``YYYY-MM-DDTHH:MM:SS.ffffff``."""

_SGP4_EPOCH_FORMAT: Final = "%Y-%m-%dT%H:%M:%S.%f"
"""How :func:`sgp4.omm.initialize` reads the epoch text."""

# The constants of sgp4.omm, computed as it does.
_SGP4_EPOCH0: Final = datetime(1949, 12, 31, tzinfo=UTC)
_NDOT_UNITS: Final = 1036800.0 / math.pi
_NDDOT_UNITS: Final = 2985984000.0 / 2.0 / math.pi

_HALF_DAY_MINUTES: Final = 720.0
"""Minutes in half a day: n / 720 * pi is a mean motion in rev/day in
rad/min, as :mod:`sgp4.omm` computes it."""

PairStates: TypeAlias = Callable[[int, int], States]
"""The states of one pair, from its primary's and its secondary's positions
in their propagators: rows 0 and 1, as :func:`pair_states` gives them."""


def satrec_of(omm: Omm) -> Satrec:
    """A new SGP4 satellite record of an element set (WGS-72 constants).

    It is the record :func:`sgp4.omm.initialize` makes from the message the
    elements were read from, bit for bit: every number is converted back to
    the message's value exactly (see :meth:`keepout.omm.Omm.keyword_texts`; this
    holds for numbers of up to 15 significant digits, and CelesTrak writes
    at most 10). Only the epoch is rounded to whole microseconds, the
    resolution :func:`sgp4.omm.initialize` reads; CelesTrak's epochs have
    exactly that resolution.

    Parameters
    ----------
    omm
        The element set.

    Raises
    ------
    UnsupportedElements
        If the catalog number is above :data:`MAX_SATREC_NUMBER`, or the
        epoch falls in a leap second (SGP4 counts days of 86400 s).
    """
    if omm.catalog_number > MAX_SATREC_NUMBER:
        raise UnsupportedElements(
            f"catalog number {omm.catalog_number} is above "
            f"{MAX_SATREC_NUMBER}, the largest an SGP4 record holds"
        )
    epoch = _sgp4_epoch(omm.epoch)
    # What sgp4.omm.initialize does with the text of Omm.keyword_texts(), without
    # the text: each number is the one the text reads back as.
    satrec = Satrec()
    satrec.classification = omm.classification
    satrec.intldesg = omm.international_designator[2:].replace("-", "")
    satrec.ephtype = SGP4_EPHEMERIS_TYPE
    satrec.elnum = omm.element_set_no
    satrec.revnum = omm.rev_at_epoch
    satrec.sgp4init(
        WGS72,
        "i",
        omm.catalog_number,
        epoch,
        _message_number(omm, "bstar"),
        _message_number(omm, "mean_motion_dot") / _NDOT_UNITS,
        _message_number(omm, "mean_motion_ddot") / _NDDOT_UNITS,
        _message_number(omm, "eccentricity"),
        _message_number(omm, "arg_of_pericenter") * DEGREE,
        _message_number(omm, "inclination") * DEGREE,
        _message_number(omm, "mean_anomaly") * DEGREE,
        # rev/day -> rad/min
        _message_number(omm, "mean_motion") / _HALF_DAY_MINUTES * math.pi,
        _message_number(omm, "ra_of_asc_node") * DEGREE,
    )
    return satrec


def _message_number(omm: Omm, attribute: str) -> float:
    """The value of an attribute in the message's units, as written."""
    return message_number(getattr(omm, attribute), ATTRIBUTE_KEYWORDS[attribute].scale)


def _sgp4_epoch(epoch: Time) -> float:
    """Days from 1949-12-31 0h UTC that :func:`sgp4.omm.initialize` reads.

    That is, of the epoch's text rounded to microseconds (see
    :meth:`keepout.omm.Omm.keyword_texts`); read from the text itself only where
    it is not plain ``YYYY-MM-DDThh:mm:ss.ffffff`` from ERFA (see
    :func:`~keepout.ccsds.epochs.utc_fields`).

    Raises
    ------
    UnsupportedElements
        If the epoch falls in a leap second.
    """
    fields = utc_fields(epoch, SGP4_EPOCH_DIGITS)
    if fields is None:
        text = format_epoch(Time(epoch, precision=SGP4_EPOCH_DIGITS))
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


class Sgp4Shape(NamedTuple):
    """The size and shape of an orbit as SGP4 derives them from an element
    set (WGS 72 constants, un-Kozai'd elements); see :func:`sgp4_shape`."""

    perigee_altitude: float
    """Perigee altitude [Earth radii] (``Satrec.altp``)."""
    apogee_altitude: float
    """Apogee altitude [Earth radii] (``Satrec.alta``)."""
    earth_radius_km: float
    """The Earth radius SGP4 uses [km] (``Satrec.radiusearthkm``)."""
    mean_motion: float
    """SGP4's un-Kozai'd (Brouwer) mean motion [rad/min]; ``Satrec.no`` is
    the element set's Kozai mean motion."""


def sgp4_shape(omm: Omm) -> Sgp4Shape:
    """The orbit's size and shape as SGP4 initialises it from ``omm``.

    Parameters
    ----------
    omm
        The element set.

    Raises
    ------
    UnsupportedElements
        If SGP4 cannot be initialised from ``omm`` (:func:`satrec_of`).
    """
    satrec = satrec_of(omm)
    return Sgp4Shape(
        satrec.altp, satrec.alta, satrec.radiusearthkm, brouwer_mean_motion(satrec)
    )


def _utc_dates(epoch: Time, offsets: FloatArray) -> tuple[FloatArray, FloatArray]:
    """UTC two-part Julian dates of ``epoch + offsets`` [s], the times SGP4 takes.

    The offsets are added on TAI, which has no leap seconds, and converted
    by ERFA as astropy converts; this is exact across a leap second. Whole
    days go to the first part, so the fraction keeps ~10 ps resolution.
    """
    tai = epoch.tai
    return _tai_to_utc(tai.jd1, tai.jd2, offsets)


def _tai_to_utc(
    jd1: FloatArray, jd2: FloatArray, offsets: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """:func:`_utc_dates` from the two-part TAI Julian date of the epoch."""
    days, rest = np.divmod(offsets, DAY)
    fraction = jd2 + rest / DAY
    carry = np.round(fraction)
    utc1, utc2 = erfa.taiutc(jd1 + days + carry, fraction - carry)
    return np.asarray(utc1, dtype=np.float64), np.asarray(utc2, dtype=np.float64)


class _Partition(NamedTuple):
    """Element sets split by whether SGP4 can be initialised with them."""

    kept: list[int]
    """Position of each initialised element set."""
    omms: list[Omm]
    """The initialised element sets."""
    satrecs: list[Satrec]
    """Their SGP4 records."""
    rejected: list[tuple[int, UnsupportedElements]]
    """Position of every other element set, with the error :func:`satrec_of`
    raised for it."""


def _partition(omms: Iterable[Omm]) -> _Partition:
    """The element sets of ``omms`` initialised for SGP4 (each once) and those
    that cannot be, in order."""
    partition = _Partition([], [], [], [])
    for index, omm in enumerate(omms):
        try:
            satrec = satrec_of(omm)
        except UnsupportedElements as error:
            partition.rejected.append((index, error))
        else:
            partition.kept.append(index)
            partition.omms.append(omm)
            partition.satrecs.append(satrec)
    return partition


def _si_states(
    errors: ErrorArray, r: FloatArray, v: FloatArray
) -> tuple[ErrorArray, FloatArray, FloatArray]:
    """SGP4's errors and states, km and km/s converted to m and m/s in place."""
    r *= KM
    v *= KM
    return errors, r, v


def _minutes_since_epoch(
    jd: FloatArray | float,
    fraction: FloatArray | float,
    epoch: FloatArray,
    epoch_fraction: FloatArray,
) -> FloatArray:
    """Minutes from each element epoch to a two-part UTC Julian date, as
    ``Satrec.sgp4`` counts them."""
    minutes: FloatArray = (jd - epoch) * _MINUTES_PER_DAY + (
        fraction - epoch_fraction
    ) * _MINUTES_PER_DAY
    return minutes


class Sgp4Propagator:
    """SGP4 states of many objects at many times, in one vectorised call.

    Each object is initialised once, from its OMM, exactly as
    :func:`satrec_of` does (WGS-72 constants, the record
    :func:`sgp4.omm.initialize` makes); :meth:`propagate` then runs the
    compiled SGP4 of the ``sgp4`` package (``SatrecArray``) over every object
    and time, and returns TEME positions [m] and velocities [m/s] with SGP4's
    error codes (non-zero codes come with NaN states). Each state is the one
    ``Satrec.sgp4`` gives for its object and time, independent of the other
    objects and times.

    Times are offsets in SI seconds from a reference epoch; they are added on
    TAI and converted to the UTC Julian dates SGP4 takes, so a leap second
    inside the grid is counted. (SGP4 itself counts its time since the
    element epoch in UTC days, as Vallado's reference code does.)

    A pickled propagator holds its OMMs and is initialised again when it is
    unpickled (about 0.1 ms per object).

    Parameters
    ----------
    omms
        Element sets of the objects, in the order of the results.

    Raises
    ------
    UnsupportedElements
        If an element set cannot initialise SGP4 (a catalog number above
        :data:`MAX_SATREC_NUMBER`, or an epoch in a leap
        second); the message names its position and catalog number.
    """

    __slots__ = ("_array", "_omms", "_satrecs")

    _omms: tuple[Omm, ...]
    _satrecs: tuple[Satrec, ...]
    _array: SatrecArray

    def __init__(self, omms: Iterable[Omm]) -> None:
        elements = tuple(omms)
        partition = _partition(elements)
        if partition.rejected:
            index, error = partition.rejected[0]
            raise UnsupportedElements(
                f"element set {index} (catalog number "
                f"{elements[index].catalog_number}) cannot initialise SGP4: {error}"
            ) from error
        self._set(elements, tuple(partition.satrecs))

    @classmethod
    def from_omms(
        cls, omms: Iterable[Omm]
    ) -> tuple[Self, IndexArray, list[tuple[int, UnsupportedElements]]]:
        """A propagator of the element sets SGP4 can be initialised with.

        Unlike the constructor, an element set SGP4 cannot be initialised
        with is set aside instead of failing the others; each is initialised
        once.

        Parameters
        ----------
        omms
            Element sets of the objects.

        Returns
        -------
        propagator : Sgp4Propagator
            The objects SGP4 was initialised for, in the order given.
        kept : IndexArray
            Position of each of them in ``omms``.
        rejected : list of tuple of int and UnsupportedElements
            Position in ``omms`` of every other element set, with the error
            :func:`satrec_of` raised for it.
        """
        partition = _partition(omms)
        propagator = object.__new__(cls)
        propagator._set(tuple(partition.omms), tuple(partition.satrecs))
        return propagator, np.array(partition.kept, dtype=np.intp), partition.rejected

    def _set(self, omms: tuple[Omm, ...], satrecs: tuple[Satrec, ...]) -> None:
        self._omms = omms
        self._satrecs = satrecs
        self._array = SatrecArray(list(satrecs))

    @property
    def omms(self) -> tuple[Omm, ...]:
        """Element sets of the objects, in order."""
        return self._omms

    @property
    def frame(self) -> Literal["TEME"]:
        """Frame of the states: TEME, the frame of SGP4."""
        return "TEME"

    def __len__(self) -> int:
        return len(self._omms)

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}({len(self)} objects)"

    @override
    def __reduce__(self) -> tuple[type[Self], tuple[tuple[Omm, ...]]]:
        return type(self), (self._omms,)

    def subset(self, indices: npt.ArrayLike) -> Self:
        """A propagator of some of the objects, sharing their SGP4 records.

        Parameters
        ----------
        indices
            Positions of the objects to keep, in the order wanted; a
            one-dimensional array of integers in ``[0, len(self))``
            (repeats allowed).

        Returns
        -------
        Sgp4Propagator
            The chosen objects; no object is initialised again.

        Raises
        ------
        ValueError
            If ``indices`` are not integers in one dimension.
        IndexError
            If an index is outside ``[0, len(self))``.
        """
        chosen = object_indices(indices, len(self))
        subset = object.__new__(type(self))
        subset._set(
            tuple(self._omms[i] for i in chosen),
            tuple(self._satrecs[i] for i in chosen),
        )
        return subset

    def propagate(self, epoch: Time, offsets: npt.ArrayLike) -> Ephemeris:
        """TEME states of every object at ``epoch + offsets``.

        Parameters
        ----------
        epoch
            Reference epoch; one time, in any time scale.
        offsets
            Seconds elapsed since ``epoch``; a number is one offset.

        Returns
        -------
        Ephemeris
            Positions [m] and velocities [m/s] in TEME, shape ``(N, T, 3)``,
            and SGP4's error codes, shape ``(N, T)``.

        Raises
        ------
        ValueError
            If ``epoch`` is not one time or the offsets are not finite
            numbers in at most one dimension.
        """
        checked = time_offsets(epoch, offsets)
        errors, r, v = _si_states(*self._array.sgp4(*_utc_dates(epoch, checked)))
        return Ephemeris(epoch, checked, r, v, errors, self.frame)

    def radius_bounds(self, start: Time, duration: float) -> RadiusBounds:
        """Bounds on each object's SGP4 distance from the Earth's centre.

        Derived from SGP4's own equations, without propagating: the secular
        semi-major axis and eccentricity over the span (SGP4's drag
        polynomials; for deep-space objects SGP4's own secular elements every
        6 h, lunisolar and resonance terms included), widened by bounds on
        SGP4's periodic terms (J2 short period, J3 long period, lunisolar).
        On the public catalog of October 2026 over 7 days no propagated
        radius falls outside; in LEO the bounds are a median of 3.6 km below
        the lowest and 3.3 km above the highest radius.

        Parameters
        ----------
        start
            Start of the span; one time, in any time scale.
        duration
            Length of the span [s], finite and >= 0.

        Returns
        -------
        RadiusBounds
            Lowest and highest radius [m] of each object over the span;
            ``0`` and ``inf`` for an object that cannot be bounded (SGP4
            fails at a time the bound samples, drag would take the orbit to
            nothing, or an eccentricity would reach 1).

        Raises
        ------
        ValueError
            If ``start`` is not one time or ``duration`` is not finite and
            >= 0.
        """
        require_non_negative(duration, "duration")
        jd, fraction = _utc_dates(start, time_offsets(start, [0.0, duration]))
        epoch = np.array([satrec.jdsatepoch for satrec in self._satrecs])
        epoch_fraction = np.array([satrec.jdsatepochF for satrec in self._satrecs])
        first = _minutes_since_epoch(jd[0], fraction[0], epoch, epoch_fraction)
        last = _minutes_since_epoch(jd[1], fraction[1], epoch, epoch_fraction)
        return RadiusBounds(*radius_bounds(self._satrecs, first, last))


class Sgp4Sampler:
    """SGP4 states of chosen objects at many offsets from one epoch, fast.

    The path for a caller that evaluates a few objects at a few times very
    often, as screening refines a pair: the epoch is converted to TAI once,
    and the offsets are taken as given, with neither the checks of
    :meth:`Sgp4Propagator.propagate` nor its :class:`Ephemeris`. Every state
    is the one :meth:`Sgp4Propagator.propagate` gives, bit for bit.

    Parameters
    ----------
    epoch
        Reference epoch; one time, in any time scale.
    propagators
        The objects, numbered in order across the propagators.
    """

    __slots__ = ("_jd1", "_jd2", "_satrecs")

    def __init__(self, epoch: Time, propagators: Iterable[Sgp4Propagator]) -> None:
        tai = epoch.tai
        self._jd1 = tai.jd1
        self._jd2 = tai.jd2
        self._satrecs = tuple(
            satrec
            for propagator in propagators
            for satrec in propagator._satrecs  # noqa: SLF001
        )

    def states(self, indices: Sequence[int]) -> States:
        """The states of some of the objects, in the order wanted.

        Parameters
        ----------
        indices
            Numbers of the objects (across the propagators).

        Returns
        -------
        States
            Their states at offsets [s] from the epoch, which must be finite
            float64 numbers in one dimension (not checked).
        """
        array = SatrecArray([self._satrecs[index] for index in indices])
        jd1, jd2 = self._jd1, self._jd2

        def states(offsets: FloatArray) -> tuple[ErrorArray, FloatArray, FloatArray]:
            return _si_states(*array.sgp4(*_tai_to_utc(jd1, jd2, offsets)))

        return states


def pair_states(start: Time, primaries: Propagator, catalog: Propagator) -> PairStates:
    """The states of pairs of a primary and a catalog object, at offsets [s]
    from ``start``, as the propagators give them.

    The fast path of SGP4 (:class:`Sgp4Sampler`) when both propagators are
    :class:`Sgp4Propagator`; else each pair's propagators
    (:func:`~keepout.propagation.base.pair_sampler`).

    Parameters
    ----------
    start
        Reference epoch of the offsets; one time.
    primaries, catalog
        The primaries and the catalog objects.

    Returns
    -------
    PairStates
        For a primary's and a catalog object's positions in ``primaries`` and
        ``catalog``, the sampler of their states (rows 0 and 1). Its offsets
        must be finite float64 numbers in one dimension (not checked).
    """
    if isinstance(primaries, Sgp4Propagator) and isinstance(catalog, Sgp4Propagator):
        sampler = Sgp4Sampler(start, (primaries, catalog))
        catalog_first = len(primaries)

        def sgp4_states(primary: int, secondary: int) -> States:
            return sampler.states((primary, catalog_first + secondary))

        return sgp4_states

    def propagated_states(primary: int, secondary: int) -> States:
        return pair_sampler(
            primaries.subset(np.array([primary])),
            catalog.subset(np.array([secondary])),
            start,
        )

    return propagated_states
