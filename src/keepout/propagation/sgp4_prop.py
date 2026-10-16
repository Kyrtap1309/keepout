"""SGP4 for many objects at once, behind the
:class:`~keepout.propagation.base.Propagator` interface."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final, Literal, Self

import erfa
import numpy as np
import numpy.typing as npt
from astropy.time import Time
from sgp4.api import Satrec, SatrecArray

from keepout._types import FloatArray, IndexArray, require_non_negative
from keepout._units import DAY, KM, MINUTE
from keepout.errors import UnsupportedElements
from keepout.omm.model import Omm
from keepout.propagation._sgp4_radius import radius_bounds
from keepout.propagation.base import Ephemeris, RadiusBounds, time_offsets

__all__ = ["Sgp4Propagator"]

_MINUTES_PER_DAY: Final = DAY / MINUTE


def _utc_dates(epoch: Time, seconds: FloatArray) -> tuple[FloatArray, FloatArray]:
    """UTC two-part Julian dates of ``epoch + seconds``, the times SGP4 takes.

    The offsets are added on TAI, which has no leap seconds, and converted
    by ERFA as astropy converts; this is exact across a leap second. Whole
    days go to the first part, so the fraction keeps ~10 ps resolution.
    """
    tai = epoch.tai
    days, rest = np.divmod(seconds, DAY)
    fraction = tai.jd2 + rest / DAY
    carry = np.round(fraction)
    utc1, utc2 = erfa.taiutc(tai.jd1 + days + carry, fraction - carry)
    return np.asarray(utc1, dtype=np.float64), np.asarray(utc2, dtype=np.float64)


def _indices(indices: npt.ArrayLike, size: int) -> IndexArray:
    """``indices`` checked as positions in a sequence of ``size`` items.

    Raises
    ------
    ValueError
        If ``indices`` is not one-dimensional, or not integers when not
        empty.
    IndexError
        If an index is outside ``[0, size)``.
    """
    array = np.asarray(indices)
    if array.ndim != 1:
        raise ValueError(f"indices must be one-dimensional, got shape {array.shape}")
    if array.size == 0:
        return np.empty(0, dtype=np.intp)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"indices must be integers, got {array.dtype}")
    if array.min() < 0 or array.max() >= size:
        raise IndexError(
            f"indices must be in [0, {size}), got {array.min()} .. {array.max()}"
        )
    return array.astype(np.intp)


class Sgp4Propagator:
    """SGP4 states of many objects at many times, in one vectorised call.

    Each object is initialised once, from its OMM, exactly as
    :meth:`keepout.omm.Omm.satrec` does (WGS-72 constants, the record
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
        ``keepout.omm.model.MAX_SATREC_NUMBER``, or an epoch in a leap
        second); the message names its position and catalog number.
    """

    __slots__ = ("_array", "_omms", "_satrecs")

    _omms: tuple[Omm, ...]
    _satrecs: tuple[Satrec, ...]
    _array: SatrecArray

    def __init__(self, omms: Iterable[Omm]) -> None:
        elements = tuple(omms)
        satrecs = []
        for index, omm in enumerate(elements):
            try:
                satrecs.append(omm.satrec())
            except UnsupportedElements as error:
                raise UnsupportedElements(
                    f"element set {index} (catalog number {omm.catalog_number}) "
                    f"cannot initialise SGP4: {error}"
                ) from error
        self._set(elements, tuple(satrecs))

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
            :meth:`keepout.omm.Omm.satrec` raised for it.
        """
        kept: list[int] = []
        elements: list[Omm] = []
        satrecs: list[Satrec] = []
        rejected: list[tuple[int, UnsupportedElements]] = []
        for index, omm in enumerate(omms):
            try:
                satrec = omm.satrec()
            except UnsupportedElements as error:
                rejected.append((index, error))
                continue
            kept.append(index)
            elements.append(omm)
            satrecs.append(satrec)
        propagator = object.__new__(cls)
        propagator._set(tuple(elements), tuple(satrecs))  # noqa: SLF001
        return propagator, np.array(kept, dtype=np.intp), rejected

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

    def __repr__(self) -> str:
        return f"{type(self).__name__}({len(self)} objects)"

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
        chosen = _indices(indices, len(self))
        subset = object.__new__(type(self))
        subset._set(  # noqa: SLF001
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
        seconds = time_offsets(epoch, offsets)
        errors, r, v = self._array.sgp4(*_utc_dates(epoch, seconds))
        r *= KM
        v *= KM
        return Ephemeris(epoch, seconds, r, v, errors, self.frame)

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
        # Minutes since each element epoch, as Satrec.sgp4 counts them.
        first = (jd[0] - epoch) * _MINUTES_PER_DAY + (
            fraction[0] - epoch_fraction
        ) * _MINUTES_PER_DAY
        last = (jd[1] - epoch) * _MINUTES_PER_DAY + (
            fraction[1] - epoch_fraction
        ) * _MINUTES_PER_DAY
        return RadiusBounds(*radius_bounds(self._satrecs, first, last))
