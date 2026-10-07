"""Bounds on the radius SGP4 gives an object over a time span.

SGP4 (Hoots and Roehrich, Spacetrack Report #3; Vallado et al., AIAA
2006-6753, whose code the ``sgp4`` package compiles) computes the radius of
an object at ``t`` minutes from its element epoch, in Earth radii, as::

    mrt = rl (1 - 1.5 temp2 betal con41) + 0.5 temp1 x1mth2 cos2u
    rl = am (1 - ecose),   |ecose| <= el,   pl = am (1 - el^2),
    temp1 = j2 / (2 pl),   temp2 = temp1 / pl,   0 < betal <= 1,

where ``am`` is the secular semi-major axis, ``el`` the eccentricity with
SGP4's long-period J3 term, ``con41 = 3 cos^2 i - 1`` and ``x1mth2 = sin^2
i``. With ``am`` in ``[am_lo, am_hi]`` and ``el <= el_max`` over the span,
the radius therefore lies in::

    [am_lo (1 - el_max) f_lo - 0.5 temp1 x1mth2,
     am_hi (1 + el_max) f_hi + 0.5 temp1 x1mth2]

with ``temp1`` and ``temp2`` at ``pl_min = am_lo (1 - el_max^2)`` and
``f_lo``, ``f_hi`` the extremes of ``1 - 1.5 temp2 betal con41`` (the J2
short-period terms). The J3 term adds ``aycof / (am (1 - ep^2))`` to the
eccentricity vector, with ``|aycof| = 0.5 |J3/J2| sin i``, so ``el <=
ep_max + |aycof| / (am_lo (1 - ep_max^2))``, where ``ep`` is the mean
eccentricity ``em`` plus, in deep space, its lunisolar periodic terms.

Near-Earth objects (period below 225 min) keep their inclination, and their
secular terms are SGP4's drag polynomials, with the coefficients SGP4's
initialisation (``sgp4init``) computes, recomputed here from the record:
``am = a0 (1 - P(cc1 t))^2`` with ``P(x) = x + c2 x^2 + c3 x^3 + c4 x^4``
(``c2 = c3 = c4 = 0`` under SGP4's simplified drag, perigee below 220 km),
bounded over the span term by term, and ``em = e0 - B* cc4 t - B* cc5 (sin
M - sin M0)``, linear in ``t`` up to a once-per-revolution term of at most
``2 |B* cc5|`` (none under simplified drag).

Deep-space objects (period of 225 min or more) take their secular ``am``
and ``em`` (drag, lunisolar secular rates, the 12 h and 24 h resonances)
from SGP4 itself, at both ends of the span, every
:data:`DEEP_SPACE_SAMPLING` minutes between and at the element epoch.
``em`` is linear in ``t``, so its samples hold its largest value; ``am``
moves between samples with SGP4's integration of the resonances (by at most
2.9e-8 of itself on the public catalog of October 2026, against a 1 min
sampling), so its sampled range is widened by :data:`DEEP_SPACE_MARGIN`.
The lunisolar periodic terms add at most ``7.5 e0 (C1SS + C1L) / n0`` to the
eccentricity: in SGP4's ``dscom`` each body adds ``2 s1 (s6 f2 + s7 f3)``,
where ``|(f2, f3)| = 1/4``, ``s6^2 + s7^2 <= 1`` (products of direction
cosines) and ``|s1| <= 15 e0 C / n0``. They also move the inclination, so
the full ranges ``con41`` in ``[-1, 2]``, ``x1mth2 <= 1`` and ``|aycof| <=
0.5 |J3/J2|`` are used.

An object for which SGP4 fails at a sample, or whose bounds are not finite
or would need an eccentricity of 1 or more, gets ``[0, inf)``. Both bounds
are widened by :data:`ROUNDING` (relative).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final, NamedTuple

import numpy as np
from sgp4.api import Satrec

from keepout._types import BoolArray, FloatArray
from keepout._units import KM

DEEP_SPACE_SAMPLING: Final = 360.0
"""Longest time [min] between the samples of a deep-space object's secular
semi-major axis and eccentricity."""

DEEP_SPACE_MARGIN: Final = 1e-6
"""Relative widening of a deep-space object's sampled range of ``am``."""

ROUNDING: Final = 1e-9
"""Relative widening of both bounds, for rounding."""

SIMPLE_DRAG_PERIGEE: Final = 220.0
"""Perigee altitude [km] below which SGP4 uses its simplified drag."""

C1SS: Final = 2.9864797e-6
"""Solar perturbation coefficient of SGP4's ``dscom``."""

C1L: Final = 4.7968065e-7
"""Lunar perturbation coefficient of SGP4's ``dscom``."""

MIN_ECCENTRICITY: Final = 1e-6
"""SGP4 raises a smaller mean eccentricity to this value."""

# The atmosphere model of ``sgp4init`` (Vallado et al., AIAA 2006-6753; Hoots
# and Roehrich, Spacetrack Report #3): its density parameter ``s`` and
# ``(q0 - s)^4``, altitudes [km] above the Earth's radius.
S_PERIGEE_LIMIT: Final = 156.0
"""Perigee altitude [km] below which ``s`` follows the perigee."""

S_FLOOR_PERIGEE: Final = 98.0
"""Perigee altitude [km] below which ``s`` is :data:`S_FLOOR`."""

S_FLOOR: Final = 20.0
"""Lowest ``s`` altitude [km]."""

S0: Final = 78.0
"""Standard ``s`` altitude [km] (``s = perigee - S0`` below the limit)."""

Q0: Final = 120.0
"""Altitude [km] of the density parameter ``q0``."""


class DragTerms(NamedTuple):
    """SGP4's drag coefficients of near-Earth objects (``sgp4init``)."""

    cc1: FloatArray
    """Linear rate [1/min] of ``1 - sqrt(am / a0)``."""
    cc4: FloatArray
    """Linear rate [1/min] of the eccentricity, per unit of B*."""
    cc5: FloatArray
    """Amplitude of the once-per-revolution eccentricity term, per unit of
    B*."""
    c2: FloatArray
    """``d2 / cc1^2``."""
    c3: FloatArray
    """``d3 / cc1^3``."""
    c4: FloatArray
    """``d4 / cc1^4``."""
    simple: BoolArray
    """Whether SGP4 uses its simplified drag (``isimp``)."""


def _values(satrecs: Sequence[Satrec], name: str) -> FloatArray:
    return np.array([getattr(satrec, name) for satrec in satrecs], dtype=np.float64)


def brouwer_mean_motion(satrec: Satrec) -> float:
    """SGP4's un-Kozai'd mean motion [rad/min] (``Satrec.no`` is Kozai's)."""
    return float(satrec.xke / satrec.a**1.5)


def drag_terms(satrecs: Sequence[Satrec]) -> DragTerms:
    """SGP4's drag coefficients, as ``sgp4init`` computes them.

    Parameters
    ----------
    satrecs
        Records of near-Earth objects (deep-space records get numbers SGP4
        does not use).
    """
    ao = _values(satrecs, "a")
    ecco = _values(satrecs, "ecco")
    no = np.array([brouwer_mean_motion(satrec) for satrec in satrecs])
    j2 = _values(satrecs, "j2")
    radius = _values(satrecs, "radiusearthkm")
    cosio2 = np.cos(_values(satrecs, "inclo")) ** 2
    con41 = 3.0 * cosio2 - 1.0
    x1mth2 = 1.0 - cosio2
    omeosq = 1.0 - ecco**2
    perigee = ao * (1.0 - ecco)
    simple = perigee < SIMPLE_DRAG_PERIGEE / radius + 1.0
    altitude = (perigee - 1.0) * radius
    low = altitude < S_PERIGEE_LIMIT
    s = np.where(altitude < S_FLOOR_PERIGEE, S_FLOOR, altitude - S0)
    sfour = np.where(low, s / radius + 1.0, S0 / radius + 1.0)
    qzms24 = np.where(low, (Q0 - s) / radius, (Q0 - S0) / radius) ** 4
    tsi = 1.0 / (ao - sfour)
    eta = ao * ecco * tsi
    etasq = eta * eta
    eeta = ecco * eta
    psisq = np.abs(1.0 - etasq)
    coef = qzms24 * tsi**4
    coef1 = coef / psisq**3.5
    cc2 = (
        coef1
        * no
        * (
            ao * (1.0 + 1.5 * etasq + eeta * (4.0 + etasq))
            + 0.375 * j2 * tsi / psisq * con41 * (8.0 + 3.0 * etasq * (8.0 + etasq))
        )
    )
    cc4 = (
        2.0
        * no
        * coef1
        * ao
        * omeosq
        * (
            eta * (2.0 + 0.5 * etasq)
            + ecco * (0.5 + 2.0 * etasq)
            - j2
            * tsi
            / (ao * psisq)
            * (
                -3.0 * con41 * (1.0 - 2.0 * eeta + etasq * (1.5 - 0.5 * eeta))
                + 0.75
                * x1mth2
                * (2.0 * etasq - eeta * (1.0 + etasq))
                * np.cos(2.0 * _values(satrecs, "argpo"))
            )
        )
    )
    cc5 = 2.0 * coef1 * ao * omeosq * (1.0 + 2.75 * (etasq + eeta) + eeta * etasq)
    c2 = np.where(simple, 0.0, 4.0 * ao * tsi)
    c3 = (17.0 * ao + sfour) * c2 * tsi / 3.0
    c4 = 0.5 * c2 * tsi / 3.0 * ao * tsi * (221.0 * ao + 31.0 * sfour)
    return DragTerms(_values(satrecs, "bstar") * cc2, cc4, cc5, c2, c3, c4, simple)


def _power_range(
    lowest: FloatArray, highest: FloatArray, power: int
) -> tuple[FloatArray, FloatArray]:
    """Range of ``x**power`` for ``x`` from ``lowest`` to ``highest``."""
    at_lowest = lowest**power
    at_highest = highest**power
    smallest = np.minimum(at_lowest, at_highest)
    if power % 2 == 0:
        smallest = np.where((lowest <= 0.0) & (highest >= 0.0), 0.0, smallest)
    return smallest, np.maximum(at_lowest, at_highest)


class _Secular(NamedTuple):
    """What the radius bound needs of each object, Earth radii."""

    am_lo: FloatArray
    am_hi: FloatArray
    ep_max: FloatArray
    aycof: FloatArray
    con41_lo: FloatArray
    con41_hi: FloatArray
    x1mth2: FloatArray


def _near_earth(
    satrecs: Sequence[Satrec], first: FloatArray, last: FloatArray
) -> _Secular:
    """Secular range of near-Earth objects, from SGP4's drag polynomials."""
    terms = drag_terms(satrecs)
    ao = _values(satrecs, "a")
    ecco = _values(satrecs, "ecco")
    bstar = _values(satrecs, "bstar")
    inclo = _values(satrecs, "inclo")
    x_lo = np.minimum(terms.cc1 * first, terms.cc1 * last)
    x_hi = np.maximum(terms.cc1 * first, terms.cc1 * last)
    p_lo, p_hi = x_lo.copy(), x_hi.copy()
    for c, power in ((terms.c2, 2), (terms.c3, 3), (terms.c4, 4)):
        lowest, highest = _power_range(x_lo, x_hi, power)
        p_lo += np.minimum(c * lowest, c * highest)
        p_hi += np.maximum(c * lowest, c * highest)
    tempa_lo = 1.0 - p_hi
    tempa_hi = 1.0 - p_lo
    rate = bstar * terms.cc4
    em_max = ecco - np.minimum(rate * first, rate * last)
    em_max += np.where(terms.simple, 0.0, 2.0 * np.abs(bstar * terms.cc5))
    cosio2 = np.cos(inclo) ** 2
    con41 = 3.0 * cosio2 - 1.0
    return _Secular(
        am_lo=np.where(tempa_lo > 0.0, ao * tempa_lo**2, math.nan),
        am_hi=ao * tempa_hi**2,
        ep_max=np.maximum(em_max, MIN_ECCENTRICITY),
        aycof=np.abs(0.5 * _values(satrecs, "j3oj2") * np.sin(inclo)),
        con41_lo=con41,
        con41_hi=con41,
        x1mth2=1.0 - cosio2,
    )


def _deep_space(
    satrec: Satrec, first: float, last: float
) -> tuple[float, float, float]:
    """``am_lo``, ``am_hi`` and ``ep_max`` of a deep-space object, from
    SGP4's secular elements sampled over the span (NaN if SGP4 fails)."""
    count = math.ceil((last - first) / DEEP_SPACE_SAMPLING) + 1
    times: list[float] = np.linspace(first, last, count).tolist()
    if first < 0.0 < last:
        times.append(0.0)
    semi_major_axes = []
    eccentricities = []
    for minutes in times:
        error, _, _ = satrec.sgp4_tsince(minutes)
        if error:
            return math.nan, math.nan, math.nan
        semi_major_axes.append(satrec.am)
        eccentricities.append(satrec.em)
    lunisolar = 7.5 * satrec.ecco * (C1SS + C1L) / brouwer_mean_motion(satrec)
    return (
        min(semi_major_axes) * (1.0 - DEEP_SPACE_MARGIN),
        max(semi_major_axes) * (1.0 + DEEP_SPACE_MARGIN),
        max(eccentricities) + lunisolar,
    )


def radius_bounds(
    satrecs: Sequence[Satrec], first: FloatArray, last: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Bounds on the radius [m] SGP4 gives each object over a time span.

    Parameters
    ----------
    satrecs
        SGP4 records of the objects. Deep-space records are propagated, which
        changes their last results (``am``, ``em``, ...), not their elements.
    first, last
        Start and end of the span [min from each object's epoch], shape
        ``(N,)``, ``first <= last``.

    Returns
    -------
    tuple[FloatArray, FloatArray]
        Lowest and highest radius [m] of each object, shape ``(N,)``; ``0``
        and ``inf`` for an object that cannot be bounded.
    """
    # Overflows and divisions by zero give non-finite bounds: no bounds.
    with np.errstate(all="ignore"):
        return _bounds(satrecs, first, last)


def _bounds(
    satrecs: Sequence[Satrec], first: FloatArray, last: FloatArray
) -> tuple[FloatArray, FloatArray]:
    deep = np.array([satrec.method == "d" for satrec in satrecs], dtype=bool)
    secular = _Secular(*(np.full(len(satrecs), math.nan) for _ in _Secular._fields))
    near = np.flatnonzero(~deep)
    if near.size:
        found = _near_earth(
            [satrecs[i] for i in near.tolist()], first[near], last[near]
        )
        for array, values in zip(secular, found, strict=True):
            array[near] = values
    for i in np.flatnonzero(deep).tolist():
        satrec = satrecs[i]
        sampled = _deep_space(satrec, first[i], last[i])
        secular.am_lo[i], secular.am_hi[i], secular.ep_max[i] = sampled
        secular.aycof[i] = 0.5 * abs(satrec.j3oj2)
        secular.con41_lo[i], secular.con41_hi[i], secular.x1mth2[i] = -1.0, 2.0, 1.0
    am_lo, am_hi, ep_max, aycof, con41_lo, con41_hi, x1mth2 = secular
    el_max = ep_max + aycof / (am_lo * (1.0 - ep_max**2))
    pl_min = am_lo * (1.0 - el_max**2)
    temp1 = 0.5 * _values(satrecs, "j2") / pl_min
    temp2 = temp1 / pl_min
    f_lo = 1.0 - 1.5 * temp2 * np.maximum(con41_hi, 0.0)
    f_hi = 1.0 - 1.5 * temp2 * np.minimum(con41_lo, 0.0)
    lowest = am_lo * (1.0 - el_max) * f_lo - 0.5 * temp1 * x1mth2
    highest = am_hi * (1.0 + el_max) * f_hi + 0.5 * temp1 * x1mth2
    known = (
        (ep_max < 1.0)
        & (el_max < 1.0)
        & (am_lo > 0.0)
        & np.isfinite(lowest)
        & np.isfinite(highest)
    )
    # A negative factor (an orbit inside the Earth) bounds nothing below.
    lowest = np.where(f_lo > 0.0, np.maximum(lowest, 0.0), 0.0)
    radius = _values(satrecs, "radiusearthkm") * KM
    return (
        np.where(known, lowest * radius * (1.0 - ROUNDING), 0.0),
        np.where(known, highest * radius * (1.0 + ROUNDING), math.inf),
    )
