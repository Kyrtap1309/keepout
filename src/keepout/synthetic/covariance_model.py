"""A synthetic covariance for GP (TLE/OMM) element sets, from their age.

GP element sets, propagated with SGP4, come without a covariance, and NASA's
CA handbook advises against risk decisions on them alone (Rev. 2, Vol. 2,
App. E). To compute a Pc for a synthetic operator anyway, keepout assumes a
covariance: a diagonal **position** covariance in the object's RTN frame
whose radial and normal sigmas are constant and whose transverse
(along-track) sigma grows linearly with the age of the element set::

    sigma_R = f sigma_R0
    sigma_T = f (sigma_T0 + k |t - epoch|)
    sigma_N = f sigma_N0

``sigma_R0``, ``sigma_T0`` and ``sigma_N0`` are the epoch sigmas of the
object's orbit regime (:class:`OrbitRegime`, Flohrer et al. 2009), ``k``
the along-track growth (Levit and Marshall 2011) and ``f`` a safety factor
(default 2: Flohrer et al. found the true errors about twice their
estimates). The velocity block is zero: the velocity uncertainty is
unknown, and a 2-D Pc uses position only. The model is meant for the 2-D
methods: :func:`~keepout.pc.compute_pc` with ``"auto"`` keeps the 2-D Pc
for it, with a note, also where a 2-D assumption is questionable, since the
3-D methods would take the velocities as exact (and on slow encounters they
depend strongly on the velocity uncertainty).

This is an **assumption**, not a calibrated product: GP errors have biases,
differ forward and backward in time and between objects in similar orbits
(Kelso 2007; Geul et al. 2017), none of which a zero-mean covariance
holds. The age is taken as ``|t - epoch|`` so element sets newer than ``t``
are covered too.

References
----------
T. Flohrer, H. Krag, H. Klinkrad, B. Bastida Virgili, C. Frueh, "Improving
ESA's collision risk estimates by an assessment of the TLE orbit errors of
the US SSN catalogue", Proc. 5th European Conference on Space Debris, ESA
SP-672, 2009, Table 1 and section 3.2.

C. Levit, W. Marshall, "Improved orbit predictions using two-line
elements", Advances in Space Research 47, 2011 (arXiv:1002.2277), pp. 5-6
and Table 2.

NASA Spacecraft Conjunction Assessment and Collision Avoidance Best
Practices Handbook, NASA/SP-20205011318 Rev. 2, Vol. 2, 2026, Appendix E.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, Self

import numpy as np

from keepout._pickling import reduce_with_mappings
from keepout._types import override
from keepout._validation import (
    non_negative_float,
    positive_float,
    require_single_time,
)
from keepout.covariance import Covariance
from keepout.propagation import sgp4_shape
from keepout.units import DAY, KM, julian_seconds_between

if TYPE_CHECKING:
    from collections.abc import Mapping

    from astropy.time import Time

    from keepout.omm import Omm

__all__ = [
    "FLOHRER_SIGMAS",
    "GEO_LOWER_ALTITUDE",
    "GEO_UPPER_ALTITUDE",
    "LEO_UPPER_ALTITUDE",
    "LEVIT_MARSHALL_RATE",
    "PUBLISHED_PARAMETERS",
    "SAFETY_FACTOR",
    "OrbitRegime",
    "RegimeParameters",
    "RtnSigmas",
    "SyntheticCovarianceModel",
    "orbit_regime",
]


class OrbitRegime(enum.StrEnum):
    """Orbit regimes of Flohrer et al. (2009), by perigee and apogee altitude.

    The bounds are the paper's (p. 3), with its strict inequalities; see
    :func:`orbit_regime`.
    """

    LEO = "LEO"
    """Apogee altitude below 2000 km."""
    MEO = "MEO"
    """Perigee altitude above 2000 km and apogee altitude below 33 786 km."""
    GTO = "GTO"
    """Perigee altitude below 2000 km and apogee altitude above 33 786 km
    (geostationary transfer and Molniya-like orbits)."""
    HEO = "HEO"
    """None of the others: e.g. a perigee above 2000 km with an apogee
    above 37 786 km."""
    GEO = "GEO"
    """Perigee altitude above 33 786 km and apogee altitude below 37 786 km."""


LEO_UPPER_ALTITUDE: Final = 2000e3
"""Altitude [m] of the LEO boundary: the apogee altitude below which an orbit
is LEO, and the perigee altitude that separates MEO (above) from GTO
(below)."""

GEO_LOWER_ALTITUDE: Final = 33_786e3
"""Altitude [m] of the lower GEO boundary: the perigee altitude above which an
orbit can be GEO, and the apogee altitude below which an orbit can be MEO and
above which it can be GTO."""

GEO_UPPER_ALTITUDE: Final = 37_786e3
"""Altitude [m] of the upper GEO boundary: the apogee altitude below which an
orbit can be GEO."""


def orbit_regime(omm: Omm) -> OrbitRegime:
    """The regime of Flohrer et al. (2009) an element set belongs to.

    Perigee and apogee altitudes are those SGP4 itself derives from the
    elements (its un-Kozai'd semi-major axis and the WGS 72 Earth radius:
    ``Satrec.altp``, ``Satrec.alta``), as in
    :func:`keepout.screening.sfs_regime`. In order: LEO if the apogee is
    below 2000 km; GEO if the perigee is above 33 786 km and the apogee
    below 37 786 km; MEO if the perigee is above 2000 km and the apogee
    below 33 786 km; GTO if the perigee is below 2000 km and the apogee
    above 33 786 km; HEO otherwise (also an altitude exactly on a bound).

    Parameters
    ----------
    omm
        Element set of the object.

    Returns
    -------
    OrbitRegime
        The regime of the element set.

    Raises
    ------
    UnsupportedElements
        If SGP4 cannot be initialised from ``omm``
        (:func:`keepout.propagation.satrec_of`).
    """
    shape = sgp4_shape(omm)
    radius = shape.earth_radius_km * KM
    perigee, apogee = shape.perigee_altitude * radius, shape.apogee_altitude * radius
    if apogee < LEO_UPPER_ALTITUDE:
        return OrbitRegime.LEO
    if perigee > GEO_LOWER_ALTITUDE and apogee < GEO_UPPER_ALTITUDE:
        return OrbitRegime.GEO
    if perigee > LEO_UPPER_ALTITUDE and apogee < GEO_LOWER_ALTITUDE:
        return OrbitRegime.MEO
    if perigee < LEO_UPPER_ALTITUDE and apogee > GEO_LOWER_ALTITUDE:
        return OrbitRegime.GTO
    return OrbitRegime.HEO


@dataclass(frozen=True, slots=True)
class RegimeParameters:
    """The covariance model's parameters for one orbit regime, before the
    safety factor.

    Parameters
    ----------
    radial
        Radial sigma ``sigma_R0`` [m].
    transverse
        Transverse (along-track) sigma at the element epoch ``sigma_T0`` [m].
    normal
        Normal (cross-track) sigma ``sigma_N0`` [m].
    transverse_rate
        Growth ``k`` of the transverse sigma with the element set's age
        [m/s]; 0 keeps it constant.

    Raises
    ------
    ValueError
        If a sigma is not finite and positive, or the rate is not finite
        and non-negative.
    """

    radial: float
    transverse: float
    normal: float
    transverse_rate: float

    def __post_init__(self) -> None:
        for name in ("radial", "transverse", "normal"):
            object.__setattr__(self, name, positive_float(getattr(self, name), name))
        rate = non_negative_float(self.transverse_rate, "transverse_rate")
        object.__setattr__(self, "transverse_rate", rate)


class RtnSigmas(NamedTuple):
    """Position standard deviations along an object's RTN axes [m]."""

    radial: float
    """Along R, the radial direction."""
    transverse: float
    """Along T, the transverse (along-track) direction."""
    normal: float
    """Along N, the orbit normal (cross-track)."""


FLOHRER_SIGMAS: Final[Mapping[OrbitRegime, RtnSigmas]] = MappingProxyType(
    {
        OrbitRegime.LEO: RtnSigmas(102.0, 471.0, 126.0),
        OrbitRegime.MEO: RtnSigmas(73.0, 131.0, 54.0),
        OrbitRegime.GTO: RtnSigmas(1960.0, 3897.0, 1808.0),
        OrbitRegime.HEO: RtnSigmas(824.0, 1367.0, 1059.0),
        OrbitRegime.GEO: RtnSigmas(359.0, 432.0, 86.0),
    }
)
"""Radial, along-track and cross-track 1-sigma position errors [m] of GP
element sets near their epoch, per regime: Table 1 of Flohrer et al.
(2009), the US SSN catalog of 2008-01-01 (11 286 objects), each against a
numerical orbit fitted to its own SGP4 states over 24 h centred on the
epoch. In LEO the along-track sigma reaches about 600 m at high
inclination; in the paper's own validation on radar-tracked objects the
true errors were about twice these values (section 3.2)."""

LEVIT_MARSHALL_RATE: Final = 1500.0 / DAY
"""Growth of the along-track error of GP element sets with age [m/s]: the
"typical growth of 1,500 m/day" Levit and Marshall (2011, pp. 5-6) measure
with SGP4 against ILRS precision orbits of Stella, Starlette, Ajisai (LEO)
and Etalon-2 (MEO). Their Table 2 gives 100-3000 m/day for TLEs with SGP4;
NASA's CA handbook (Rev. 2, Vol. 2, App. E, p. 8, after Hejduk et al.)
reports errors of 600 m to 3 km in the first 72 h in LEO. keepout applies
the rate to every regime: no growth was measured for GTO, HEO or GEO."""

SAFETY_FACTOR: Final = 2.0
"""Factor on every sigma and rate: in Flohrer et al.'s validation
(2009, section 3.2) the true errors were about twice their estimates, and
the GEO estimates "might be too optimistic"."""

PUBLISHED_PARAMETERS: Final[Mapping[OrbitRegime, RegimeParameters]] = MappingProxyType(
    {
        regime: RegimeParameters(*sigmas, LEVIT_MARSHALL_RATE)
        for regime, sigmas in FLOHRER_SIGMAS.items()
    }
)
"""The default parameters of each regime: the sigmas of
:data:`FLOHRER_SIGMAS` and the rate :data:`LEVIT_MARSHALL_RATE`, before
the safety factor."""


@dataclass(frozen=True, slots=True)
class SyntheticCovarianceModel:
    """The age-dependent RTN position covariance of a GP element set.

    See the module's description for the model and its sources. Every
    parameter can be overridden, e.g. ``dataclasses.replace(model,
    regimes={**model.regimes, OrbitRegime.LEO: RegimeParameters(100.0,
    500.0, 100.0, 0.02)})`` or ``SyntheticCovarianceModel(safety_factor=1.0)``.

    Parameters
    ----------
    regimes
        Parameters of each :class:`OrbitRegime`, before the safety factor;
        default :data:`PUBLISHED_PARAMETERS`. Stored as a read-only copy.
    safety_factor
        Factor ``f`` on every sigma (and so on the rate) [-]; default
        :data:`SAFETY_FACTOR`.

    Raises
    ------
    ValueError
        If ``regimes`` does not give exactly every regime, or
        ``safety_factor`` is not finite and positive.
    """

    # A factory, as 3.11 dataclasses reject an unhashable mapping proxy as default.
    regimes: Mapping[OrbitRegime, RegimeParameters] = field(
        default_factory=lambda: PUBLISHED_PARAMETERS
    )
    safety_factor: float = SAFETY_FACTOR

    def __post_init__(self) -> None:
        regimes = dict(self.regimes)
        if regimes.keys() != set(OrbitRegime):
            given = ", ".join(map(str, regimes))
            raise ValueError(
                f"regimes must give every one of {', '.join(OrbitRegime)}, got {given}"
            )
        # In regime order, so that equal models hash alike.
        ordered = {regime: regimes[regime] for regime in OrbitRegime}
        object.__setattr__(self, "regimes", MappingProxyType(ordered))
        factor = positive_float(self.safety_factor, "safety_factor")
        object.__setattr__(self, "safety_factor", factor)

    @override
    def __hash__(self) -> int:
        # The regimes' mapping proxy does not hash; its items do.
        return hash((tuple(self.regimes.items()), self.safety_factor))

    @override
    def __reduce__(self) -> tuple[type[Self], tuple[object, ...]]:
        return reduce_with_mappings(self)

    def parameters(self, regime: OrbitRegime) -> RegimeParameters:
        """The parameters of ``regime``, before the safety factor.

        Parameters
        ----------
        regime
            An orbit regime.

        Returns
        -------
        RegimeParameters
            The regime's parameters.
        """
        return self.regimes[regime]

    def sigmas(self, omm: Omm, time: Time) -> RtnSigmas:
        """The model's position sigmas of an element set at a time.

        Parameters
        ----------
        omm
            Element set of the object.
        time
            When the covariance is wanted (e.g. a TCA); one time, before or
            after the element epoch.

        Returns
        -------
        RtnSigmas
            The radial, transverse and normal sigmas [m].

        Raises
        ------
        UnsupportedElements
            If SGP4 cannot be initialised from ``omm`` (:func:`orbit_regime`).
        ValueError
            If ``time`` is not a single time.
        """
        return self._evaluated(omm, time).sigmas

    def covariance(self, omm: Omm, time: Time) -> Covariance:
        """The model's covariance of an element set at a time.

        Parameters as for :meth:`sigmas`.

        Returns
        -------
        Covariance
            ``diag(sigma_R^2, sigma_T^2, sigma_N^2)`` [m^2] in the object's
            RTN frame at ``time``, with a zero velocity block
            (:meth:`~keepout.covariance.Covariance.from_position`).

        Raises
        ------
        UnsupportedElements
            If SGP4 cannot be initialised from ``omm`` (:func:`orbit_regime`).
        ValueError
            If ``time`` is not a single time.
        """
        return _covariance(self._evaluated(omm, time).sigmas, time)

    def comments(self, omm: Omm, time: Time) -> tuple[str, ...]:
        """Lines describing the model's covariance of an element set at a time.

        For a CDM's ``COMMENT`` lines: the model, the regime's parameters,
        the element set's epoch and age, and the resulting sigmas (ASCII,
        under 80 characters each). Parameters as for :meth:`sigmas`.

        Returns
        -------
        tuple of str
            The lines, for a CDM's ``COMMENT``.

        Raises
        ------
        UnsupportedElements
            If SGP4 cannot be initialised from ``omm`` (:func:`orbit_regime`).
        ValueError
            If ``time`` is not a single time.
        """
        return self._comments(self._evaluated(omm, time), omm, time)

    def described_covariance(
        self, omm: Omm, time: Time
    ) -> tuple[Covariance, tuple[str, ...]]:
        """:meth:`covariance` and :meth:`comments` together, the regime and
        sigmas derived once. Parameters as for :meth:`sigmas`.

        Returns
        -------
        tuple[Covariance, tuple[str, ...]]
            The covariance and the lines describing it.

        Raises
        ------
        UnsupportedElements
            If SGP4 cannot be initialised from ``omm`` (:func:`orbit_regime`).
        ValueError
            If ``time`` is not a single time.
        """
        evaluated = self._evaluated(omm, time)
        return _covariance(evaluated.sigmas, time), self._comments(evaluated, omm, time)

    def _evaluated(self, omm: Omm, time: Time) -> _Evaluated:
        """The regime, its parameters and the sigmas of ``omm`` at ``time``."""
        require_single_time(time, "time")
        regime = orbit_regime(omm)
        parameters = self.parameters(regime)
        age = abs(julian_seconds_between(omm.epoch, time.utc))
        transverse = parameters.transverse + parameters.transverse_rate * age
        factor = self.safety_factor
        sigmas = RtnSigmas(
            factor * parameters.radial,
            factor * transverse,
            factor * parameters.normal,
        )
        return _Evaluated(regime, parameters, sigmas)

    def _comments(self, evaluated: _Evaluated, omm: Omm, time: Time) -> tuple[str, ...]:
        """The :meth:`comments` lines of an evaluation."""
        regime, parameters, sigmas = evaluated
        age = julian_seconds_between(omm.epoch, time.utc) / DAY
        epoch = omm.epoch.utc.isot
        return (
            "Synthetic covariance: keepout GP element-set age model, an assumption",
            "sigma_T = f (sigma_T0 + k |t - epoch|); sigma_R, sigma_N constant",
            (
                f"Regime {regime.value}: sigma_R0 {parameters.radial / KM:.3f} km, "
                f"sigma_T0 {parameters.transverse / KM:.3f} km, "
                f"sigma_N0 {parameters.normal / KM:.3f} km"
            ),
            (
                f"k {parameters.transverse_rate * DAY / KM:.3f} km/day, "
                f"safety factor f {self.safety_factor:g}"
            ),
            f"Element set epoch {epoch} UTC, age {age:+.3f} days",
            (
                f"Sigmas: R {sigmas.radial / KM:.3f} km, "
                f"T {sigmas.transverse / KM:.3f} km, N {sigmas.normal / KM:.3f} km"
            ),
        )


class _Evaluated(NamedTuple):
    """The model at one element set and time."""

    regime: OrbitRegime
    parameters: RegimeParameters
    sigmas: RtnSigmas


def _covariance(sigmas: RtnSigmas, time: Time) -> Covariance:
    """The diagonal RTN position covariance of ``sigmas`` at ``time``."""
    variances = np.array(sigmas)
    return Covariance.from_position(np.diag(variances * variances), "RTN", time)
