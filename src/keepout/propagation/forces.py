"""The forces of a Cowell integration: zonal gravity and optional drag.

:class:`ForceModel` gives the acceleration of N states at once, in an inertial
frame whose orientation it knows only through the Earth's rotation axis
(:func:`earth_pole`). Gravity is the gradient of

    ``Phi = (mu / r) [1 - sum_n Jn (R / r)^n Pn(s)]``, ``s = r_hat . pole``,

to degree 6; tesseral terms, third bodies and solar radiation pressure are left
out. Drag is ``-1/2 rho (CD A / m) |v_rel| v_rel`` in the air that turns with
the Earth, with the density of an exponential atmosphere at the spherical
altitude ``|r| - EARTH_RADIUS``.

The coefficients are the fully normalised tide-free zonals of EGM2008 (Pavlis
et al., "The development and evaluation of the Earth Gravitational Model 2008
(EGM2008)", J. Geophys. Res. 117, B04406, 2012), as the ``gfc n 0`` lines of
``EGM2008.gfc`` on the ICGEM service give them. The model's own GM is
3.986004415e14 m³/s²; :data:`~keepout.orbits.EARTH_MU` is used here as
everywhere in keepout, 7.5e-10 away from it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Final, Self

import erfa
import numpy as np

from keepout._linalg import as_vector
from keepout._types import FloatArray, override
from keepout._validation import non_negative_float, require_single_time
from keepout.orbits import EARTH_MU
from keepout.propagation._atmosphere import density

if TYPE_CHECKING:
    from astropy.time import Time

__all__ = ["EARTH_RADIUS", "ForceModel", "earth_pole"]

EARTH_RADIUS: Final = 6_378_136.3
"""Reference radius of EGM2008 [m]."""

_EARTH_ROTATION: Final = 7.292115e-5
"""Rotation rate of the Earth [rad/s], which carries the air along."""

_NORMALISED_ZONALS: Final = (
    -0.484165143790815e-3,
    0.957161207093473e-6,
    0.539965866638991e-6,
    0.686702913736681e-7,
    -0.149953927978527e-6,
)
"""EGM2008 tide-free C̄n0 for n = 2, 3, ... (see the module docstring)."""

_ZONALS: Final = tuple(
    -coefficient * math.sqrt(2 * n + 1)
    for n, coefficient in enumerate(_NORMALISED_ZONALS, start=2)
)
"""Jn for n = 2, 3, ..."""

_MAX_DEGREE: Final = len(_ZONALS) + 1

_UNIT_TOLERANCE: Final = 1e-12
"""How far from 1 the norm of a unit vector may be."""


def earth_pole(epoch: Time) -> FloatArray:
    """The Earth's rotation axis at a time, in GCRF.

    The direction of the Celestial Intermediate Pole, by IAU 2006/2000A
    precession-nutation: the third row of ``erfa.pnm06a``. Polar motion, about
    ten metres at the surface, is left out.

    Parameters
    ----------
    epoch
        The time; a single time, in any time scale.

    Returns
    -------
    FloatArray
        Unit vector, shape ``(3,)``.

    Raises
    ------
    ValueError
        If ``epoch`` is not a single time.
    """
    require_single_time(epoch, "epoch")
    terrestrial = epoch.tt
    rotation = erfa.pnm06a(terrestrial.jd1, terrestrial.jd2)
    pole: FloatArray = np.array(rotation[2], dtype=np.float64)
    return pole


@dataclass(frozen=True, eq=False, slots=True)
class ForceModel:
    """Gravity to some zonal degree, and drag if a drag term is given.

    Parameters
    ----------
    pole
        The Earth's rotation axis in the frame of the states, shape ``(3,)``,
        a unit vector (:func:`earth_pole`). It stays fixed over an
        integration: precession turns it by about 0.06 arcsec a day.
    degree
        Highest zonal degree, 0 to 6; below 2 there is only the central force.
    drag_term
        CD·A/m [m²/kg], finite and >= 0; ``None`` for no drag.

    Raises
    ------
    ValueError
        If ``pole`` is not a unit vector of shape ``(3,)``, ``degree`` is out
        of range or ``drag_term`` is negative or not finite.
    """

    pole: FloatArray
    degree: int = field(default=6, kw_only=True)
    drag_term: float | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        pole = as_vector(self.pole, "pole")
        if not abs(np.linalg.norm(pole) - 1.0) <= _UNIT_TOLERANCE:
            raise ValueError(f"pole must be a unit vector, got {pole}")
        if not 0 <= self.degree <= _MAX_DEGREE:
            raise ValueError(
                f"degree must be from 0 to {_MAX_DEGREE}, got {self.degree}"
            )
        object.__setattr__(self, "pole", pole)
        if self.drag_term is not None:
            term = non_negative_float(self.drag_term, "drag_term")
            object.__setattr__(self, "drag_term", term)

    @override
    def __reduce__(self) -> tuple[partial[Self], tuple[FloatArray]]:
        # Rebuilt through the constructor, so the pole comes back read-only.
        rebuild = partial(type(self), degree=self.degree, drag_term=self.drag_term)
        return rebuild, (self.pole,)

    def acceleration(self, r: FloatArray, v: FloatArray) -> FloatArray:
        """Acceleration of N states.

        Parameters
        ----------
        r
            Positions [m], shape ``(N, 3)``, not at the Earth's centre.
        v
            Velocities [m/s], shape ``(N, 3)``, used for drag only.

        Returns
        -------
        FloatArray
            Accelerations [m/s²], shape ``(N, 3)``.
        """
        radius = np.linalg.norm(r, axis=1, keepdims=True)
        gravity = self._gravity(r / radius, radius)
        if self.drag_term is None:
            return gravity
        air_velocity = _EARTH_ROTATION * np.cross(self.pole, r)
        relative = v - air_velocity
        speed = np.linalg.norm(relative, axis=1, keepdims=True)
        rho = density(radius - EARTH_RADIUS)
        total: FloatArray = gravity - 0.5 * rho * self.drag_term * speed * relative
        return total

    def derivative(self, state: FloatArray) -> FloatArray:
        """Time derivative of N states, for an ODE solver.

        Parameters
        ----------
        state
            Flat array of shape ``(6N,)``: each object's position [m] and
            velocity [m/s], the ravelled ``(N, 6)`` array.

        Returns
        -------
        FloatArray
            Each object's velocity and acceleration, shape ``(6N,)``.
        """
        states = state.reshape(-1, 6)
        r, v = states[:, :3], states[:, 3:]
        derivative: FloatArray = np.hstack((v, self.acceleration(r, v))).ravel()
        return derivative

    def _gravity(self, direction: FloatArray, radius: FloatArray) -> FloatArray:
        """The gradient of the potential, from unit vectors ``(N, 3)`` along
        the positions and radii ``(N, 1)``.

        With ``rho = R / r``, the gradient of the potential is

        ``-(mu / r²) [r_hat (1 - sum Jn rho^n ((n + 1) Pn + s P'n))
        + pole sum Jn rho^n P'n]``

        with Pn and P'n from Bonnet's recursion, one degree at a time.
        """
        sine = np.sum(direction * self.pole, axis=1, keepdims=True)
        ratio = EARTH_RADIUS / radius
        previous, legendre = np.ones_like(radius), sine
        slope = np.ones_like(radius)
        along_radius = np.ones_like(radius)
        along_pole = np.zeros_like(radius)
        for n, zonal in enumerate(_ZONALS[: max(self.degree - 1, 0)], start=2):
            previous, legendre = (
                legendre,
                ((2 * n - 1) * sine * legendre - (n - 1) * previous) / n,
            )
            slope = n * previous + sine * slope
            term = zonal * ratio**n
            along_radius -= term * ((n + 1) * legendre + sine * slope)
            along_pole += term * slope
        gravity: FloatArray = (
            -EARTH_MU / radius**2 * (along_radius * direction + along_pole * self.pole)
        )
        return gravity
