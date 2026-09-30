"""One object's orbit and its uncertainty in equinoctial elements.

The 3-D methods treat the Gaussian uncertainty at TCA as Gaussian in the
equinoctial elements, not in Cartesian space (Hall 2021, sec. IV): a sample
displaced along track then stays on a curved orbit instead of leaving it along
the tangent. For large along-track uncertainty this matters — a 180 km
displacement along the tangent puts a Cartesian sample ~2.3 km above the orbit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from keepout._types import FloatArray
from keepout.equinoctial import (
    RetrogradeFactor,
    complex_step_jacobian,
    propagate,
    retrograde_factor,
    to_cartesian,
    to_equinoctial,
    wrap_angle,
)
from keepout.errors import CovarianceError
from keepout.objects import ObjectState
from keepout.orbits import EARTH_MU

_COMPLEX_STEP = 1e-30
"""Imaginary step of :meth:`OrbitUncertainty.mean_expansions` (as in
:func:`~keepout.equinoctial.complex_step_jacobian`)."""


def _symmetric(matrix: FloatArray) -> FloatArray:
    result: FloatArray = 0.5 * (matrix + matrix.T)
    return result


@dataclass(frozen=True, eq=False)
class OrbitUncertainty:
    """Mean equinoctial elements at TCA and their covariance.

    Parameters
    ----------
    elements
        Mean elements ``(n, af, ag, chi, psi, lambda_M)`` at TCA.
    covariance
        6x6 covariance of the elements, mapped from the Cartesian covariance
        through the Jacobian at the mean state.
    fr
        Retrograde factor of the orbit.
    position_covariance
        The Cartesian 3x3 position covariance at TCA [m^2].
    """

    elements: FloatArray
    covariance: FloatArray
    fr: RetrogradeFactor
    position_covariance: FloatArray

    @classmethod
    def of(cls, item: ObjectState) -> OrbitUncertainty:
        """The orbit of an object with a known covariance.

        Raises
        ------
        CovarianceError
            If the object's covariance is unknown.
        GeometryError
            If the orbit is not elliptic.
        """
        if item.covariance is None:
            raise CovarianceError(f"covariance of {item.object_id} is unknown")
        state = np.concatenate([item.state.r, item.state.v])
        fr = retrograde_factor(item.state.r, item.state.v)
        elements = np.asarray(to_equinoctial(state, fr), dtype=np.float64)
        to_elements = np.linalg.inv(
            complex_step_jacobian(lambda e: to_cartesian(e, fr), elements)
        )
        covariance = _symmetric(to_elements @ item.covariance.matrix @ to_elements.T)
        return cls(elements, covariance, fr, item.covariance.position)

    def cartesian(self, elements: FloatArray, t: float | FloatArray) -> FloatArray:
        """Cartesian states ``(..., 6)`` at ``t`` [s from TCA] of TCA elements."""
        state: FloatArray = np.asarray(
            to_cartesian(propagate(elements, t), self.fr), dtype=np.float64
        )
        return state

    def expansion(self, centre: FloatArray, t: float) -> tuple[FloatArray, FloatArray]:
        """First-order Gaussian of the Cartesian state at ``t`` (Hall 2021, eq. 44).

        Linearises the two-body motion about the TCA elements ``centre``,
        which need not be the mean.

        Returns
        -------
        tuple
            Mean state ``X`` (6,) and covariance ``P`` (6x6) at ``t``.
        """
        jacobian = complex_step_jacobian(
            lambda e: to_cartesian(propagate(e, t), self.fr), centre
        )
        offset = self.elements - centre
        offset[5] = wrap_angle(offset[5])
        mean = self.cartesian(centre, t) + jacobian @ offset
        return mean, _symmetric(jacobian @ self.covariance @ jacobian.T)

    def mean_expansions(self, t: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Mean-centred :meth:`expansion` at many times at once.

        Parameters
        ----------
        t
            Times [s from TCA], shape ``(m,)``.

        Returns
        -------
        tuple
            Mean states ``(m, 6)`` and covariances ``(m, 6, 6)``.
        """
        times = np.asarray(t, dtype=np.float64).ravel()
        # The complex-step Jacobian of every time in one vectorised call.
        shifted = self.elements + 1j * _COMPLEX_STEP * np.eye(6)
        rows = np.broadcast_to(shifted, (times.size, 6, 6)).reshape(-1, 6)
        value = np.asarray(
            to_cartesian(propagate(rows, np.repeat(times, 6)), self.fr)
        ).reshape(times.size, 6, 6)
        jacobian = np.imag(value).transpose(0, 2, 1) / _COMPLEX_STEP
        mean: FloatArray = np.real(value[:, 0, :])
        covariance = jacobian @ self.covariance @ jacobian.transpose(0, 2, 1)
        symmetric: FloatArray = 0.5 * (covariance + covariance.transpose(0, 2, 1))
        return mean, symmetric

    @property
    def max_acceleration(self) -> float:
        """Largest two-body acceleration on the mean orbit (at perigee) [m/s^2]."""
        n, af, ag = (float(v) for v in self.elements[:3])
        semi_major = (EARTH_MU / (n * n)) ** (1.0 / 3.0)
        perigee = semi_major * (1.0 - math.hypot(af, ag))
        return float(EARTH_MU / (perigee * perigee))

    def centre_at(self, state: FloatArray, t: float) -> FloatArray:
        """TCA elements of the orbit through the Cartesian ``state`` at ``t``."""
        elements = np.asarray(to_equinoctial(state, self.fr), dtype=np.float64)
        result: FloatArray = propagate(elements, -t)
        return result

    def sample(self, rng: np.random.Generator, count: int) -> FloatArray:
        """``count`` TCA element vectors ``(count, 6)`` drawn from the Gaussian."""
        # Factor the correlation matrix: the elements differ in scale by many
        # orders of magnitude (n in rad/s, lambda_M in rad).
        scale = np.sqrt(np.clip(np.diag(self.covariance), 0.0, None))
        safe = np.where(scale > 0.0, scale, 1.0)
        values, vectors = np.linalg.eigh(self.covariance / np.outer(safe, safe))
        factor = (scale[:, None] * vectors) * np.sqrt(np.clip(values, 0.0, None))
        drawn: FloatArray = self.elements + rng.standard_normal((count, 6)) @ factor.T
        return drawn

    @property
    def period(self) -> float:
        """Orbital period [s]."""
        return 2.0 * math.pi / float(self.elements[0])
