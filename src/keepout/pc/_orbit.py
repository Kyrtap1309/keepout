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
from functools import cached_property

import numpy as np

from keepout._types import FloatArray, from_eigen, symmetric_part
from keepout.covariance import to_correlation
from keepout.equinoctial import (
    RetrogradeFactor,
    complex_step,
    propagate,
    retrograde_factor,
    semi_major_axis,
    to_cartesian,
    to_equinoctial,
    wrap_angle,
)
from keepout.errors import CovarianceError
from keepout.objects import ObjectState
from keepout.orbits import EARTH_MU


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
        elements, to_elements = complex_step(lambda x: to_equinoctial(x, fr), state)
        covariance = symmetric_part(
            to_elements @ item.covariance.matrix @ to_elements.T
        )
        return cls(elements, covariance, fr, item.covariance.position)

    def expansions(
        self, centres: FloatArray, t: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """First-order Gaussians of the Cartesian state (Hall 2021, eq. 44).

        At each time the two-body motion is linearised about its own TCA
        elements, which need not be the mean; the Jacobians come from one
        batched complex-step evaluation.

        Parameters
        ----------
        centres
            TCA elements to linearise about, shape ``(m, 6)``.
        t
            Times [s from TCA], shape ``(m,)``.

        Returns
        -------
        tuple
            Mean states ``(m, 6)`` and covariances ``(m, 6, 6)``.
        """
        times = np.asarray(t, dtype=np.float64).ravel()
        centres = np.asarray(centres, dtype=np.float64).reshape(times.size, 6)
        # Each centre's perturbations arrive as axis 1, so its time broadcasts.
        state, jacobian = complex_step(
            lambda e: to_cartesian(propagate(e, times[:, None]), self.fr), centres
        )
        offset = self.elements - centres
        offset[:, 5] = wrap_angle(offset[:, 5])
        mean: FloatArray = state + (jacobian @ offset[..., None])[..., 0]
        covariance = jacobian @ self.covariance @ jacobian.transpose(0, 2, 1)
        return mean, symmetric_part(covariance)

    def mean_expansions(self, t: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Mean-centred :meth:`expansions` at times ``t`` ``(m,)``."""
        times = np.asarray(t, dtype=np.float64).ravel()
        return self.expansions(np.broadcast_to(self.elements, (times.size, 6)), times)

    @property
    def max_acceleration(self) -> float:
        """Largest two-body acceleration on the mean orbit (at perigee) [m/s^2]."""
        n, af, ag = (float(v) for v in self.elements[:3])
        semi_major = semi_major_axis(n)
        perigee = semi_major * (1.0 - math.hypot(af, ag))
        return float(EARTH_MU / (perigee * perigee))

    def centre_at(self, state: FloatArray, t: float | FloatArray) -> FloatArray:
        """TCA elements of the orbits through the Cartesian states at ``t``.

        ``state`` is ``(6,)`` or ``(m, 6)`` with ``t`` a scalar or ``(m,)``.

        Raises
        ------
        GeometryError
            If an orbit is not elliptic.
        """
        elements = np.asarray(to_equinoctial(state, self.fr), dtype=np.float64)
        result: FloatArray = propagate(elements, -np.asarray(t))
        return result

    @cached_property
    def _factor(self) -> FloatArray:
        """``L`` with ``L L^T`` the element covariance, for :meth:`sample`."""
        # Factor the correlation matrix: the elements differ in scale by many
        # orders of magnitude (n in rad/s, lambda_M in rad).
        deviations, correlation = to_correlation(self.covariance)
        values, vectors = np.linalg.eigh(correlation)
        # The symmetric square root, unlike vectors * sqrt(values), does not
        # depend on the eigenvector signs and bases LAPACK picks on each CPU.
        root = from_eigen(np.sqrt(np.clip(values, 0.0, None)), vectors)
        factor: FloatArray = deviations[:, None] * root
        return factor

    def sample(self, rng: np.random.Generator, count: int) -> FloatArray:
        """``count`` TCA element vectors ``(count, 6)`` drawn from the Gaussian."""
        drawn: FloatArray = (
            self.elements + rng.standard_normal((count, 6)) @ self._factor.T
        )
        return drawn

    @property
    def period(self) -> float:
        """Orbital period [s]."""
        return 2.0 * math.pi / float(self.elements[0])
