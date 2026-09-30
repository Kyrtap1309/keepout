"""Equinoctial orbital elements and two-body motion.

The elements ``(n, af, ag, chi, psi, lambda_M)`` — mean motion, the two
eccentricity components, the two inclination components and the mean
longitude — are non-singular for circular and equatorial orbits (Broucke &
Cefola 1972; Vallado & Alfano 2015). A retrograde factor ``fr = -1`` moves the
remaining singularity from ``i = 180 deg`` to ``i = 0``; each orbit keeps the
factor chosen from its own angular momentum.

Under two-body motion only the mean longitude changes, ``lambda_M(t) =
lambda_M(0) + n t``, so the 6x6 state transition matrix of the elements is the
identity plus ``t`` in the ``(lambda_M, n)`` entry (Hall 2021, eq. 40).

Every conversion accepts complex input, so :func:`complex_step_jacobian`
gives derivatives accurate to machine precision.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Literal, NamedTuple, TypeAlias

import numpy as np
import numpy.typing as npt

from keepout._types import FloatArray
from keepout.errors import GeometryError
from keepout.orbits import EARTH_MU

RetrogradeFactor: TypeAlias = Literal[1, -1]

KEPLER_TOLERANCE = 1e-14
"""Convergence of the eccentric-longitude Newton iteration [rad]."""

KEPLER_RESIDUAL_TOLERANCE = 64.0 * float(np.finfo(np.float64).eps)
"""Residual of Kepler's equation at the rounding level of its terms [rad]. Near
perihelion of a very eccentric orbit ``dlambda/dF = 1 - e cos E`` is tiny, so
the Newton step can stall above :data:`KEPLER_TOLERANCE` while the residual
is already at rounding level; one more step from there is final."""

KEPLER_MAX_ITERATIONS = 60

COMPLEX_STEP = 1e-30
"""Default imaginary step of :func:`complex_step_jacobian`."""

SINGULARITY_TOLERANCE = 1e-12
"""Smallest accepted ``|h| / (|r| |v|)`` and ``1 + fr w_z`` (both vanish at a
singularity of the elements)."""


def retrograde_factor(r: npt.ArrayLike, v: npt.ArrayLike) -> RetrogradeFactor:
    """``+1`` for a prograde orbit (angular momentum ``h_z >= 0``), else ``-1``."""
    h = np.cross(np.real(np.asarray(r)), np.real(np.asarray(v)))
    return 1 if h[2] >= 0.0 else -1


def _basis(chi: Any, psi: Any, fr: RetrogradeFactor) -> tuple[Any, Any]:
    """Unit vectors ``f`` and ``g`` of the equinoctial frame, shape ``(..., 3)``."""
    d = 1.0 + chi * chi + psi * psi
    f = np.stack([1.0 - chi * chi + psi * psi, 2.0 * chi * psi, -2.0 * fr * chi], -1)
    g = np.stack(
        [2.0 * fr * chi * psi, fr * (1.0 + chi * chi - psi * psi), 2.0 * psi], -1
    )
    return f / d[..., None], g / d[..., None]


def _atan2(y: Any, x: Any) -> Any:
    """``arctan2`` that also carries a complex-step derivative."""
    if not (np.iscomplexobj(y) or np.iscomplexobj(x)):
        return np.arctan2(y, x)
    y, x = np.asarray(y, complex), np.asarray(x, complex)
    angle = np.arctan2(y.real, x.real)
    return angle + 1j * (x.real * y.imag - y.real * x.imag) / (y.real**2 + x.real**2)


def to_equinoctial(
    state: npt.ArrayLike, fr: RetrogradeFactor, mu: float = EARTH_MU
) -> Any:
    """Equinoctial elements of Cartesian states.

    Parameters
    ----------
    state
        Position and velocity ``(x, y, z, vx, vy, vz)`` [m, m/s], shape
        ``(6,)`` or ``(n, 6)``; complex input is allowed.
    fr
        Retrograde factor; see :func:`retrograde_factor`.
    mu
        Gravitational parameter [m^3/s^2].

    Returns
    -------
    numpy.ndarray
        ``(n [rad/s], af, ag, chi, psi, lambda_M [rad])``, same shape;
        ``lambda_M`` is not wrapped: it lies within ``pi + e`` of zero,
        ``e = |(af, ag)| < 1``.

    Raises
    ------
    GeometryError
        If an orbit is not elliptic (two-body energy >= 0), is rectilinear
        (zero angular momentum), or is equatorial with a retrograde factor
        that puts it on the singularity (``fr = -1`` at ``i = 0``, ``fr = +1``
        at ``i = 180 deg``).
    """
    x = np.asarray(state)
    r, v = x[..., :3], x[..., 3:]
    rn = np.sqrt(np.sum(r * r, -1))
    v2 = np.sum(v * v, -1)
    inverse_a = 2.0 / rn - v2 / mu
    if np.any(np.real(inverse_a) <= 0.0):
        raise GeometryError(
            "equinoctial elements need an elliptic orbit; the state is unbound"
        )
    a = 1.0 / inverse_a
    h = np.cross(r, v)
    hn = np.sqrt(np.sum(h * h, -1))
    if np.any(~(np.real(hn) > SINGULARITY_TOLERANCE * np.real(rn * np.sqrt(v2)))):
        raise GeometryError(
            "equinoctial elements need a nonzero angular momentum; "
            "the state is rectilinear"
        )
    w = h / hn[..., None]
    denominator = 1.0 + fr * w[..., 2]
    if np.any(~(np.real(denominator) > SINGULARITY_TOLERANCE)):
        raise GeometryError(
            f"the retrograde factor fr = {fr} puts an equatorial orbit on the "
            "singularity of the elements; use retrograde_factor(r, v)"
        )
    chi = w[..., 0] / denominator
    psi = -w[..., 1] / denominator
    f, g = _basis(chi, psi, fr)
    eccentricity = (
        (v2 - mu / rn)[..., None] * r - np.sum(r * v, -1)[..., None] * v
    ) / mu
    af, ag = np.sum(eccentricity * f, -1), np.sum(eccentricity * g, -1)
    big_x, big_y = np.sum(r * f, -1), np.sum(r * g, -1)
    root = np.sqrt(1.0 - af * af - ag * ag)
    b = 1.0 / (1.0 + root)
    sin_f = ag + ((1.0 - ag * ag * b) * big_y - af * ag * b * big_x) / (a * root)
    cos_f = af + ((1.0 - af * af * b) * big_x - af * ag * b * big_y) / (a * root)
    longitude = _atan2(sin_f, cos_f)
    mean_longitude = longitude + ag * cos_f - af * sin_f
    return np.stack([np.sqrt(mu / a**3), af, ag, chi, psi, mean_longitude], -1)


def to_cartesian(
    elements: npt.ArrayLike, fr: RetrogradeFactor, mu: float = EARTH_MU
) -> Any:
    """Cartesian states of equinoctial elements (inverse of :func:`to_equinoctial`).

    Parameters
    ----------
    elements
        ``(n, af, ag, chi, psi, lambda_M)``, shape ``(6,)`` or ``(m, 6)``;
        complex input is allowed.
    fr
        Retrograde factor the elements were computed with.
    mu
        Gravitational parameter [m^3/s^2].

    Returns
    -------
    numpy.ndarray
        ``(x, y, z, vx, vy, vz)`` [m, m/s], same shape.

    Raises
    ------
    GeometryError
        If Kepler's equation does not converge (e.g. NaN elements).
    """
    array = np.asarray(elements)
    n, af, ag, chi, psi, lam = (array[..., k] for k in range(6))
    a = (mu / (n * n)) ** (1.0 / 3.0)
    shape = _Shape.of(af, ag)
    f, g = _basis(chi, psi, fr)
    return _state(n, a, af, ag, shape, _eccentric_longitude(lam, af, ag), f, g)


class _Shape(NamedTuple):
    """Coefficients of the in-plane state that depend on ``af`` and ``ag`` only."""

    c_ag: Any
    """``1 - ag^2 b``, with ``b = 1 / (1 + sqrt(1 - af^2 - ag^2))``."""
    c_af: Any
    """``1 - af^2 b``."""
    cross: Any
    """``af ag b``."""

    @classmethod
    def of(cls, af: Any, ag: Any) -> _Shape:
        root = np.sqrt(1.0 - af * af - ag * ag)
        b = 1.0 / (1.0 + root)
        return cls(1.0 - ag * ag * b, 1.0 - af * af * b, af * ag * b)


def _eccentric_longitude(lam: Any, af: Any, ag: Any) -> Any:
    """Solve Kepler's equation for the eccentric longitude ``F``.

    Raises
    ------
    GeometryError
        If it does not converge (e.g. NaN elements).
    """
    # Kepler's equation in equinoctial form: lam = F + ag cos F - af sin F,
    # i.e. E - e sin E = M with F = E + varpi, lam = M + varpi. Reducing lam
    # by a real multiple of 2 pi keeps the iterate small (absolute tolerance)
    # and leaves the complex-step derivative untouched; only sin F and cos F
    # are used below.
    lam = lam - 2.0 * math.pi * np.round(np.real(lam) / (2.0 * math.pi))
    # Danby's start E0 = M + 0.85 e sign(sin M), on real parts: Newton then
    # converges for every e < 1 (Danby 1987).
    e_real = np.hypot(np.real(af), np.real(ag))
    varpi = np.arctan2(np.real(ag), np.real(af))
    longitude = lam + 0.85 * e_real * np.sign(np.sin(np.real(lam) - varpi))
    # Promote to complex when the elements are (a complex-step derivative).
    longitude = longitude + 0.0 * af
    for _ in range(KEPLER_MAX_ITERATIONS):
        residual = longitude + ag * np.cos(longitude) - af * np.sin(longitude) - lam
        step = residual / (1.0 - ag * np.sin(longitude) - af * np.cos(longitude))
        longitude = longitude - step
        converged = (np.abs(step) < KEPLER_TOLERANCE) | (
            np.abs(residual) <= KEPLER_RESIDUAL_TOLERANCE
        )
        if np.all(converged):
            return longitude
    raise GeometryError(
        "Kepler's equation did not converge in "
        f"{KEPLER_MAX_ITERATIONS} iterations; the elements are not a "
        "valid elliptic orbit"
    )


def _state(
    n: Any, a: Any, af: Any, ag: Any, shape: _Shape, longitude: Any, f: Any, g: Any
) -> Any:
    """Cartesian state from the eccentric longitude and the orbit's constants."""
    cos_f, sin_f = np.cos(longitude), np.sin(longitude)
    big_x = a * (shape.c_ag * cos_f + shape.cross * sin_f - af)
    big_y = a * (shape.c_af * sin_f + shape.cross * cos_f - ag)
    radius = a * (1.0 - af * cos_f - ag * sin_f)
    rate = n * a * a / radius
    big_xd = rate * (shape.cross * cos_f - shape.c_ag * sin_f)
    big_yd = rate * (shape.c_af * cos_f - shape.cross * sin_f)
    position = big_x[..., None] * f + big_y[..., None] * g
    velocity = big_xd[..., None] * f + big_yd[..., None] * g
    return np.concatenate([position, velocity], -1)


class Orbits:
    """Real elliptic orbits, prepared for evaluation at many times.

    Everything but the mean longitude is constant along two-body motion, so
    it is computed once; :meth:`cartesian` then only solves Kepler's
    equation. The states equal ``to_cartesian(propagate(elements, t), fr)``.

    Parameters
    ----------
    elements
        Real equinoctial elements at TCA, shape ``(m, 6)``.
    fr
        Retrograde factor.
    mu
        Gravitational parameter [m^3/s^2].
    """

    def __init__(
        self, elements: FloatArray, fr: RetrogradeFactor, mu: float = EARTH_MU
    ) -> None:
        self.elements = np.asarray(elements, dtype=np.float64)
        n, af, ag, chi, psi = (self.elements[:, k] for k in range(5))
        self._n, self._af, self._ag = n, af, ag
        self._a = (mu / (n * n)) ** (1.0 / 3.0)
        self._shape = _Shape.of(af, ag)
        self._f, self._g = _basis(chi, psi, fr)

    def cartesian(
        self, t: float | FloatArray, rows: slice | npt.NDArray[np.intp] = slice(None)
    ) -> FloatArray:
        """States ``(k, 6)`` of the orbits ``rows`` at ``t`` [s from TCA]."""
        n, af, ag = self._n[rows], self._af[rows], self._ag[rows]
        lam = self.elements[rows, 5] + n * np.asarray(t)
        shape = _Shape(*(c[rows] for c in self._shape))
        state: FloatArray = _state(
            n,
            self._a[rows],
            af,
            ag,
            shape,
            _eccentric_longitude(lam, af, ag),
            self._f[rows],
            self._g[rows],
        )
        return state


def propagate(elements: npt.ArrayLike, dt: npt.ArrayLike) -> Any:
    """Two-body motion of equinoctial elements: ``lambda_M += n dt``.

    Parameters
    ----------
    elements
        Elements, shape ``(6,)`` or ``(m, 6)``.
    dt
        Time step [s], a scalar or shape ``(m,)``.
    """
    propagated = np.array(elements, dtype=np.result_type(np.asarray(elements), float))
    propagated[..., 5] = propagated[..., 5] + propagated[..., 0] * np.asarray(dt)
    return propagated


def wrap_angle(angle: Any) -> Any:
    """Angle difference wrapped to ``[-pi, pi)`` (real input only)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def complex_step(
    function: Callable[[Any], Any], x: npt.ArrayLike, step: float = COMPLEX_STEP
) -> tuple[FloatArray, FloatArray]:
    """Value and Jacobian of a real-analytic vector function by the complex step.

    As :func:`complex_step_jacobian`, which see; the value ``f(x)`` is the
    real part of the same evaluation (exact to ``O(h^2)``).
    """
    point = np.asarray(x, dtype=np.float64)
    value = function(point + 1j * step * np.eye(point.size))
    if not np.iscomplexobj(value):
        raise TypeError(
            "complex_step_jacobian needs a function that keeps complex input "
            "complex; it returned a real array, which would give a zero Jacobian"
        )
    real: FloatArray = np.real(value[0])
    jacobian: FloatArray = np.imag(value).T / step
    return real, jacobian


def complex_step_jacobian(
    function: Callable[[Any], Any], x: npt.ArrayLike, step: float = COMPLEX_STEP
) -> FloatArray:
    """Jacobian of a real-analytic vector function by the complex step.

    ``J[:, k] = Im f(x + i h e_k) / h`` has no subtractive cancellation, so it
    is accurate to machine precision for any tiny ``h``.

    Parameters
    ----------
    function
        Maps complex vectors of shape ``(m, n)`` row by row to ``(m, k)``
        (every conversion in this module does).
    x
        Point of evaluation (real), shape ``(n,)``.
    step
        Imaginary step ``h``.
    """
    return complex_step(function, x, step)[1]
