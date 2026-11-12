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

Every conversion accepts complex input, so :func:`complex_step` gives
derivatives accurate to machine precision.
"""

# The conversions are generic over a constrained TypeVar (real or complex
# arrays). mypy checks each constraint in turn; pyright cannot follow NumPy's
# overloads through one (it types ``chi * chi`` of float64 arrays as
# complex), so the lines it misreads carry a pyright ignore.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Generic,
    Literal,
    NamedTuple,
    TypeAlias,
    TypeVar,
)

import numpy as np
import numpy.typing as npt

from keepout._linalg import take
from keepout._types import ComplexArray, FloatArray, IndexArray
from keepout.errors import GeometryError
from keepout.orbits import EARTH_MU

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "COMPLEX_STEP",
    "KEPLER_MAX_ITERATIONS",
    "KEPLER_RESIDUAL_TOLERANCE",
    "KEPLER_TOLERANCE",
    "SINGULARITY_TOLERANCE",
    "Orbits",
    "RetrogradeFactor",
    "complex_step",
    "propagate",
    "retrograde_factor",
    "semi_major_axis",
    "to_cartesian",
    "to_equinoctial",
    "wrap_angle",
]

RetrogradeFactor: TypeAlias = Literal[1, -1]

_Array = TypeVar("_Array", FloatArray, ComplexArray)
"""A real array, or a complex one carrying a complex-step derivative; the
conversions return the same kind."""

_Number = TypeVar("_Number", float, FloatArray, ComplexArray)
"""As ``_Array``, or a scalar."""

_Real = TypeVar("_Real", float, FloatArray)
"""A real scalar or array."""

KEPLER_TOLERANCE: Final = 1e-14
"""Convergence of the eccentric-longitude Newton iteration [rad]."""

KEPLER_RESIDUAL_TOLERANCE: Final = 64.0 * float(np.finfo(np.float64).eps)
"""Residual of Kepler's equation at the rounding level of its terms [rad]. Near
perihelion of a very eccentric orbit ``dlambda/dF = 1 - e cos E`` is tiny, so
the Newton step can stall above :data:`KEPLER_TOLERANCE` while the residual
is already at rounding level; one more step from there is final."""

KEPLER_MAX_ITERATIONS: Final = 60

COMPLEX_STEP: Final = 1e-30
"""Default imaginary step of :func:`complex_step`."""

SINGULARITY_TOLERANCE: Final = 1e-12
"""Smallest accepted ``|h| / (|r| |v|)`` and ``1 + fr w_z`` (both vanish at a
singularity of the elements)."""


def retrograde_factor(r: npt.ArrayLike, v: npt.ArrayLike) -> RetrogradeFactor:
    """``+1`` for a prograde orbit (angular momentum ``h_z >= 0``), else ``-1``.

    Parameters
    ----------
    r
        Position [m], shape ``(3,)``; a complex-step perturbation is ignored.
    v
        Velocity [m/s], shape ``(3,)``; likewise.
    """
    h = np.cross(np.real(np.asarray(r)), np.real(np.asarray(v)))
    return 1 if h[2] >= 0.0 else -1


def _basis(chi: _Array, psi: _Array, fr: RetrogradeFactor) -> tuple[_Array, _Array]:
    """Unit vectors ``f`` and ``g`` of the equinoctial frame, shape ``(..., 3)``."""
    d = 1.0 + chi * chi + psi * psi
    f = np.stack([1.0 - chi * chi + psi * psi, 2.0 * chi * psi, -2.0 * fr * chi], -1)
    g = np.stack(
        [2.0 * fr * chi * psi, fr * (1.0 + chi * chi - psi * psi), 2.0 * psi], -1
    )
    return f / d[..., None], g / d[..., None]  # pyright: ignore[reportReturnType]


def _atan2(y: _Array, x: _Array) -> _Array:
    """``arctan2`` that also carries a complex-step derivative."""
    if not (np.iscomplexobj(y) or np.iscomplexobj(x)):
        real: _Array = np.arctan2(y, x)  # pyright: ignore[reportAssignmentType]
        return real
    y, x = np.asarray(y, complex), np.asarray(x, complex)  # pyright: ignore[reportAssignmentType]
    angle = np.arctan2(y.real, x.real)
    result: _Array = angle + 1j * (x.real * y.imag - y.real * x.imag) / (  # pyright: ignore[reportAssignmentType]
        y.real**2 + x.real**2
    )
    return result


def to_equinoctial(state: _Array, fr: RetrogradeFactor, mu: float = EARTH_MU) -> _Array:
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
    root: _Array = np.sqrt(1.0 - af * af - ag * ag)  # pyright: ignore[reportAssignmentType]
    shape = _Shape.of(af, ag, root)
    sin_f = ag + (shape.c_ag * big_y - shape.cross * big_x) / (a * root)
    cos_f = af + (shape.c_af * big_x - shape.cross * big_y) / (a * root)
    eccentric_longitude = _atan2(sin_f, cos_f)
    mean_longitude = eccentric_longitude + ag * cos_f - af * sin_f
    return np.stack([np.sqrt(mu / a**3), af, ag, chi, psi, mean_longitude], -1)  # pyright: ignore[reportReturnType]


def to_cartesian(
    elements: _Array, fr: RetrogradeFactor, mu: float = EARTH_MU
) -> _Array:
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
    a = semi_major_axis(n, mu)  # pyright: ignore[reportArgumentType]
    shape = _Shape.of(af, ag)
    f, g = _basis(chi, psi, fr)  # pyright: ignore[reportArgumentType]
    return _state(n, a, af, ag, shape, _eccentric_longitude(lam, af, ag), f, g)  # pyright: ignore[reportArgumentType]


def semi_major_axis(n: _Number, mu: float = EARTH_MU) -> _Number:
    """Semi-major axis [m] of the mean motion ``n`` [rad/s] (Kepler's third law).

    Parameters
    ----------
    n
        Mean motion [rad/s], a number or an array.
    mu
        Gravitational parameter [m^3/s^2]; the Earth's by default.
    """
    a: _Number = (mu / (n * n)) ** (1.0 / 3.0)  # pyright: ignore[reportAssignmentType]
    return a


class _Shape(NamedTuple, Generic[_Array]):
    """Coefficients of the in-plane state that depend on ``af`` and ``ag`` only."""

    c_ag: _Array
    """``1 - ag^2 b``, with ``b = 1 / (1 + sqrt(1 - af^2 - ag^2))``."""
    c_af: _Array
    """``1 - af^2 b``."""
    cross: _Array
    """``af ag b``."""

    @classmethod
    def of(cls, af: _Array, ag: _Array, root: _Array | None = None) -> _Shape[_Array]:
        """The coefficients; ``root = sqrt(1 - af^2 - ag^2)`` if already known."""
        if root is None:
            root = np.sqrt(1.0 - af * af - ag * ag)  # pyright: ignore[reportAssignmentType]
        b = 1.0 / (1.0 + root)  # pyright: ignore[reportOperatorIssue]
        return cls(1.0 - ag * ag * b, 1.0 - af * af * b, af * ag * b)  # pyright: ignore[reportArgumentType]


class _Apse(NamedTuple):
    """Real eccentricity and longitude of perigee, for the Kepler start."""

    eccentricity: FloatArray
    """``e = |(af, ag)|`` of the real parts."""
    perigee_longitude: FloatArray
    """``varpi = atan2(ag, af)`` of the real parts [rad]."""

    @classmethod
    def of(cls, af: _Array, ag: _Array) -> _Apse:
        real_af, real_ag = np.real(af), np.real(ag)
        return cls(np.hypot(real_af, real_ag), np.arctan2(real_ag, real_af))


def _eccentric_longitude(
    lam: _Array, af: _Array, ag: _Array, apse: _Apse | None = None
) -> _Array:
    """The eccentric longitude ``F``: the solution of Kepler's equation.

    ``apse`` is ``_Apse.of(af, ag)`` if already known.

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
    if apse is None:
        apse = _Apse.of(af, ag)
    eccentric_longitude = lam + 0.85 * apse.eccentricity * np.sign(
        np.sin(np.real(lam) - apse.perigee_longitude)
    )
    # Promote to complex when the elements are (a complex-step derivative).
    eccentric_longitude = eccentric_longitude + 0.0 * af
    for _ in range(KEPLER_MAX_ITERATIONS):
        cos_f, sin_f = np.cos(eccentric_longitude), np.sin(eccentric_longitude)
        residual = eccentric_longitude + ag * cos_f - af * sin_f - lam
        step = residual / (1.0 - ag * sin_f - af * cos_f)
        eccentric_longitude = eccentric_longitude - step
        converged = (np.abs(step) < KEPLER_TOLERANCE) | (
            np.abs(residual) <= KEPLER_RESIDUAL_TOLERANCE
        )
        if np.all(converged):
            return eccentric_longitude
    raise GeometryError(
        "Kepler's equation did not converge in "
        f"{KEPLER_MAX_ITERATIONS} iterations; the elements are not a "
        "valid elliptic orbit"
    )


def _state(
    n: _Array,
    a: _Array,
    af: _Array,
    ag: _Array,
    shape: _Shape[_Array],
    eccentric_longitude: _Array,
    f: _Array,
    g: _Array,
) -> _Array:
    """Cartesian state from the eccentric longitude and the orbit's constants."""
    cos_f, sin_f = np.cos(eccentric_longitude), np.sin(eccentric_longitude)
    big_x = a * (shape.c_ag * cos_f + shape.cross * sin_f - af)
    big_y = a * (shape.c_af * sin_f + shape.cross * cos_f - ag)
    radius = a * (1.0 - af * cos_f - ag * sin_f)
    rate = n * a * a / radius
    big_xd = rate * (shape.cross * cos_f - shape.c_ag * sin_f)
    big_yd = rate * (shape.c_af * cos_f - shape.cross * sin_f)
    position = big_x[..., None] * f + big_y[..., None] * g
    velocity = big_xd[..., None] * f + big_yd[..., None] * g
    return np.concatenate([position, velocity], -1)  # pyright: ignore[reportReturnType]


@dataclass(frozen=True, eq=False, slots=True)
class Orbits:
    """Real elliptic orbits, prepared for evaluation at many times.

    Everything but the mean longitude is constant along two-body motion, so
    it is computed once (:meth:`from_elements`); :meth:`cartesian` then only
    solves Kepler's equation. The states equal
    ``to_cartesian(propagate(elements, t), fr)``.

    Every field has one row per orbit.
    """

    elements: FloatArray
    """Real equinoctial elements at TCA, shape ``(m, 6)``."""
    n: FloatArray
    """Mean motion [rad/s]."""
    af: FloatArray
    """Equinoctial ``af``."""
    ag: FloatArray
    """Equinoctial ``ag``."""
    a: FloatArray
    """Semi-major axis [m]."""
    shape: _Shape[FloatArray]
    """Coefficients of the in-plane state."""
    apse: _Apse
    """Eccentricity and longitude of perigee, for the Kepler start."""
    f: FloatArray
    """First in-plane basis vector, ``(m, 3)``."""
    g: FloatArray
    """Second in-plane basis vector, ``(m, 3)``."""

    @classmethod
    def from_elements(
        cls, elements: FloatArray, fr: RetrogradeFactor, mu: float = EARTH_MU
    ) -> Orbits:
        """The orbits of equinoctial elements.

        Parameters
        ----------
        elements
            Real equinoctial elements at TCA, shape ``(m, 6)``.
        fr
            Retrograde factor of every orbit: ``+1`` for prograde orbits, ``-1`` for
            retrograde ones (see :func:`retrograde_factor`).
        mu
            Gravitational parameter [m^3/s^2].
        """
        elements = np.asarray(elements, dtype=np.float64)
        n, af, ag, chi, psi = (elements[:, k] for k in range(5))
        f, g = _basis(chi, psi, fr)
        return cls(
            elements,
            n,
            af,
            ag,
            semi_major_axis(n, mu),
            _Shape.of(af, ag),
            _Apse.of(af, ag),
            f,
            g,
        )

    def subset(self, rows: slice | IndexArray) -> Orbits:
        """The orbits ``rows``, with their constants taken, not recomputed.

        Parameters
        ----------
        rows
            The orbits kept: a slice or an array of indices.
        """
        return Orbits(
            self.elements[rows],
            self.n[rows],
            self.af[rows],
            self.ag[rows],
            self.a[rows],
            take(self.shape, rows),
            take(self.apse, rows),
            self.f[rows],
            self.g[rows],
        )

    def cartesian(
        self, t: float | FloatArray, rows: slice | IndexArray = slice(None)
    ) -> FloatArray:
        """States ``(k, 6)`` of the orbits ``rows`` at ``t`` [s from TCA].

        Parameters
        ----------
        t
            Time [s] from TCA, the epoch of the elements: one, or one per orbit of
            ``rows``.
        rows
            The orbits to propagate: a slice or an array of indices; all by default.
        """
        n, af, ag = self.n[rows], self.af[rows], self.ag[rows]
        lam = self.elements[rows, 5] + n * np.asarray(t)
        state: FloatArray = _state(
            n,
            self.a[rows],
            af,
            ag,
            take(self.shape, rows),
            _eccentric_longitude(lam, af, ag, take(self.apse, rows)),
            self.f[rows],
            self.g[rows],
        )
        return state


def propagate(elements: _Array, dt: npt.ArrayLike) -> _Array:
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
    return propagated  # pyright: ignore[reportReturnType]


def wrap_angle(angle: _Real) -> _Real:
    """Angle difference wrapped to ``[-pi, pi)`` (real input only).

    Parameters
    ----------
    angle
        An angle or angle difference [rad], a number or a real array.
    """
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def complex_step(
    function: Callable[[ComplexArray], npt.NDArray[Any]],
    x: npt.ArrayLike,
    step: float = COMPLEX_STEP,
) -> tuple[FloatArray, FloatArray]:
    """Value and Jacobian of a real-analytic vector function by the complex step.

    ``J[:, k] = Im f(x + i h e_k) / h`` has no subtractive cancellation, so it
    is accurate to machine precision for any tiny ``h``; the value ``f(x)`` is
    the real part of the same evaluation (exact to ``O(h^2)``).

    Parameters
    ----------
    function
        Maps complex arrays ``(..., n)`` along the last axis to ``(..., k)``
        (every conversion in this module does). It is called once, with the
        ``n`` perturbed copies of each point along the second-to-last axis:
        shape ``(n, n)`` for one point, ``(m, n, n)`` for ``m``.
    x
        Point of evaluation (real), shape ``(n,)``, or ``m`` of them ``(m, n)``.
    step
        Imaginary step ``h``.

    Returns
    -------
    tuple
        Values ``(k,)`` and Jacobians ``(k, n)``, or ``(m, k)`` and
        ``(m, k, n)`` for ``m`` points.

    Raises
    ------
    TypeError
        If ``function`` returns a real array (its Jacobian would be zero).
    """
    point = np.asarray(x, dtype=np.float64)
    value = function(point[..., None, :] + 1j * step * np.eye(point.shape[-1]))
    if not np.iscomplexobj(value):
        raise TypeError(
            "complex_step needs a function that keeps complex input "
            "complex; it returned a real array, which would give a zero Jacobian"
        )
    real: FloatArray = np.real(value[..., 0, :])
    jacobian: FloatArray = np.swapaxes(np.imag(value), -1, -2) / step
    return real, jacobian
