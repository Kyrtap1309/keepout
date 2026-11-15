"""The 3-D relative geometry at TCA, in the primary's RTN frame: the
secondary's straight-line motion, the covariance ellipsoid and the hard-body
sphere."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from keepout._linalg import sigma_axes, symmetric_part
from keepout._types import FloatArray, read_only
from keepout._validation import require_at_least, require_positive
from keepout.frames import rtn_rotation

if TYPE_CHECKING:
    from keepout.conjunction import Conjunction

__all__ = ["GeometryFigure", "geometry_figure"]

_Surface = tuple[FloatArray, FloatArray, FloatArray]

_SIGMA: Final = 3.0
"""The sigma level of the covariance ellipsoid."""


@dataclass(frozen=True, eq=False, slots=True)
class GeometryFigure:
    """What a 3-D geometry picture draws, in the primary's RTN axes [m].

    The primary is at the origin; every point is the secondary relative to
    it.

    Parameters
    ----------
    path
        The secondary's straight-line motion around TCA, shape ``(n, 3)``,
        at evenly spaced times from ``-span`` to ``span`` seconds.
    at_tca
        The secondary's position at TCA (the middle of ``path``).
    velocity
        The secondary's velocity relative to the primary at TCA [m/s], in
        the same axes.
    ellipsoid
        The combined covariance's 3 sigma ellipsoid around ``at_tca``: the
        surface's ``x``, ``y`` and ``z`` (R, T and N) as meshgrid arrays of
        shape ``(n, n)``.
    hbr_sphere
        The combined hard-body sphere around the primary (the origin), in
        the same form as ``ellipsoid``.
    axes
        The axis labels, R, T and N.
    """

    path: FloatArray
    at_tca: tuple[float, float, float]
    velocity: tuple[float, float, float]
    ellipsoid: _Surface
    hbr_sphere: _Surface
    axes: tuple[str, str, str] = ("R [m]", "T [m]", "N [m]")


def geometry_figure(
    conjunction: Conjunction, hbr: float, *, span: float = 2.0, points: int = 41
) -> GeometryFigure:
    """The 3-D relative geometry of a conjunction at TCA.

    Positions, velocity and covariance are rotated into the RTN frame of
    the primary's state at TCA (:func:`~keepout.frames.rtn_rotation`); the
    motion is linear (the relative velocity at TCA), as the encounter plane
    assumes. The ellipsoid is ``3·sqrt(λ)`` along each eigenvector of the
    combined 3x3 position covariance (eigenvalue ``λ``); an unknown
    covariance (all zeros) gives an ellipsoid shrunk to ``at_tca``.

    Parameters
    ----------
    conjunction
        Both objects' states at TCA, and their covariances where known.
    hbr
        Combined hard-body radius [m].
    span
        Half-length [s] of the path, before and after TCA.
    points
        Points of the path, and of each direction of the surfaces' grids;
        odd, so that TCA is one of them.

    Returns
    -------
    GeometryFigure
        The path, the relative velocity, the ellipsoid and the hard-body
        sphere.

    Raises
    ------
    ValueError
        If ``hbr`` or ``span`` is not finite and positive, or ``points`` is
        below 3 or even.
    GeometryError
        If the relative velocity is zero (the encounter plane is
        undefined), or the primary's position is zero or parallel to its
        velocity (its RTN frame is undefined).
    """
    require_positive(hbr, "hbr")
    require_positive(span, "span")
    require_at_least(points, 3, "points")
    if points % 2 == 0:
        raise ValueError(f"points must be odd, got {points}")
    plane = conjunction.plane
    primary = conjunction.primary.state
    rotation = rtn_rotation(primary.r, primary.v)
    position = rotation @ plane.relative_position
    velocity = rotation @ plane.relative_velocity
    covariance = symmetric_part(rotation @ plane.combined_covariance @ rotation.T)
    times = np.linspace(-span, span, points)
    path = position + times[:, np.newaxis] * velocity
    # TCA exactly, not position + 0.0 * velocity rounded.
    path[points // 2] = position
    unit = _unit_sphere(points)
    axes = sigma_axes(covariance, _SIGMA)
    ellipsoid = _components(position + unit @ axes)
    sphere = _components(hbr * unit)
    read_only(path)
    return GeometryFigure(
        path,
        (float(position[0]), float(position[1]), float(position[2])),
        (float(velocity[0]), float(velocity[1]), float(velocity[2])),
        ellipsoid,
        sphere,
    )


def _components(grid: FloatArray) -> _Surface:
    """A ``(n, n, 3)`` grid of points as read-only ``x``, ``y`` and ``z``
    arrays."""
    x, y, z = (np.array(grid[..., i]) for i in range(3))
    read_only(x, y, z)
    return x, y, z


def _unit_sphere(points: int) -> FloatArray:
    """The unit sphere as a ``(points, points, 3)`` grid: rows by polar angle
    from +z, columns by azimuth from +x, the first column repeated as the
    last."""
    polar, azimuth = np.meshgrid(
        np.linspace(0.0, np.pi, points),
        np.linspace(0.0, 2.0 * np.pi, points),
        indexing="ij",
    )
    unit: FloatArray = np.stack(
        (
            np.sin(polar) * np.cos(azimuth),
            np.sin(polar) * np.sin(azimuth),
            np.cos(polar),
        ),
        axis=-1,
    )
    unit[:, -1] = unit[:, 0]
    return unit
