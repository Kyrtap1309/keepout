"""The encounter-plane picture: the miss, the covariance ellipses and the
hard-body circle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from keepout._linalg import sigma_axes, symmetric_part
from keepout._types import FloatArray, read_only
from keepout._validation import require_at_least, require_positive

if TYPE_CHECKING:
    from keepout.encounter import EncounterPlane

__all__ = ["SIGMAS", "EncounterFigure", "encounter_figure"]

SIGMAS: Final = (1.0, 2.0, 3.0)
"""The sigma levels of the covariance ellipses."""


@dataclass(frozen=True, eq=False, slots=True)
class EncounterFigure:
    """What an encounter-plane picture draws, in the plane axes ``(x, z)`` [m].

    Parameters
    ----------
    miss
        The miss vector: the secondary relative to the primary.
    ellipses
        The combined covariance's 1, 2 and 3 sigma ellipses around the miss,
        each a closed curve of shape ``(n, 2)``.
    hbr_circle
        The combined hard-body circle around the primary (the origin), a
        closed curve of shape ``(n, 2)``.
    """

    miss: tuple[float, float]
    ellipses: tuple[FloatArray, ...]
    hbr_circle: FloatArray


def encounter_figure(
    plane: EncounterPlane, hbr: float, *, points: int = 181
) -> EncounterFigure:
    """The encounter-plane picture of a conjunction.

    Each ellipse is ``k·sqrt(λ)`` along each eigenvector of the plane's
    2x2 covariance (eigenvalue ``λ``) for ``k`` in :data:`SIGMAS`; an unknown
    covariance (all zeros) gives ellipses shrunk to the miss.

    Parameters
    ----------
    plane
        The conjunction in the encounter plane.
    hbr
        Combined hard-body radius [m].
    points
        Points of each curve, the first repeated as the last.

    Returns
    -------
    EncounterFigure
        The miss, the ellipses and the hard-body circle.

    Raises
    ------
    ValueError
        If ``hbr`` is not finite and positive, or ``points`` is below 3.
    """
    require_positive(hbr, "hbr")
    require_at_least(points, 3, "points")
    angles = np.linspace(0.0, 2.0 * np.pi, points)
    unit = np.column_stack((np.cos(angles), np.sin(angles)))
    unit[-1] = unit[0]
    axes = sigma_axes(symmetric_part(plane.covariance))
    miss = np.asarray(plane.miss_vector, dtype=np.float64)
    ellipses = tuple(miss + k * (unit @ axes) for k in SIGMAS)
    circle = hbr * unit
    read_only(*ellipses, circle)
    return EncounterFigure((float(miss[0]), float(miss[1])), ellipses, circle)
