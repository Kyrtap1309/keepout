"""Filters that rule out pairs before anything is propagated.

Screening compares every primary with every object of a catalog over a
window of days. Most pairs can never come close: their orbits are at
different altitudes. The apogee/perigee filter (Hoots, Crawford and
Roehrich 1984) finds them from bounds on each object's distance from the
Earth's centre over the window, which the propagator gives without
propagating (:meth:`keepout.propagation.Propagator.radius_bounds`): two
objects whose radii stay in ``[lo1, hi1]`` and ``[lo2, hi2]`` are always at
least ``max(lo2 - hi1, lo1 - hi2)`` apart, because ``|r2 - r1| >= | |r2| -
|r1| |``. A pair whose gap exceeds the bounding radius of the primary's
volume (:attr:`~keepout.screening.volume.ScreeningVolume.bounding_radius`)
cannot come inside it, whatever the volume's orientation. The filter is
conservative as far as the radius bounds are: it never drops a pair that
comes inside the volume.

There is no orbit-path (geometric) filter. Hoots' second filter compares
the two orbits as curves in space (their closest distance, near the line of
nodes), which holds only while the orbits keep their orientation. Over a
window of days they turn: in LEO the J2 nodal regression of two orbits
differs by several degrees a day, and their perigees rotate as fast.
Without a pad for that motion the filter drops real approaches: in the
public catalog of October 2026, for 186 of the 267 fast (1 km/s or more)
approaches closer than 5 km that SOCRATES reported for the following 7 days
(of those keepout can compare), the two orbits at the start of the window
are more than 5 km apart, up to 75 km. A pad covering the motion (the
radius at the line of nodes changes by up to about ``2 a (e1 + e2)`` as the
perigees turn, plus SGP4's periodic terms) leaves the filter little to add,
because two near-circular orbits (92 % of the near-Earth catalog has e <
0.01) come about as close as their altitudes, which the apogee/perigee
filter already tests. Pairs that pass the apogee/perigee filter go straight
to :class:`~keepout.screening.tca.ApproachSearch` (its speed and chord
sieves).

References
----------
F. R. Hoots, L. L. Crawford and R. L. Roehrich, "An analytic method to
determine future close approaches between satellites", Celestial Mechanics
33, 143-158 (1984).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from keepout.screening.volume import ScreeningVolume, per_primary

if TYPE_CHECKING:
    from collections.abc import Sequence

    from keepout._types import BoolArray
    from keepout.propagation.base import RadiusBounds

__all__ = ["apogee_perigee_filter"]


def apogee_perigee_filter(
    primaries: RadiusBounds,
    catalog: RadiusBounds,
    volumes: ScreeningVolume | Sequence[ScreeningVolume],
) -> BoolArray:
    """Which catalog objects can come inside each primary's volume.

    Parameters
    ----------
    primaries
        Radius bounds of the P primaries over the window.
    catalog
        Radius bounds of the S catalog objects over the same window.
    volumes
        The volume of every primary, or one volume per primary (in order),
        e.g. from :func:`~keepout.screening.volume.volume_for`. Each primary
        is padded with its own volume's bounding radius.

    Returns
    -------
    BoolArray
        Shape ``(P, S)``: ``True`` where the pair's radius bounds come within
        the primary's bounding radius of each other, i.e. where the pair
        must be screened; ``False`` where it cannot come inside the volume.
        A primary that is also in the catalog keeps itself.

    Raises
    ------
    ValueError
        If ``volumes`` is a sequence of another length than ``primaries``.
    """
    bounding_radii = np.array(
        [volume.bounding_radius for volume in per_primary(volumes, len(primaries))]
    )
    highest = (primaries.highest + bounding_radii)[:, None]
    lowest = (primaries.lowest - bounding_radii)[:, None]
    keep: BoolArray = (catalog.lowest <= highest) & (catalog.highest >= lowest)
    return keep
