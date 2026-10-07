"""Screening volumes: the region around a primary where approaches are reported.

A :class:`ScreeningVolume` is an ellipsoid centred on the primary with its
semi-axes along the primary's radial, transverse and normal (RTN)
directions; a sphere is the case of three equal semi-axes. A secondary is
inside when its position relative to the primary, in the primary's RTN
frame (:func:`keepout.frames.relative_position_rtn`), satisfies
``(R/a)^2 + (T/b)^2 + (N/c)^2 <= 1``.

Presets:

* :data:`DEFAULT_VOLUME`: a 5 km sphere, the threshold of CelesTrak's
  SOCRATES reports.
* :meth:`ScreeningVolume.nasa_gp_data`: a 10 km sphere, the miss distance
  NASA finds necessary on GP (SGP4) data.
* :meth:`ScreeningVolume.sfs_handbook`: the volume USSPACECOM screens the
  primary's orbit regime with (:class:`SfsRegime`).

Where the volume depends on the primary, screening takes a
:data:`VolumeRule` (such as ``ScreeningVolume.sfs_handbook`` itself) and
resolves it per primary with :func:`volume_for`.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, TypeAlias

import numpy as np
import numpy.typing as npt

from keepout._types import BoolArray, FloatArray, read_only, require_positive
from keepout.frames import relative_position_rtn
from keepout.omm import Omm

__all__ = [
    "DEFAULT_VOLUME",
    "NASA_GP_DATA_RADIUS",
    "SFS_DEEP_SPACE_INCLINATION",
    "SFS_DEEP_SPACE_PERIOD",
    "SFS_ECCENTRICITY",
    "SFS_HANDBOOK_VOLUMES",
    "SFS_LEO_PERIGEES",
    "SOCRATES_RADIUS",
    "ScreeningVolume",
    "SfsRegime",
    "VolumeRule",
    "sfs_regime",
    "volume_for",
]

SOCRATES_RADIUS: Final = 5_000.0
"""Radius [m] of the default volume: CelesTrak's SOCRATES reports every
approach closer than 5 km."""

NASA_GP_DATA_RADIUS: Final = 10_000.0
"""Radius [m] of :meth:`ScreeningVolume.nasa_gp_data`."""


@dataclass(frozen=True)
class ScreeningVolume:
    """An ellipsoid around the primary, with semi-axes along its RTN axes.

    Parameters
    ----------
    radial
        Semi-axis along the primary's radial direction R [m].
    transverse
        Semi-axis along the transverse (in-track) direction T [m].
    normal
        Semi-axis along the orbit normal (cross-track) N [m].
    label
        Where the volume comes from, for reports; not part of equality.

    Raises
    ------
    ValueError
        If a semi-axis is not finite and positive.
    """

    radial: float
    transverse: float
    normal: float
    label: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        for name in ("radial", "transverse", "normal"):
            value = float(getattr(self, name))
            require_positive(value, name)
            object.__setattr__(self, name, value)

    @classmethod
    def sphere(cls, radius: float, *, label: str = "") -> ScreeningVolume:
        """A sphere of ``radius`` [m] around the primary."""
        return cls(radius, radius, radius, label)

    @classmethod
    def nasa_gp_data(cls) -> ScreeningVolume:
        """A 10 km sphere: the miss distance needed on GP (SGP4) data.

        NASA's analysis of screenings on GP data (TLE/OMM propagated with
        SGP4) against screenings on precision orbits finds that a miss
        distance threshold of about 10 km is needed to catch 95 % of the
        high-Pc events, because of the errors of GP data.

        References
        ----------
        NASA Spacecraft Conjunction Assessment and Collision Avoidance Best
        Practices Handbook, NASA/SP-20205011318 Rev. 2, 2026, Appendix E.
        """
        return cls.sphere(
            NASA_GP_DATA_RADIUS,
            label="10 km sphere (NASA CA Handbook Rev. 2, App. E, GP data)",
        )

    @classmethod
    def sfs_handbook(cls, primary: Omm) -> ScreeningVolume:
        """The volume USSPACECOM screens the primary's orbit regime with.

        The regime is :func:`sfs_regime` of the primary's mean elements; the
        volume is :data:`SFS_HANDBOOK_VOLUMES` of it: Table 3 ("HAC screening
        volumes") of the Spaceflight Safety Handbook. The table gives radial,
        in-track and cross-track sizes without saying whether they are
        semi-axes or full extents; keepout reads them as semi-axes (the
        larger volume of the two readings, so no approach the handbook would
        report is missed). The table's LEO rows are covariance screenings,
        for which 18 SDS builds a joint volume from both objects'
        time-varying covariances; a GP catalog has no covariance, so keepout
        uses the rows' sizes as a fixed ellipsoid. A primary outside every
        regime of the table gets the deep-space volume, which the table
        labels "GEO/HEO/MEO".

        This method is itself a :data:`VolumeRule`: pass
        ``ScreeningVolume.sfs_handbook`` to screening to choose per primary.

        Parameters
        ----------
        primary
            Element set of the primary.

        Raises
        ------
        ValueError
            If SGP4 cannot be initialised from ``primary``
            (:meth:`keepout.omm.Omm.satrec`).

        References
        ----------
        USSPACECOM, Spaceflight Safety Handbook for Satellite Operators,
        version 1.7, April 2023, Table 3.
        """
        return SFS_HANDBOOK_VOLUMES[sfs_regime(primary)]

    @property
    def semi_axes(self) -> FloatArray:
        """Radial, transverse and normal semi-axes [m], read-only ``(3,)``."""
        axes = np.array([self.radial, self.transverse, self.normal])
        read_only(axes)
        return axes

    @property
    def bounding_radius(self) -> float:
        """Radius [m] of the smallest sphere around the primary that holds
        the volume: its largest semi-axis. Whatever the primary's attitude
        of RTN, no point farther than this is inside."""
        return max(self.radial, self.transverse, self.normal)

    @property
    def is_sphere(self) -> bool:
        """Whether the three semi-axes are equal."""
        return self.radial == self.transverse == self.normal

    def contains(self, offsets: npt.ArrayLike) -> BoolArray:
        """Whether relative positions lie in the volume (boundary included).

        Parameters
        ----------
        offsets
            Positions relative to the primary in its RTN frame [m], shape
            ``(..., 3)``.

        Returns
        -------
        BoolArray
            Shape ``(...)``; ``False`` where an offset is not finite.

        Raises
        ------
        ValueError
            If ``offsets`` does not have 3 components on its last axis.
        """
        array = np.asarray(offsets, dtype=np.float64)
        if array.ndim == 0 or array.shape[-1] != 3:
            raise ValueError("offsets must have 3 components on their last axis")
        scaled = array / self.semi_axes
        inside: BoolArray = np.sum(scaled * scaled, axis=-1) <= 1.0
        return inside

    def contains_position(
        self, r: npt.ArrayLike, v: npt.ArrayLike, r_other: npt.ArrayLike
    ) -> BoolArray:
        """Whether other objects lie in the volume around a primary.

        ``contains(relative_position_rtn(r, v, r_other))``: the inputs
        broadcast as in :func:`keepout.frames.relative_position_rtn`, so
        many pairs and times are tested at once.

        Parameters
        ----------
        r
            Inertial positions of the primary [m], shape ``(..., 3)``.
        v
            Inertial velocities of the primary [m/s], shape ``(..., 3)``.
        r_other
            Inertial positions of the other objects [m], ``(..., 3)``.

        Raises
        ------
        ValueError
            If the shapes are not ``(..., 3)`` or do not broadcast.
        GeometryError
            If a primary's position is zero or parallel to its velocity.
        """
        return self.contains(relative_position_rtn(r, v, r_other))


DEFAULT_VOLUME: Final = ScreeningVolume.sphere(
    SOCRATES_RADIUS, label="5 km sphere (as CelesTrak SOCRATES)"
)
"""The volume screening uses unless told otherwise: a 5 km sphere."""

VolumeRule: TypeAlias = Callable[[Omm], ScreeningVolume]
"""A screening volume chosen from the primary's element set, e.g.
:meth:`ScreeningVolume.sfs_handbook`."""


def volume_for(volume: ScreeningVolume | VolumeRule, primary: Omm) -> ScreeningVolume:
    """The volume to screen ``primary`` with.

    Parameters
    ----------
    volume
        A volume for every primary, or a rule choosing one per primary.
    primary
        Element set of the primary.

    Raises
    ------
    TypeError
        If the rule does not return a :class:`ScreeningVolume`.
    """
    if isinstance(volume, ScreeningVolume):
        return volume
    chosen = volume(primary)
    if not isinstance(chosen, ScreeningVolume):
        raise TypeError(
            f"a volume rule must return a ScreeningVolume, got {type(chosen).__name__}"
        )
    return chosen


class SfsRegime(enum.Enum):
    """Orbit regimes of the Spaceflight Safety Handbook's Table 3.

    The LEO regimes need an eccentricity below :data:`SFS_ECCENTRICITY`
    and are bounded by the perigee altitude (:data:`SFS_LEO_PERIGEES`).
    """

    LEO_1 = "LEO 1"
    """Perigee altitude up to 500 km."""
    LEO_2 = "LEO 2"
    """Perigee altitude above 500 km, up to 750 km."""
    LEO_3 = "LEO 3"
    """Perigee altitude above 750 km, up to 1200 km."""
    LEO_4 = "LEO 4"
    """Perigee altitude above 1200 km, up to 2000 km."""
    DEEP_SPACE = "Deep Space"
    """Period between 1300 and 1800 min, eccentricity below 0.25 and
    inclination below 35 degrees (geosynchronous)."""
    OUTSIDE = "outside Table 3"
    """None of the above: e.g. MEO, HEO, inclined geosynchronous orbits."""


SFS_ECCENTRICITY: Final = 0.25
"""Eccentricity below which an orbit can be in a LEO or the deep-space
regime of Table 3."""

SFS_LEO_PERIGEES: Final[Mapping[SfsRegime, float]] = MappingProxyType(
    {
        SfsRegime.LEO_1: 500e3,
        SfsRegime.LEO_2: 750e3,
        SfsRegime.LEO_3: 1200e3,
        SfsRegime.LEO_4: 2000e3,
    }
)
"""Highest perigee altitude [m] of each LEO regime, lowest regime first."""

SFS_DEEP_SPACE_PERIOD: Final = (1300.0 * 60.0, 1800.0 * 60.0)
"""Bounds [s] of the period of the deep-space regime (both excluded)."""

SFS_DEEP_SPACE_INCLINATION: Final = math.radians(35.0)
"""Inclination [rad] below which a geosynchronous orbit is in the
deep-space regime."""

_SFS: Final = "SFS Handbook v1.7 Table 3"
_DEEP_SPACE_AXIS: Final = 10e3

SFS_HANDBOOK_VOLUMES: Final[Mapping[SfsRegime, ScreeningVolume]] = MappingProxyType(
    {
        SfsRegime.LEO_1: ScreeningVolume(400.0, 44e3, 51e3, f"{_SFS}, LEO 1"),
        SfsRegime.LEO_2: ScreeningVolume(400.0, 25e3, 25e3, f"{_SFS}, LEO 2"),
        SfsRegime.LEO_3: ScreeningVolume(400.0, 12e3, 12e3, f"{_SFS}, LEO 3"),
        SfsRegime.LEO_4: ScreeningVolume(400.0, 2e3, 2e3, f"{_SFS}, LEO 4"),
        SfsRegime.DEEP_SPACE: ScreeningVolume.sphere(
            _DEEP_SPACE_AXIS, label=f"{_SFS}, Deep Space"
        ),
        SfsRegime.OUTSIDE: ScreeningVolume.sphere(
            _DEEP_SPACE_AXIS,
            label=f"{_SFS}, Deep Space (orbit outside the table's regimes)",
        ),
    }
)
"""Volume of each regime of Table 3 (radial x in-track x cross-track
semi-axes: LEO 1 0.4 x 44 x 51 km, LEO 2 0.4 x 25 x 25 km, LEO 3 0.4 x 12
x 12 km, LEO 4 0.4 x 2 x 2 km, Deep Space 10 x 10 x 10 km). Orbits outside
the table get the deep-space volume."""


def sfs_regime(primary: Omm) -> SfsRegime:
    """The regime of Table 3 an element set belongs to.

    Perigee altitude and period are those SGP4 itself derives from the
    elements: its un-Kozai'd (Brouwer) semi-major axis and mean motion, and
    the WGS 72 constants it uses (``Satrec.altp``, ``Satrec.a``,
    ``Satrec.xke``; ``Satrec.no`` is the element set's Kozai mean motion,
    not SGP4's). A two-body reading of the Kozai mean motion would give
    perigees a few kilometres off (-3 to +6 km in LEO on the public catalog
    of October 2026) and periods a few seconds off (3 s at GEO), enough to
    move some orbits across a regime boundary.

    Parameters
    ----------
    primary
        Element set of the primary.

    Raises
    ------
    ValueError
        If SGP4 cannot be initialised from ``primary``
        (:meth:`keepout.omm.Omm.satrec`).
    """
    if primary.eccentricity >= SFS_ECCENTRICITY:
        return SfsRegime.OUTSIDE
    satrec = primary.satrec()
    perigee = satrec.altp * satrec.radiusearthkm * 1e3
    for regime, highest in SFS_LEO_PERIGEES.items():
        if perigee <= highest:
            return regime
    # Un-Kozai'd mean motion xke / a**1.5 [rad/min], a in Earth radii.
    period = 2.0 * math.pi * satrec.a**1.5 / satrec.xke * 60.0
    lowest, highest = SFS_DEEP_SPACE_PERIOD
    if lowest < period < highest and primary.inclination < SFS_DEEP_SPACE_INCLINATION:
        return SfsRegime.DEEP_SPACE
    return SfsRegime.OUTSIDE
