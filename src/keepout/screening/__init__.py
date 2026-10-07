"""Screening: finding the close approaches of primaries with a catalog.

:class:`ScreeningVolume` is the region around a primary inside which an
approach is reported: an ellipsoid along the primary's RTN axes (a sphere by
default, 5 km as CelesTrak's SOCRATES), with presets from the USSPACECOM
Spaceflight Safety Handbook (chosen per primary's orbit regime) and NASA's
CA Best Practices Handbook. :func:`apogee_perigee_filter` rules out the
pairs whose distances from the Earth's centre never come within a primary's
volume, before anything is propagated.
"""

from keepout.screening.filters import apogee_perigee_filter
from keepout.screening.volume import (
    DEFAULT_VOLUME,
    SFS_HANDBOOK_VOLUMES,
    ScreeningVolume,
    SfsRegime,
    VolumeRule,
    sfs_regime,
    volume_for,
)

__all__ = [
    "DEFAULT_VOLUME",
    "SFS_HANDBOOK_VOLUMES",
    "ScreeningVolume",
    "SfsRegime",
    "VolumeRule",
    "apogee_perigee_filter",
    "sfs_regime",
    "volume_for",
]
