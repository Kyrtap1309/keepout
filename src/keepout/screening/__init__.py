"""Screening: finding the close approaches of primaries with a catalog.

:class:`ScreeningVolume` is the region around a primary inside which an
approach is reported: an ellipsoid along the primary's RTN axes (a sphere by
default, 5 km as CelesTrak's SOCRATES), with presets from the USSPACECOM
Spaceflight Safety Handbook (chosen per primary's orbit regime) and NASA's
CA Best Practices Handbook. :func:`apogee_perigee_filter` rules out the
pairs whose distances from the Earth's centre never come within a primary's
volume, before anything is propagated. :class:`ApproachSearch` finds, on a
time grid (:func:`time_grid`), every local minimum of the range of the
remaining pairs inside the volumes, with its time of closest approach
(:class:`RangeMinimum`, :class:`SearchResult`). :func:`screen` does it all
for element sets: it returns a :class:`ScreeningResult` of close approaches
(:class:`CloseApproach`, in GCRF) and the element sets it left out
(:class:`SkippedObject`, with a :class:`SkipReason`).
"""

from keepout.screening.filters import apogee_perigee_filter
from keepout.screening.pipeline import DEFAULT_DURATION, MAX_AGE, MAX_LEAD, screen
from keepout.screening.result import (
    CloseApproach,
    ScreeningResult,
    ScreeningRole,
    SkippedObject,
    SkipReason,
)
from keepout.screening.tca import (
    ApproachSearch,
    ObjectFailure,
    RangeMinimum,
    SearchResult,
    time_grid,
)
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
    "DEFAULT_DURATION",
    "DEFAULT_VOLUME",
    "MAX_AGE",
    "MAX_LEAD",
    "SFS_HANDBOOK_VOLUMES",
    "ApproachSearch",
    "CloseApproach",
    "ObjectFailure",
    "RangeMinimum",
    "ScreeningResult",
    "ScreeningRole",
    "ScreeningVolume",
    "SearchResult",
    "SfsRegime",
    "SkipReason",
    "SkippedObject",
    "VolumeRule",
    "apogee_perigee_filter",
    "screen",
    "sfs_regime",
    "time_grid",
    "volume_for",
]
