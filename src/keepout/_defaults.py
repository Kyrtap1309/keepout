"""Library defaults the command line's help states, without importing the
numerical libraries.

Each is defined here once and re-exported where it belongs (named in each
docstring); the command line formats its help from these, so the help
cannot drift from the library and ``keepout --help`` need not load NumPy.
"""

from __future__ import annotations

from typing import Final

from keepout.units import DAY, HOUR, MINUTE

__all__ = [
    "CDM_PUBLIC_INTERVAL",
    "CELESTRAK_TTL",
    "DECISION_POINT",
    "GP_INTERVAL",
    "MIN_BURN_LEAD",
    "SCREENING_DURATION",
    "SOCRATES_RADIUS",
    "TARGET_PC",
    "TCA_TOLERANCE",
]

TCA_TOLERANCE: Final = 10 * MINUTE
"""Largest spread [s] of the TCAs of one event's CDMs
(:data:`keepout.events.DEFAULT_TCA_TOLERANCE`)."""

DECISION_POINT: Final = 2 * DAY
"""Time before TCA [s] from which a manoeuvre is decided
(:attr:`keepout.risk.DecisionConfig.decision_point`)."""

TARGET_PC: Final = 1e-5
"""Pc below which a burn is an option (:data:`keepout.maneuver.TARGET_PC`)."""

MIN_BURN_LEAD: Final = 6 * HOUR
"""Latest time before TCA [s] of a planned burn
(:data:`keepout.maneuver.MIN_BURN_LEAD`)."""

SCREENING_DURATION: Final = 7 * DAY
"""Length of a screening window [s]
(:data:`keepout.screening.DEFAULT_DURATION`)."""

SOCRATES_RADIUS: Final = 5_000.0
"""Radius [m] of the default screening volume
(:data:`keepout.screening.volume.SOCRATES_RADIUS`)."""

GP_INTERVAL: Final = HOUR
"""Least seconds between two Space-Track ``gp`` queries
(:data:`keepout.sources.spacetrack.CLASS_INTERVALS`)."""

CDM_PUBLIC_INTERVAL: Final = 8 * HOUR
"""Least seconds between two Space-Track ``cdm_public`` queries
(:data:`keepout.sources.spacetrack.CLASS_INTERVALS`)."""

CELESTRAK_TTL: Final = 2 * HOUR
"""Seconds a CelesTrak GP download is reused
(:data:`keepout.sources.celestrak.TTL`)."""
