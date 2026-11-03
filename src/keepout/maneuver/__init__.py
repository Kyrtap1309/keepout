"""Avoidance manoeuvres: what a burn does at the time of closest approach.

:func:`plan_maneuver` turns a conjunction event into a :class:`ManeuverPlan`:
the Pc of a grid of burns (time before TCA by signed along-track Δv) and the
smallest burn per direction that meets a target, and :func:`evaluate_burn`
answers for one burn. :func:`burn_effects` gives the change of an object's
state at TCA for each burn of a grid, as a :class:`BurnEffects`, and
:func:`closest_approach_shift` how far the closest approach moves in time.
:func:`rescreen` screens a burn against a catalog for a
:class:`RescreenResult`: the :class:`NewApproach` it brings, and
:func:`judge` says whether a burn can be recommended (:class:`BurnVerdict`).
"""

from keepout.maneuver._differential import (
    BurnEffects,
    burn_effects,
    closest_approach_shift,
)
from keepout.maneuver.planner import (
    DEFAULT_BURN_GRID,
    MAX_CONFIRMATIONS,
    MIN_BURN_LEAD,
    TARGET_PC,
    BurnDirection,
    BurnGrid,
    ManeuverOption,
    ManeuverPlan,
    evaluate_burn,
    plan_maneuver,
)
from keepout.maneuver.rescreen import (
    BurnVerdict,
    NewApproach,
    RescreenResult,
    judge,
    rescreen,
)

__all__ = [
    "DEFAULT_BURN_GRID",
    "MAX_CONFIRMATIONS",
    "MIN_BURN_LEAD",
    "TARGET_PC",
    "BurnDirection",
    "BurnEffects",
    "BurnGrid",
    "BurnVerdict",
    "ManeuverOption",
    "ManeuverPlan",
    "NewApproach",
    "RescreenResult",
    "burn_effects",
    "closest_approach_shift",
    "evaluate_burn",
    "judge",
    "plan_maneuver",
    "rescreen",
]
