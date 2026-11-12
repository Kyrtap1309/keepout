"""Avoidance manoeuvres: what a burn does at the time of closest approach.

:func:`plan_maneuver` turns a conjunction event into a :class:`ManeuverPlan`:
the Pc of a grid of burns (time before TCA by signed along-track Δv) and the
smallest burn per direction that meets a target. :func:`evaluate_burn`
answers for one burn, of a plan or of the :class:`ManeuverContext` of
:func:`prepare_maneuver`, which skips the grid. :func:`burn_effects` gives
the change of an object's state at TCA for each burn of a grid, as a
:class:`BurnEffects`, and
:func:`closest_approach_shift` how far the closest approach moves in time.
:func:`rescreen` screens a burn against a catalog and returns a
:class:`RescreenResult` listing the approaches it brings
(:class:`NewApproach`). :func:`judge` says whether a burn can be recommended
(:class:`BurnVerdict`). :func:`plan_event` does all of it with an
operator's settings and returns a :class:`ManeuverOutcome`: the burns
(:class:`PlannedBurn`), their re-screens and the grid
(:class:`ManeuverGrid`).
"""

from keepout.maneuver._differential import (
    BurnEffects,
    burn_effects,
    closest_approach_shift,
)
from keepout.maneuver.outcome import (
    ManeuverGrid,
    ManeuverOutcome,
    PlannedBurn,
    plan_event,
)
from keepout.maneuver.planner import (
    DEFAULT_BURN_GRID,
    MAX_CONFIRMATIONS,
    MIN_BURN_LEAD,
    TARGET_PC,
    BurnDirection,
    BurnGrid,
    ManeuverContext,
    ManeuverOption,
    ManeuverPlan,
    evaluate_burn,
    plan_maneuver,
    prepare_maneuver,
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
    "ManeuverContext",
    "ManeuverGrid",
    "ManeuverOption",
    "ManeuverOutcome",
    "ManeuverPlan",
    "NewApproach",
    "PlannedBurn",
    "RescreenResult",
    "burn_effects",
    "closest_approach_shift",
    "evaluate_burn",
    "judge",
    "plan_event",
    "plan_maneuver",
    "prepare_maneuver",
    "rescreen",
]
