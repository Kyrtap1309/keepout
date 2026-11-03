"""Probability of collision.

Use :func:`compute_pc` on a :class:`~keepout.conjunction.Conjunction`, or
:func:`computation` for a method with options to run on many. The 2-D and
maximum-Pc functions (:func:`foster`, :func:`elrod`, :func:`frisbee`,
:func:`dilution`, :func:`alfano`) work on an encounter plane; the 3-D and
Monte Carlo functions (:func:`hall3d`, :func:`montecarlo`) on a conjunction.
:data:`METHODS` describes every method.
"""

from keepout.pc.api import (
    AUTO,
    METHODS,
    ElrodOptions,
    FosterOptions,
    Hall3dOptions,
    MethodChoice,
    MethodKind,
    MethodSpec,
    MonteCarloOptions,
    PcComputation,
    computation,
    compute_pc,
    require_options,
)
from keepout.pc.elrod import elrod
from keepout.pc.foster import foster
from keepout.pc.hall3d import hall3d
from keepout.pc.maxpc import alfano, dilution, frisbee
from keepout.pc.montecarlo import montecarlo
from keepout.pc.result import Method, PcResult, ResultKind

__all__ = [
    "AUTO",
    "METHODS",
    "ElrodOptions",
    "FosterOptions",
    "Hall3dOptions",
    "Method",
    "MethodChoice",
    "MethodKind",
    "MethodSpec",
    "MonteCarloOptions",
    "PcComputation",
    "PcResult",
    "ResultKind",
    "alfano",
    "computation",
    "compute_pc",
    "dilution",
    "elrod",
    "foster",
    "frisbee",
    "hall3d",
    "montecarlo",
    "require_options",
]
