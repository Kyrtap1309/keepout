"""Probability of collision.

Use :func:`compute_pc` on a :class:`~keepout.conjunction.Conjunction`. The
2-D and maximum-Pc functions (:func:`foster`, :func:`elrod`, :func:`frisbee`,
:func:`dilution`, :func:`alfano`) work on an encounter plane; the 3-D and
Monte Carlo functions (:func:`hall3d`, :func:`montecarlo`) on a conjunction.
"""

from keepout.pc.api import MethodChoice, compute_pc
from keepout.pc.elrod import elrod
from keepout.pc.foster import foster
from keepout.pc.hall3d import hall3d
from keepout.pc.maxpc import alfano, dilution, frisbee
from keepout.pc.montecarlo import montecarlo
from keepout.pc.result import Method, PcResult

__all__ = [
    "Method",
    "MethodChoice",
    "PcResult",
    "alfano",
    "compute_pc",
    "dilution",
    "elrod",
    "foster",
    "frisbee",
    "hall3d",
    "montecarlo",
]
