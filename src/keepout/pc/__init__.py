"""Probability of collision.

Use :func:`compute_pc` on a :class:`~keepout.conjunction.Conjunction`; the
method functions :func:`foster` and :func:`elrod` work on an encounter plane.
"""

from keepout.pc.api import compute_pc
from keepout.pc.elrod import elrod
from keepout.pc.foster import foster
from keepout.pc.result import PcResult

__all__ = ["PcResult", "compute_pc", "elrod", "foster"]
