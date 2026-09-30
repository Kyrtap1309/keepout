"""keepout — spacecraft conjunction assessment.

The core types and the Pc entry point are available from the package itself;
CDM input is in :mod:`keepout.cdm`, the individual Pc methods in
:mod:`keepout.pc`.
"""

from importlib.metadata import version

from keepout.conjunction import Conjunction
from keepout.covariance import Covariance, make_covariance
from keepout.objects import ObjectState
from keepout.pc import PcResult, compute_pc
from keepout.state import RefFrame, StateVector

__version__ = version("keepout")

__all__ = [
    "Conjunction",
    "Covariance",
    "ObjectState",
    "PcResult",
    "RefFrame",
    "StateVector",
    "__version__",
    "compute_pc",
    "make_covariance",
]
