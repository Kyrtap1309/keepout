"""keepout — spacecraft conjunction assessment.

The core types, the Pc entry point and the exception hierarchy are available
from the package itself; CDM input is in :mod:`keepout.cdm`, the individual Pc
methods in :mod:`keepout.pc`.
"""

from importlib.metadata import version

from keepout.conjunction import Conjunction
from keepout.covariance import Covariance, make_covariance
from keepout.errors import (
    CaraFormatError,
    CdmParseError,
    CdmWriteError,
    ChecksumMismatch,
    CovarianceError,
    GeometryError,
    KeepoutError,
    PcConvergenceError,
    SourceUnavailable,
    UnsupportedFrame,
)
from keepout.objects import ObjectState
from keepout.pc import Method, MethodChoice, PcResult, compute_pc
from keepout.state import RefFrame, StateVector

__version__ = version("keepout")

__all__ = [
    "CaraFormatError",
    "CdmParseError",
    "CdmWriteError",
    "ChecksumMismatch",
    "Conjunction",
    "Covariance",
    "CovarianceError",
    "GeometryError",
    "KeepoutError",
    "Method",
    "MethodChoice",
    "ObjectState",
    "PcConvergenceError",
    "PcResult",
    "RefFrame",
    "SourceUnavailable",
    "StateVector",
    "UnsupportedFrame",
    "__version__",
    "compute_pc",
    "make_covariance",
]
