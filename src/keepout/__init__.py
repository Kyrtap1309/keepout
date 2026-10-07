"""keepout — spacecraft conjunction assessment.

The core types, the Pc entry point, the event assessment, screening and the
exception hierarchy are available from the package itself; CDM input is in
:mod:`keepout.cdm`, the individual Pc methods in :mod:`keepout.pc`, risk
levels, trends and decisions in :mod:`keepout.risk`, element sets in
:mod:`keepout.omm`, screening volumes and the search in
:mod:`keepout.screening`.
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
    HttpStatusError,
    KeepoutError,
    KelvinsFormatError,
    OmmParseError,
    PcConvergenceError,
    SourceFormatError,
    SourceUnavailable,
    UnsupportedFrame,
)
from keepout.objects import ObjectState
from keepout.pc import Method, MethodChoice, PcResult, compute_pc
from keepout.risk import ConjunctionEvent, PcObservation, assess
from keepout.screening import CloseApproach, ScreeningResult, screen
from keepout.state import RefFrame, StateVector

__version__ = version("keepout")

__all__ = [
    "CaraFormatError",
    "CdmParseError",
    "CdmWriteError",
    "ChecksumMismatch",
    "CloseApproach",
    "Conjunction",
    "ConjunctionEvent",
    "Covariance",
    "CovarianceError",
    "GeometryError",
    "HttpStatusError",
    "KeepoutError",
    "KelvinsFormatError",
    "Method",
    "MethodChoice",
    "ObjectState",
    "OmmParseError",
    "PcConvergenceError",
    "PcObservation",
    "PcResult",
    "RefFrame",
    "ScreeningResult",
    "SourceFormatError",
    "SourceUnavailable",
    "StateVector",
    "UnsupportedFrame",
    "__version__",
    "assess",
    "compute_pc",
    "make_covariance",
    "screen",
]
