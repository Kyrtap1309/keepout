"""keepout — spacecraft conjunction assessment.

The core types, the Pc entry point, the event assessment, screening (with
its volume) and the exception hierarchy are available from the package
itself; CDM input is in :mod:`keepout.cdm`, the individual Pc methods in
:mod:`keepout.pc`, risk levels, trends and decisions in :mod:`keepout.risk`,
element sets in :mod:`keepout.omm`, the screening search, its parts and the
reasons for skipped element sets in :mod:`keepout.screening`, propagators in
:mod:`keepout.propagation` and data sources in :mod:`keepout.sources`.
"""

from importlib.metadata import version

from keepout.conjunction import Conjunction
from keepout.covariance import Covariance, make_covariance
from keepout.errors import (
    CaraFormatError,
    CcsdsParseError,
    CdmParseError,
    CdmWriteError,
    ChecksumMismatch,
    CovarianceError,
    GeometryError,
    HttpStatusError,
    KeepoutError,
    KelvinsFormatError,
    OmmParseError,
    OmmWriteError,
    PcConvergenceError,
    SourceFormatError,
    SourceUnavailable,
    UnsupportedElements,
    UnsupportedFrame,
)
from keepout.objects import ObjectState
from keepout.pc import Method, MethodChoice, PcResult, compute_pc
from keepout.risk import ConjunctionEvent, PcObservation, assess
from keepout.screening import (
    CloseApproach,
    ScreeningResult,
    ScreeningVolume,
    screen,
)
from keepout.state import RefFrame, StateVector

__version__ = version("keepout")

__all__ = [
    "CaraFormatError",
    "CcsdsParseError",
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
    "OmmWriteError",
    "PcConvergenceError",
    "PcObservation",
    "PcResult",
    "RefFrame",
    "ScreeningResult",
    "ScreeningVolume",
    "SourceFormatError",
    "SourceUnavailable",
    "StateVector",
    "UnsupportedElements",
    "UnsupportedFrame",
    "__version__",
    "assess",
    "compute_pc",
    "make_covariance",
    "screen",
]
