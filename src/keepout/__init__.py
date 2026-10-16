"""keepout — spacecraft conjunction assessment.

The core types, the Pc entry point, the event assessment, screening (with
its volume) and the exception hierarchy are available from the package
itself; CDM input and output are in :mod:`keepout.cdm`, the individual Pc
methods in :mod:`keepout.pc`, risk levels, trends and decisions in
:mod:`keepout.risk`, element sets in :mod:`keepout.omm`, the screening
search, its parts and the reasons for skipped element sets in
:mod:`keepout.screening`, propagators in :mod:`keepout.propagation`, the
synthetic operator (covariance model, fleet, run) in
:mod:`keepout.synthetic` and data sources (CelesTrak, Space-Track, CARA,
Kelvins) in :mod:`keepout.sources`. The ``keepout`` command
(:mod:`keepout.cli`) needs the ``cli`` extra.
"""

from importlib.metadata import version

from keepout.conjunction import Conjunction
from keepout.covariance import Covariance, make_covariance
from keepout.errors import (
    AuthError,
    CaraFormatError,
    CcsdsParseError,
    CdmParseError,
    CdmWriteError,
    ChecksumMismatch,
    CovarianceError,
    FleetError,
    GeometryError,
    HttpStatusError,
    KeepoutError,
    KelvinsFormatError,
    OmmParseError,
    OmmWriteError,
    PcConvergenceError,
    RateLimited,
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
    "AuthError",
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
    "FleetError",
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
    "RateLimited",
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
