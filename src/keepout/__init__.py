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

from __future__ import annotations

import importlib
from importlib.metadata import version
from typing import TYPE_CHECKING, Final

from keepout.errors import (
    AuthError,
    CaraFormatError,
    CcsdsParseError,
    CdmParseError,
    CdmWriteError,
    ChecksumMismatch,
    CovarianceError,
    EventError,
    FleetError,
    GeometryError,
    HardBodyRadiusError,
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

if TYPE_CHECKING:
    from keepout.conjunction import Conjunction
    from keepout.covariance import Covariance, make_covariance
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

_LAZY: Final = {
    "Conjunction": "keepout.conjunction",
    "Covariance": "keepout.covariance",
    "make_covariance": "keepout.covariance",
    "ObjectState": "keepout.objects",
    "Method": "keepout.pc",
    "MethodChoice": "keepout.pc",
    "PcResult": "keepout.pc",
    "compute_pc": "keepout.pc",
    "ConjunctionEvent": "keepout.risk",
    "PcObservation": "keepout.risk",
    "assess": "keepout.risk",
    "CloseApproach": "keepout.screening",
    "ScreeningResult": "keepout.screening",
    "ScreeningVolume": "keepout.screening",
    "screen": "keepout.screening",
    "RefFrame": "keepout.state",
    "StateVector": "keepout.state",
}
"""The names imported on first use, by the module that defines them: numpy,
scipy and astropy load only when one is needed (``keepout --help`` does not
need them)."""


# Hidden from type checkers: they see the imports above, so a name the package
# lacks stays an error instead of an ``object``.
if not TYPE_CHECKING:

    def __getattr__(name: str) -> object:
        """A name of :data:`_LAZY`, imported on first use (PEP 562).

        Raises
        ------
        AttributeError
            If the package has no such name.
        """
        module = _LAZY.get(name)
        if module is None:
            raise AttributeError(f"module 'keepout' has no attribute {name!r}")
        value = getattr(importlib.import_module(module), name)
        globals()[name] = value
        return value

    def __dir__() -> list[str]:
        return sorted({*globals(), *_LAZY})


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
    "EventError",
    "FleetError",
    "GeometryError",
    "HardBodyRadiusError",
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
