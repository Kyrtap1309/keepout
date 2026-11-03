"""keepout — spacecraft conjunction assessment.

The package itself exports the core types, the Pc entry point, the event
assessment, screening (with its volume) and the exception hierarchy. The
rest is in subpackages:

- :mod:`keepout.cdm`: CDM input and output.
- :mod:`keepout.pc`: the individual Pc methods.
- :mod:`keepout.events`: conjunction events, the CDMs of one encounter.
- :mod:`keepout.risk`: risk levels, trends and decisions.
- :mod:`keepout.omm`: element sets.
- :mod:`keepout.screening`: the screening search, its parts and the reasons
  for skipped element sets.
- :mod:`keepout.propagation`: propagators.
- :mod:`keepout.synthetic`: the synthetic operator (covariance model, fleet,
  run).
- :mod:`keepout.maneuver`: avoidance manoeuvre planning.
- :mod:`keepout.sources`: data sources (CelesTrak, Space-Track, CARA,
  Kelvins).

The ``keepout`` command (:mod:`keepout.cli`) needs the ``cli`` extra.
"""

from __future__ import annotations

import importlib
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
    ManeuverError,
    OmmParseError,
    OmmWriteError,
    PcConvergenceError,
    PropagationError,
    RateLimited,
    SourceFormatError,
    SourceUnavailable,
    UnsupportedElements,
    UnsupportedFrame,
)

if TYPE_CHECKING:
    from keepout.conjunction import Conjunction
    from keepout.covariance import Covariance, make_covariance
    from keepout.events import ConjunctionEvent
    from keepout.objects import ObjectState
    from keepout.pc import Method, MethodChoice, PcResult, compute_pc
    from keepout.risk import PcObservation, assess
    from keepout.screening import (
        CloseApproach,
        ScreeningResult,
        ScreeningVolume,
        screen,
    )
    from keepout.state import RefFrame, StateVector

    __version__: str
    """keepout's version, from its distribution's metadata."""

_LAZY: Final = {
    "Conjunction": "keepout.conjunction",
    "Covariance": "keepout.covariance",
    "make_covariance": "keepout.covariance",
    "ObjectState": "keepout.objects",
    "Method": "keepout.pc",
    "MethodChoice": "keepout.pc",
    "PcResult": "keepout.pc",
    "compute_pc": "keepout.pc",
    "ConjunctionEvent": "keepout.events",
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
        """A name of :data:`_LAZY`, imported on first use (PEP 562), or
        ``__version__``, read from the metadata on first use.

        Raises
        ------
        AttributeError
            If the package has no such name.
        """
        if name == "__version__":
            # importlib.metadata takes most of the time of import keepout.
            from importlib.metadata import version

            value = version("keepout")
            globals()[name] = value
            return value
        module = _LAZY.get(name)
        if module is None:
            raise AttributeError(f"module 'keepout' has no attribute {name!r}")
        value = getattr(importlib.import_module(module), name)
        globals()[name] = value
        return value

    def __dir__() -> list[str]:
        return sorted({*globals(), *_LAZY, "__version__"})


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
    "ManeuverError",
    "Method",
    "MethodChoice",
    "ObjectState",
    "OmmParseError",
    "OmmWriteError",
    "PcConvergenceError",
    "PcObservation",
    "PcResult",
    "PropagationError",
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
