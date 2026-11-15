"""keepout: spacecraft conjunction assessment.

keepout reads CDMs and computes their probability of collision (Pc), follows
the risk of conjunction events, plans avoidance manoeuvres and screens a
fleet against a GP catalog. The package exports the core types,
:func:`compute_pc`, :func:`assess`, :func:`screen` and the exceptions;
``__version__`` is keepout's version. The API reference's overview
(``docs/reference/index.md``) maps the subpackages, and the guide's
Getting started page (``docs/index.md``) has the examples.
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
    InvalidArgument,
    KeepoutError,
    KelvinsFormatError,
    ManeuverError,
    OmmParseError,
    OmmWriteError,
    PcConvergenceError,
    PropagationError,
    RateLimited,
    SettingsError,
    SourceFormatError,
    SourceUnavailable,
    StoreError,
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
    """keepout's version, from its distribution's metadata (PEP 440, e.g.
    ``"0.1.0"``); read on first use."""

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
        """Import the public names on first use (PEP 562).

        A public name is imported from the module that defines it the first
        time it is read; ``__version__`` is read from the distribution's
        metadata the first time.

        Parameters
        ----------
        name
            The attribute asked for.

        Returns
        -------
        object
            The public name's value.

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
        """The package's names, with the public names not imported yet.

        Returns
        -------
        list[str]
            Every name, sorted.
        """
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
    "InvalidArgument",
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
    "SettingsError",
    "SourceFormatError",
    "SourceUnavailable",
    "StateVector",
    "StoreError",
    "UnsupportedElements",
    "UnsupportedFrame",
    "__version__",
    "assess",
    "compute_pc",
    "make_covariance",
    "screen",
]
