"""keepout — spacecraft conjunction assessment.

keepout reads Conjunction Data Messages (CDMs: the warnings, in the format
of the Consultative Committee for Space Data Systems, CCSDS, that two
objects will pass close) and computes the probability of collision (Pc) of
each at its time of closest approach (TCA), for the combined hard-body
radius (HBR) of the two objects. It groups CDMs into events, follows the
trend of their Pc and recommends an action, plans avoidance manoeuvres, and
screens a fleet against a catalog of general perturbations (GP) element
sets, read as Orbit Mean-Elements Messages (OMMs) and propagated with SGP4
(Simplified General Perturbations 4). States are in SI units (metres,
seconds); covariances are given in a satellite's RTN frame (radial,
transverse, normal) or in the inertial GCRF (Geocentric Celestial Reference
Frame), and SGP4 states are in TEME (True Equator, Mean Equinox).

The package itself exports the core types, the Pc entry point
(:func:`compute_pc`), the event assessment (:func:`assess`), screening
(:func:`screen`, with its volume) and the exception hierarchy; ``__version__``
is keepout's version, from its distribution's metadata. The rest is in
subpackages:

- :mod:`keepout.cdm`: CDM input and output.
- :mod:`keepout.pc`: the individual Pc methods.
- :mod:`keepout.events`: conjunction events, the CDMs of one encounter.
- :mod:`keepout.risk`: risk levels, trends and decisions.
- :mod:`keepout.omm`: element sets (OMM, CelesTrak CSV and JSON, TLE).
- :mod:`keepout.screening`: the screening search, its parts and the reasons
  for skipped element sets.
- :mod:`keepout.propagation`: propagators (SGP4, numerical, with burns).
- :mod:`keepout.synthetic`: the synthetic operator (covariance model, fleet,
  run).
- :mod:`keepout.maneuver`: avoidance manoeuvre planning.
- :mod:`keepout.settings`: an operator's settings, read from a TOML file.
- :mod:`keepout.sources`: data sources (CelesTrak, Space-Track, CARA,
  Kelvins).
- :mod:`keepout.store`: the conjunction store, a SQLite database (needs the
  ``store`` extra).
- :mod:`keepout.validation`: the Pc methods checked against NASA CARA's
  reference values.

The modules beneath them hold the geometry: :mod:`keepout.state` (state
vectors), :mod:`keepout.covariance`, :mod:`keepout.objects`,
:mod:`keepout.conjunction`, :mod:`keepout.encounter` (the encounter plane),
:mod:`keepout.frames`, :mod:`keepout.orbits`, :mod:`keepout.equinoctial`,
:mod:`keepout.units`; :mod:`keepout.ccsds` holds what the CCSDS formats
share and :mod:`keepout.errors` the exceptions.

The ``keepout`` command (:mod:`keepout.cli`) needs the ``cli`` extra.

Examples
--------
```python
import keepout
from keepout import cdm, compute_pc

message = cdm.load("tests/cdm/data/tracss_v2.kvn")
result = compute_pc(message.conjunction(combined_hard_body_radius=20.0))  # [m]
print(result.method, f"{result.value:.3e}")
# hall3d 4.504e-02
print(keepout.__version__)  # e.g. 0.1.0.dev0
```
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
