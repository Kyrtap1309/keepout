"""A synthetic operator: assumed covariances, a fleet, CDMs and their risk.

:class:`SyntheticCovarianceModel` gives a GP (TLE/OMM) element set an RTN
position covariance that grows with the element set's age, with parameters
per orbit regime (:class:`OrbitRegime`, :func:`orbit_regime`) from the
literature. It is a documented assumption, not a calibrated product.

A :class:`Fleet` (:class:`FleetSatellite`; read from a TOML file by
:func:`load_fleet` or :func:`read_fleet`) names the satellites a synthetic
operator protects. :func:`run` screens the
fleet against a catalog, issues a CDM for every close approach with the
model's covariances and assesses its risk (:class:`OperatorResult`,
:class:`OperatorEvent`).
"""

from keepout.synthetic.covariance_model import (
    OrbitRegime,
    RegimeParameters,
    RtnSigmas,
    SyntheticCovarianceModel,
    orbit_regime,
)
from keepout.synthetic.fleet import Fleet, FleetSatellite, load_fleet, read_fleet
from keepout.synthetic.operator import OperatorEvent, OperatorResult, run

__all__ = [
    "Fleet",
    "FleetSatellite",
    "OperatorEvent",
    "OperatorResult",
    "OrbitRegime",
    "RegimeParameters",
    "RtnSigmas",
    "SyntheticCovarianceModel",
    "load_fleet",
    "orbit_regime",
    "read_fleet",
    "run",
]
