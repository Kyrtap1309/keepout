"""A synthetic operator: assumed covariances for GP element sets.

:class:`SyntheticCovarianceModel` gives a GP (TLE/OMM) element set an RTN
position covariance that grows with the element set's age, with parameters
per orbit regime (:class:`OrbitRegime`, :func:`orbit_regime`) from the
literature. It is a documented assumption, not a calibrated product.
"""

from keepout.synthetic.covariance_model import (
    OrbitRegime,
    RegimeParameters,
    RtnSigmas,
    SyntheticCovarianceModel,
    orbit_regime,
)

__all__ = [
    "OrbitRegime",
    "RegimeParameters",
    "RtnSigmas",
    "SyntheticCovarianceModel",
    "orbit_regime",
]
