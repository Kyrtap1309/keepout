"""Propagation of many objects at once.

:class:`Propagator` is the interface screening relies on: the states of N
objects at T times, with an error code for each, as an :class:`Ephemeris`,
and bounds on each object's distance from the Earth's centre over a time
span (:class:`RadiusBounds`). :class:`Sgp4Propagator` implements it with
SGP4 for OMM element sets (TEME output); :func:`satrec_of` gives one element
set's SGP4 record and :func:`sgp4_shape` its orbit's size as SGP4 sees it.
:class:`CowellPropagator` implements it with numerical integration under a
:class:`ForceModel` (zonal gravity to J6 about :func:`earth_pole`, drag if
given; no third bodies or solar radiation pressure) in GCRF, on
:func:`integrate`, which carries many states through one integration and is
what :mod:`keepout.maneuver` uses for a whole grid of burns.
:class:`ManeuveredPropagator` adds the effect of a burn on one object, as a
Cowell difference, to any propagator: the re-screen of a planned burn.
:func:`propagate_in_chunks` walks a long time grid in chunks of bounded
memory, and :func:`pair_states` gives the states of single pairs (SGP4's
fast path when it can).
"""

from keepout.propagation.base import (
    CHUNK_STATES,
    DECAYED,
    ERROR_MESSAGES,
    NO_ERROR,
    Ephemeris,
    ErrorArray,
    Propagator,
    RadiusBounds,
    States,
    describe_error,
    pair_sampler,
    propagate_in_chunks,
    time_offsets,
)
from keepout.propagation.cowell import CowellPropagator, integrate
from keepout.propagation.forces import EARTH_RADIUS, ForceModel, earth_pole
from keepout.propagation.maneuvered import ManeuveredPropagator
from keepout.propagation.sgp4_prop import (
    MAX_SATREC_NUMBER,
    PairStates,
    Sgp4Propagator,
    Sgp4Shape,
    pair_states,
    satrec_of,
    sgp4_shape,
)

__all__ = [
    "CHUNK_STATES",
    "DECAYED",
    "EARTH_RADIUS",
    "ERROR_MESSAGES",
    "MAX_SATREC_NUMBER",
    "NO_ERROR",
    "CowellPropagator",
    "Ephemeris",
    "ErrorArray",
    "ForceModel",
    "ManeuveredPropagator",
    "PairStates",
    "Propagator",
    "RadiusBounds",
    "Sgp4Propagator",
    "Sgp4Shape",
    "States",
    "describe_error",
    "earth_pole",
    "integrate",
    "pair_sampler",
    "pair_states",
    "propagate_in_chunks",
    "satrec_of",
    "sgp4_shape",
    "time_offsets",
]
