"""Propagation of many objects at once.

:class:`Propagator` is the interface screening relies on: the states of N
objects at T times, with an error code for each, as an :class:`Ephemeris`,
and bounds on each object's distance from the Earth's centre over a time
span (:class:`RadiusBounds`). :class:`Sgp4Propagator` implements it with
SGP4 for OMM element sets (TEME output); :func:`satrec_of` gives one element
set's SGP4 record and :func:`sgp4_shape` its orbit's size as SGP4 sees it.
:func:`propagate_in_chunks` walks a long time grid in chunks of bounded
memory, and :func:`pair_states` gives the states of single pairs (SGP4's
fast path when it can).
"""

from keepout.propagation.base import (
    CHUNK_STATES,
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
    "ERROR_MESSAGES",
    "MAX_SATREC_NUMBER",
    "NO_ERROR",
    "Ephemeris",
    "ErrorArray",
    "PairStates",
    "Propagator",
    "RadiusBounds",
    "Sgp4Propagator",
    "Sgp4Shape",
    "States",
    "describe_error",
    "pair_sampler",
    "pair_states",
    "propagate_in_chunks",
    "satrec_of",
    "sgp4_shape",
    "time_offsets",
]
