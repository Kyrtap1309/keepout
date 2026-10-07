"""Propagation of many objects at once.

:class:`Propagator` is the interface screening relies on: the states of N
objects at T times, with an error code for each, as an :class:`Ephemeris`.
:class:`Sgp4Propagator` implements it with SGP4 for OMM element sets (TEME
output). :func:`propagate_in_chunks` walks a long time grid in chunks of
bounded memory.
"""

from keepout.propagation.base import (
    CHUNK_STATES,
    ERROR_MESSAGES,
    NO_ERROR,
    Ephemeris,
    ErrorArray,
    Propagator,
    describe_error,
    propagate_in_chunks,
    time_offsets,
)
from keepout.propagation.sgp4_prop import Sgp4Propagator

__all__ = [
    "CHUNK_STATES",
    "ERROR_MESSAGES",
    "NO_ERROR",
    "Ephemeris",
    "ErrorArray",
    "Propagator",
    "Sgp4Propagator",
    "describe_error",
    "propagate_in_chunks",
    "time_offsets",
]
