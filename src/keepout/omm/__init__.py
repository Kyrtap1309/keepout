"""Orbit Mean-Elements Messages: SGP4 mean elements of catalogued objects.

:class:`Omm` holds one object's elements in SI units; :meth:`Omm.satrec`
initialises SGP4 with them. :mod:`keepout.omm.celestrak` reads and writes
CelesTrak's CSV and JSON.
"""

from keepout.omm import celestrak
from keepout.omm.model import Omm

__all__ = ["Omm", "celestrak"]
