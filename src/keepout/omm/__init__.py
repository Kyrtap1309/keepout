"""Orbit Mean-Elements Messages: SGP4 mean elements of catalogued objects.

:class:`Omm` holds one object's elements in SI units
(:func:`keepout.propagation.satrec_of` initialises SGP4 with them);
:func:`deduplicate` keeps the newest of several sets of one object.
:func:`load` and :func:`loads` read every format
keepout knows, recognised by content: CCSDS OMM in KVN (:mod:`keepout.omm.kvn`)
or XML (:mod:`keepout.omm.xml`), CelesTrak's CSV or JSON
(:mod:`keepout.omm.celestrak`) and legacy TLEs (:mod:`keepout.omm.tle`). Each
of those modules reads one format and, but for TLE, writes it; each is
imported on first use, so ``from keepout.omm import Omm`` loads none of them.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Final

from keepout.ccsds.files import strip_bom
from keepout.errors import OmmParseError
from keepout.omm._io import OMM
from keepout.omm.keywords import VERSION_KEYWORD
from keepout.omm.model import Deduplicated, Omm, deduplicate

if TYPE_CHECKING:
    from keepout.omm import celestrak, kvn, tle, xml

__all__ = [
    "Deduplicated",
    "Omm",
    "celestrak",
    "deduplicate",
    "kvn",
    "load",
    "loads",
    "tle",
    "xml",
]

_FORMATS: Final = frozenset({"celestrak", "kvn", "tle", "xml"})
"""The format modules, imported on first use (PEP 562)."""


# Hidden from type checkers: they see the imports above.
if not TYPE_CHECKING:

    def __getattr__(name: str) -> object:
        """A format module of :data:`_FORMATS`, imported on first use.

        Raises
        ------
        AttributeError
            If the package has no such name.
        """
        if name not in _FORMATS:
            raise AttributeError(f"module 'keepout.omm' has no attribute {name!r}")
        return importlib.import_module(f"keepout.omm.{name}")

    def __dir__() -> list[str]:
        return sorted({*globals(), *_FORMATS})


def loads(text: str) -> tuple[Omm, ...]:
    """The OMMs in ``text`` in any format keepout reads, recognised by its content.

    Leading blanks aside, the text is XML if it starts with ``<``, CelesTrak
    JSON if with ``[`` or ``{``, KVN if with ``CCSDS_OMM_VERS``, CelesTrak CSV
    if its first line is a header with a ``NORAD_CAT_ID`` column, and TLEs if
    its first or second line starts with ``1`` and a blank. Line numbers in
    errors count from the first non-blank line for XML and CSV, from the
    first line otherwise.

    Parameters
    ----------
    text
        The data: element sets in one format.

    Returns
    -------
    tuple[Omm, ...]
        The objects in the order the data gives them.

    Raises
    ------
    OmmParseError
        If the text is in none of these formats, or the format's reader
        refuses it.
    """
    from keepout.omm import celestrak, kvn, tle, xml

    body = strip_bom(text)
    start = body.lstrip()
    first, *second = start.splitlines()[:2] or [""]
    if start.startswith("<"):
        return xml.loads(start)
    if start.startswith(("[", "{")):
        return celestrak.loads_json(body)
    if start.startswith(VERSION_KEYWORD):
        return kvn.loads(body)
    if "," in first and "NORAD_CAT_ID" in first:
        return celestrak.loads_csv(start)
    if any(line.startswith("1 ") for line in (first, *second)):
        return tle.loads(body)
    raise OmmParseError(
        "not OMM data (CCSDS KVN or XML, CelesTrak CSV or JSON, or TLE): "
        f"starts with {first[:40]!r}"
    )


def load(path: str | Path) -> tuple[Omm, ...]:
    """The OMMs in a file in any format keepout reads; see :func:`loads`.

    Parameters
    ----------
    path
        File to read: UTF-8 text, or gzip-compressed UTF-8 text (recognised
        by its content, whatever the file name).

    Raises
    ------
    OmmParseError
        If the file is broken gzip data, not UTF-8 text, or not OMM data.
    """
    return OMM.load(path, loads)
