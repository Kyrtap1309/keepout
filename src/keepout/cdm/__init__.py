"""CCSDS Conjunction Data Messages: model, KVN and XML.

Use :func:`load` / :func:`loads` to read either format; :mod:`keepout.cdm.kvn`
and :mod:`keepout.cdm.xml` read and write a specific one.
:func:`~keepout.cdm.builder.make_cdm` builds the CDM of a conjunction.
"""

from pathlib import Path

from keepout.cdm import kvn, xml
from keepout.cdm._io import load_file
from keepout.cdm.builder import (
    Maneuverable,
    ObjectMetadata,
    ObjectType,
    make_cdm,
    make_message_id,
)
from keepout.cdm.model import Cdm, CdmObject, CovarianceMethod, Field, Section

__all__ = [
    "Cdm",
    "CdmObject",
    "CovarianceMethod",
    "Field",
    "Maneuverable",
    "ObjectMetadata",
    "ObjectType",
    "Section",
    "kvn",
    "load",
    "loads",
    "make_cdm",
    "make_message_id",
    "xml",
]


def loads(text: str) -> Cdm:
    """The CDM in KVN or XML ``text``, told apart by the first non-blank character.

    Parameters
    ----------
    text
        The message.
    """
    stripped = text.lstrip("\ufeff \t\r\n")
    return xml.loads(stripped) if stripped.startswith("<") else kvn.loads(stripped)


def load(path: str | Path) -> Cdm:
    """The CDM in a KVN or XML file.

    Parameters
    ----------
    path
        File to read (UTF-8, gzip-compressed or not).
    """
    return load_file(path, loads)
