"""What the CCSDS message formats share: keyword values, KVN lines and XML.

:mod:`keepout.ccsds.message` describes a message family (its errors) and
the values its readers find, with their place; :mod:`keepout.ccsds.keywords`
types and converts keyword values; :mod:`keepout.ccsds.epochs` reads and
writes UTC epochs;
:mod:`keepout.ccsds.kvn` reads and writes ``KEYWORD = value [unit]`` lines;
:mod:`keepout.ccsds.xml` reads NDM/XML documents and writes them;
:mod:`keepout.ccsds.files` reads message files, gzip-compressed or not.
The CDM (:mod:`keepout.cdm`) and OMM (:mod:`keepout.omm`) readers and
writers are built on them.
"""

from keepout.ccsds import epochs, files, keywords, kvn, message, xml

__all__ = ["epochs", "files", "keywords", "kvn", "message", "xml"]
