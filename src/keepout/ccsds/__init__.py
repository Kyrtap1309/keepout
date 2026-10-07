"""What the CCSDS message formats share: keyword values, KVN lines and XML.

:mod:`keepout.ccsds.keywords` types and converts keyword values and reads
and writes UTC epochs; :mod:`keepout.ccsds.kvn` reads and writes
``KEYWORD = value [unit]`` lines; :mod:`keepout.ccsds.xml` reads NDM/XML
documents and writes them. The CDM (:mod:`keepout.cdm`) and OMM
(:mod:`keepout.omm`) readers and writers are built on them.
"""

from keepout.ccsds import keywords, kvn, xml

__all__ = ["keywords", "kvn", "xml"]
