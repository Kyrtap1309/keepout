"""Space-Track's public CDM summaries (``cdm_public``): the rows and their
reader.

:class:`~keepout.sources.spacetrack.SpaceTrackClient` downloads them
(:data:`~keepout.sources.spacetrack.PUBLIC_CDMS`);
:func:`read_public_cdms` reads an answer into :class:`PublicCdm` rows. The
rows may not be redistributed without Space-Track's approval.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from astropy.time import Time

from keepout._types import JsonValue
from keepout._validation import (
    require_at_least,
    require_non_negative,
    require_probability,
)
from keepout.errors import SourceFormatError
from keepout.sources._format import refuse_format
from keepout.sources.catalog import Retrieval

__all__ = [
    "PUBLIC_CDM_COLUMNS",
    "PublicCdm",
    "PublicCdmObject",
    "PublicCdms",
    "read_public_cdms",
]

PUBLIC_CDM_COLUMNS: Final = (
    "CDM_ID",
    "CREATED",
    "EMERGENCY_REPORTABLE",
    "TCA",
    "MIN_RNG",
    "PC",
    "SAT_1_ID",
    "SAT_1_NAME",
    "SAT1_OBJECT_TYPE",
    "SAT1_RCS",
    "SAT_1_EXCL_VOL",
    "SAT_2_ID",
    "SAT_2_NAME",
    "SAT2_OBJECT_TYPE",
    "SAT2_RCS",
    "SAT_2_EXCL_VOL",
)
"""The 16 columns of the ``cdm_public`` class, as Space-Track names them."""

_YES_NO: Final[Mapping[str, bool]] = MappingProxyType({"Y": True, "N": False})
"""``EMERGENCY_REPORTABLE``'s values."""


@dataclass(frozen=True, slots=True)
class PublicCdmObject:
    """One of the two objects of a :class:`PublicCdm`.

    Parameters
    ----------
    catalog_number
        Its catalog number (``SAT_1_ID``, ``SAT_2_ID``).
    name
        Its name (``SAT_1_NAME``); ``None`` if not given.
    object_type
        ``PAYLOAD``, ``ROCKET BODY``, ``DEBRIS``, ``UNKNOWN``, ...
        (``SAT1_OBJECT_TYPE``); ``None`` if not given.
    rcs_size
        Its radar cross-section class, ``SMALL``, ``MEDIUM`` or ``LARGE``
        (``SAT1_RCS``); ``None`` if not given.
    exclusion_volume
        ``SAT_1_EXCL_VOL`` as Space-Track gives it (e.g. ``5.00``); its
        meaning and unit are not published. ``None`` if not given.
    """

    catalog_number: int
    name: str | None
    object_type: str | None
    rcs_size: str | None
    exclusion_volume: str | None


@dataclass(frozen=True, slots=True)
class PublicCdm:
    """A row of Space-Track's ``cdm_public`` class: the public summary of a
    conjunction data message (CDM) of the 18th Space Defense Squadron.

    The summary has no states, covariances or hard-body radii (the CDMs
    themselves are for the objects' operators); it may not be redistributed
    without Space-Track's approval.

    Parameters
    ----------
    cdm_id
        The CDM's number at Space-Track (``CDM_ID``).
    created
        When the CDM was made (``CREATED``, UTC).
    emergency_reportable
        Whether the approach meets 18 SDS's emergency reporting criteria
        (``EMERGENCY_REPORTABLE``, ``Y`` or ``N``).
    tca
        Time of closest approach (``TCA``, UTC).
    range_at_tca
        Range at TCA [m] (``MIN_RNG``, metres in the answer).
    pc
        Probability of collision (``PC``); ``None`` if not given.
    object1, object2
        The two objects (``SAT_1_*`` and ``SAT_2_*`` columns).
    """

    cdm_id: int
    created: Time
    emergency_reportable: bool
    tca: Time
    range_at_tca: float
    pc: float | None
    object1: PublicCdmObject
    object2: PublicCdmObject


@dataclass(frozen=True, slots=True)
class PublicCdms:
    """Public CDM summaries as one query returned them.

    Parameters
    ----------
    cdms
        The rows, in Space-Track's order.
    retrieval
        Where and when they were downloaded.
    """

    cdms: tuple[PublicCdm, ...]
    retrieval: Retrieval


def read_public_cdms(data: bytes) -> tuple[PublicCdm, ...]:
    """The rows of a Space-Track ``cdm_public`` answer in JSON.

    Each row must have the 16 columns of :data:`PUBLIC_CDM_COLUMNS` (more
    are ignored); values are JSON strings or numbers, and ``null`` or an
    empty string is a missing value, which only the names, object types,
    radar cross-sections, exclusion volumes and ``PC`` may be.

    Parameters
    ----------
    data
        The body of the answer, as Space-Track sent it.

    Raises
    ------
    SourceFormatError
        If ``data`` is not UTF-8 JSON, not an array of objects, or a row
        lacks a column or has a value that cannot be read.
    """
    try:
        records: JsonValue = json.loads(data.decode("utf-8"))
    except ValueError as exc:
        refuse_format("Space-Track", "JSON", data, exc)
    if not isinstance(records, list):
        refuse_format("Space-Track", "an array of cdm_public rows", data)
    cdms: list[PublicCdm] = []
    for number, record in enumerate(records, start=1):
        try:
            cdms.append(_public_cdm(record))
        except ValueError as exc:
            raise SourceFormatError(f"cdm_public row {number}: {exc}") from exc
    return tuple(cdms)


def _public_cdm(record: object) -> PublicCdm:
    """A ``cdm_public`` row; ``ValueError`` if it cannot be read.

    Raises
    ------
    ValueError
        If ``record`` is not a JSON object, lacks a column, or has a
        missing, malformed or out-of-range value.
    """
    if not isinstance(record, dict):
        raise ValueError(f"not a JSON object: {record!r}")
    missing = [column for column in PUBLIC_CDM_COLUMNS if column not in record]
    if missing:
        raise ValueError(f"no {', '.join(missing)}")
    values = {column: _cell(record[column], column) for column in PUBLIC_CDM_COLUMNS}

    def required(column: str) -> str:
        value = values[column]
        if value is None:
            raise ValueError(f"{column} is empty")
        return value

    def number(column: str) -> float:
        value = float(required(column))
        require_non_negative(value, column)
        return value

    def catalog_number(column: str) -> int:
        value = int(required(column))
        require_at_least(value, 1, column)
        return value

    def public_object(n: int) -> PublicCdmObject:
        # Space-Track spells the columns SAT_1_ID but SAT1_OBJECT_TYPE.
        return PublicCdmObject(
            catalog_number=catalog_number(f"SAT_{n}_ID"),
            name=values[f"SAT_{n}_NAME"],
            object_type=values[f"SAT{n}_OBJECT_TYPE"],
            rcs_size=values[f"SAT{n}_RCS"],
            exclusion_volume=values[f"SAT_{n}_EXCL_VOL"],
        )

    reportable = required("EMERGENCY_REPORTABLE")
    if reportable not in _YES_NO:
        raise ValueError(f"EMERGENCY_REPORTABLE must be Y or N, got {reportable!r}")
    probability = None if values["PC"] is None else float(values["PC"])
    if probability is not None:
        require_probability(probability, "PC")
    return PublicCdm(
        cdm_id=catalog_number("CDM_ID"),
        created=_time(required("CREATED"), "CREATED"),
        emergency_reportable=_YES_NO[reportable],
        tca=_time(required("TCA"), "TCA"),
        range_at_tca=number("MIN_RNG"),
        pc=probability,
        object1=public_object(1),
        object2=public_object(2),
    )


def _cell(value: object, column: str) -> str | None:
    """A JSON value as text, without surrounding blanks; ``None`` if missing.

    Raises
    ------
    ValueError
        If ``value`` is neither text nor a number.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        raise ValueError(f"{column} must be text or a number, got {value!r}")
    text = str(value).strip()
    return text or None


def _time(text: str, column: str) -> Time:
    """A UTC time in ISO 8601 (``T`` or a space between date and time).

    Raises
    ------
    ValueError
        If ``text`` is not an ISO 8601 time.
    """
    try:
        return Time(text.replace(" ", "T", 1), format="isot", scale="utc")
    except ValueError as exc:
        raise ValueError(f"{column} is not an ISO 8601 time: {text!r}") from exc
