"""CelesTrak's CSV and JSON forms of OMM general perturbations data.

CelesTrak (https://celestrak.org/NORAD/documentation/gp-data-formats.php)
publishes the GP data of the 18th Space Defense Squadron as CSV (one header
row of OMM keywords, then one row per object) and as JSON (an array of
objects keyed by OMM keyword, numbers as JSON numbers). Both carry the
keywords of :data:`keepout.omm.keywords.KEYWORDS`; their reference frame
(TEME), time system (UTC) and theory (SGP4) are implied. The readers also
take Space-Track's ``gp`` JSON, whose numbers are strings and which has
more keywords: the extra ones are ignored, except that the metadata of
:data:`~keepout.omm.keywords.FIXED_METADATA` must match if present.

The writers give back what the readers take: reading a written file gives
equal :class:`~keepout.omm.model.Omm` objects.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable
from pathlib import Path

from keepout.errors import OmmParseError
from keepout.omm._io import load_file, parse_record
from keepout.omm.keywords import KEYWORDS
from keepout.omm.model import Omm

__all__ = [
    "dumps_csv",
    "dumps_json",
    "load_csv",
    "load_json",
    "loads_csv",
    "loads_json",
]


def loads_csv(text: str) -> tuple[Omm, ...]:
    """Parse CelesTrak CSV: a header row of keywords, then one row per object.

    Columns may come in any order; columns other than the keywords of
    :data:`~keepout.omm.keywords.KEYWORDS` are ignored, and blank lines are
    skipped.

    Parameters
    ----------
    text
        The CSV text (RFC 4180).

    Returns
    -------
    tuple[Omm, ...]
        The objects in file order (none for a header without rows).

    Raises
    ------
    OmmParseError
        If there is no header, the header repeats a column or lacks a
        keyword, a row has more or fewer values than the header, the CSV
        quoting is broken, or an object's values are invalid.
    """
    reader = csv.reader(io.StringIO(text.removeprefix("﻿"), newline=""), strict=True)
    try:
        header = next(reader, None)
        if header is None:
            raise OmmParseError("no CSV header", line=1)
        _check_header(header)
        omms: list[Omm] = []
        for row in reader:
            if not row:
                continue
            record, line = len(omms) + 1, reader.line_num
            if len(row) != len(header):
                raise OmmParseError(
                    f"{len(row)} values for {len(header)} columns",
                    record=record,
                    line=line,
                )
            values = dict(zip(header, row, strict=True))
            omms.append(parse_record(values, record=record, line=line))
    except csv.Error as exc:
        raise OmmParseError(f"invalid CSV: {exc}", line=reader.line_num) from exc
    return tuple(omms)


def _check_header(header: list[str]) -> None:
    seen: set[str] = set()
    for column in header:
        if column in seen:
            raise OmmParseError("repeated column", line=1, keyword=column)
        seen.add(column)
    for keyword in KEYWORDS:
        if keyword not in seen:
            raise OmmParseError("missing column", line=1, keyword=keyword)


def loads_json(text: str) -> tuple[Omm, ...]:
    """Parse CelesTrak JSON: an array with one object per element set.

    Values may be JSON numbers or strings (Space-Track writes every value as
    a string); a number keeps its exact text, so it reads like the same
    number in CSV.

    Parameters
    ----------
    text
        The JSON text.

    Returns
    -------
    tuple[Omm, ...]
        The objects in array order.

    Raises
    ------
    OmmParseError
        If the text is not JSON, not an array of objects, or an object's
        values are invalid.
    """
    try:
        data = json.loads(
            text.removeprefix("﻿"),
            parse_float=str,
            parse_int=str,
            parse_constant=str,
        )
    except json.JSONDecodeError as exc:
        raise OmmParseError(f"invalid JSON: {exc.msg}", line=exc.lineno) from exc
    if not isinstance(data, list):
        raise OmmParseError("JSON must be an array of OMM objects")
    omms: list[Omm] = []
    for record, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise OmmParseError("not a JSON object", record=record)
        omms.append(parse_record(item, record=record))
    return tuple(omms)


def load_csv(path: str | Path) -> tuple[Omm, ...]:
    """Read a CelesTrak CSV file (UTF-8); see :func:`loads_csv`.

    Parameters
    ----------
    path
        File to read.
    """
    return load_file(path, loads_csv)


def load_json(path: str | Path) -> tuple[Omm, ...]:
    """Read a CelesTrak JSON file (UTF-8); see :func:`loads_json`.

    Parameters
    ----------
    path
        File to read.
    """
    return load_file(path, loads_json)


def dumps_csv(omms: Iterable[Omm]) -> str:
    """Write CelesTrak CSV: the header, then one row per object.

    Parameters
    ----------
    omms
        The objects, written in order.
    """
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(KEYWORDS)
    writer.writerows(omm.keywords().values() for omm in omms)
    return stream.getvalue()


def dumps_json(omms: Iterable[Omm]) -> str:
    """Write CelesTrak JSON: an array of objects, numbers as JSON numbers.

    Parameters
    ----------
    omms
        The objects, written in order.
    """
    return json.dumps([_json_object(omm) for omm in omms])


def _json_object(omm: Omm) -> dict[str, str | int | float]:
    values: dict[str, str | int | float] = {}
    for keyword, text in omm.keywords().items():
        match KEYWORDS[keyword].kind:
            case "int":
                values[keyword] = int(text)
            case "float":
                # json writes repr(), which reads back as the same float.
                values[keyword] = float(text)
            case _:
                values[keyword] = text
    return values
