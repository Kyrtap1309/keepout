"""States and covariances from CARA's legacy sample CDMs.

Several CDMs that CARA's unit tests use do not conform to CCSDS 508.0-B-1
(e.g. ``RELATIVE_VELOCITY_*`` in ``[m]``, ``NaN`` in numeric fields, units
in a non-UTF-8 encoding), so :mod:`keepout.cdm` rightly rejects them. This
reader takes only what Pc needs — ``TCA``, and per object ``REF_FRAME``,
``OBJECT_DESIGNATOR``, the state and the 6x6 RTN covariance — and ignores
the rest. It is for validation only; operational CDMs go through
:mod:`keepout.cdm`.
"""

from __future__ import annotations

import re
from functools import partial
from pathlib import Path
from typing import Final

from astropy.time import Time

from keepout._units import KM
from keepout.cdm.keywords import covariance_matrix
from keepout.errors import CaraFormatError, UnsupportedFrame
from keepout.objects import ObjectState, ReportedObject, gcrf_object_states
from keepout.state import StateVector, as_ref_frame

__all__ = ["legacy_objects"]

_LINE: Final = re.compile(r"^\s*([A-Z][A-Z0-9_]*)\s*=\s*(\S+)")
_OBJECT: Final = re.compile(r"^\s*OBJECT\s*=")
_UNKNOWN_DESIGNATOR: Final = "?"
"""Object ID of a section without ``OBJECT_DESIGNATOR``."""
_STATE: Final = ("X", "Y", "Z", "X_DOT", "Y_DOT", "Z_DOT")


def _fields(lines: list[str]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in lines:
        match = _LINE.match(line)
        if match and match.group(1) != "COMMENT":
            fields.setdefault(match.group(1), match.group(2))
    return fields


def _number(fields: dict[str, str], keyword: str, path: Path) -> float:
    try:
        return float(fields[keyword])
    except (KeyError, ValueError) as exc:
        raise CaraFormatError(f"{path}: no numeric {keyword}") from exc


def _reported(
    fields: dict[str, str], tca: Time, path: Path, *, known: bool
) -> ReportedObject:
    """One object section; its covariance only if ``known``.

    Raises
    ------
    CaraFormatError
        If the section's ``REF_FRAME`` is not supported.
    """
    try:
        frame = as_ref_frame(fields.get("REF_FRAME"))
    except UnsupportedFrame as exc:
        raise CaraFormatError(f"{path}: bad REF_FRAME ({exc})") from exc
    values = [KM * _number(fields, k, path) for k in _STATE]  # km, km/s
    return ReportedObject(
        fields.get("OBJECT_DESIGNATOR", _UNKNOWN_DESIGNATOR),
        StateVector(tca, values[:3], values[3:], frame),
        covariance_matrix(6, partial(_number, fields, path=path)) if known else None,
    )


def legacy_objects(
    path: Path, *, known: tuple[bool, bool] = (True, True)
) -> tuple[ObjectState, ObjectState]:
    """Both objects of a legacy CARA sample CDM, in GCRF.

    Parameters
    ----------
    path
        The CDM (KVN).
    known
        Whether each object's covariance is used; ``False`` gives ``None``,
        as when a test passes a zero covariance.

    Raises
    ------
    CaraFormatError
        If ``TCA``, a state, a covariance entry or a supported ``REF_FRAME``
        is missing.
    """
    lines = path.read_text(encoding="latin-1").splitlines()
    starts = [i for i, line in enumerate(lines) if _OBJECT.match(line)]
    if len(starts) != 2:
        raise CaraFormatError(f"{path}: expected 2 object sections, got {len(starts)}")
    header = _fields(lines[: starts[0]])
    if "TCA" not in header:
        raise CaraFormatError(f"{path}: no TCA")
    tca = Time(header["TCA"], scale="utc")
    sections = (lines[starts[0] : starts[1]], lines[starts[1] :])
    first, second = gcrf_object_states(
        [
            _reported(_fields(section), tca, path, known=use)
            for section, use in zip(sections, known, strict=True)
        ]
    )
    return first, second
