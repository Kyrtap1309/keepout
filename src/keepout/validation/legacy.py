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
from pathlib import Path

import numpy as np
from astropy.time import Time

from keepout._types import REF_FRAMES, StateVector
from keepout.cdm.keywords import covariance_entries
from keepout.covariance import make_covariance
from keepout.errors import CaraFormatError
from keepout.frames import to_gcrf
from keepout.objects import ObjectState

_LINE = re.compile(r"^\s*([A-Z][A-Z0-9_]*)\s*=\s*(\S+)")
_STATE = ("X", "Y", "Z", "X_DOT", "Y_DOT", "Z_DOT")


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
    starts = [i for i, line in enumerate(lines) if re.match(r"^\s*OBJECT\s*=", line)]
    if len(starts) != 2:
        raise CaraFormatError(f"{path}: expected 2 object sections, got {len(starts)}")
    header = _fields(lines[: starts[0]])
    if "TCA" not in header:
        raise CaraFormatError(f"{path}: no TCA")
    tca = Time(header["TCA"], scale="utc")
    objects = []
    for section, use in zip(
        (lines[starts[0] : starts[1]], lines[starts[1] :]), known, strict=True
    ):
        fields = _fields(section)
        frame = fields.get("REF_FRAME")
        if frame not in REF_FRAMES:
            raise CaraFormatError(f"{path}: unsupported REF_FRAME {frame!r}")
        values = [1e3 * _number(fields, k, path) for k in _STATE]  # km, km/s
        state = to_gcrf(StateVector(tca, values[:3], values[3:], frame))
        covariance = None
        if use:
            matrix = np.zeros((6, 6))
            for name, row, column in covariance_entries(6):
                matrix[row, column] = matrix[column, row] = _number(fields, name, path)
            covariance = make_covariance(matrix, "RTN", tca).in_gcrf(state)
        designator = fields.get("OBJECT_DESIGNATOR", "?")
        objects.append(ObjectState(designator, state, covariance))
    return objects[0], objects[1]
