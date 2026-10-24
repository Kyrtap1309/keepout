"""NASA CARA's reference inputs read from its pinned files: the Alfano
(2009) sheets and results and the Pc method test set's table and CDMs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, SupportsFloat

import numpy as np
import numpy.typing as npt
from astropy.time import Time

from keepout import cdm
from keepout._types import FloatArray
from keepout.conjunction import Conjunction
from keepout.covariance import Covariance, make_covariance
from keepout.errors import CaraFormatError
from keepout.objects import ObjectState
from keepout.sources import cara
from keepout.sources.cara import Cell
from keepout.state import StateVector

__all__ = [
    "NOMINAL_EPOCH",
    "AlfanoInput",
    "PcMethodInput",
    "alfano_objects",
    "column_number",
    "column_value",
    "load_alfano_inputs",
    "load_alfano_windows",
    "load_pc_method_inputs",
]

NOMINAL_EPOCH: Final = Time("2000-01-01T00:00:00", scale="utc")
"""Nominal epoch of the cases whose states have no calendar date (the Alfano
sheets and CARA's inline unit-test cases): inertial vectors at TCA; no Pc
depends on it."""


@dataclass(frozen=True, eq=False, slots=True)
class AlfanoInput:
    """One Alfano (2009) case as CARA gives it.

    Parameters
    ----------
    number
        Case number, 1-12.
    conjunction
        The conjunction of its ``.xls`` sheet, with the HBR of ``results``.
    results
        Its row of CARA's results table.
    """

    number: int
    conjunction: Conjunction
    results: Mapping[str, Cell]


@dataclass(frozen=True, eq=False, slots=True)
class PcMethodInput:
    """One conjunction of the CARA Pc method test set.

    Parameters
    ----------
    conjunction_id
        CARA's identifier.
    conjunction
        The conjunction of its CDM, with the HBR of ``row``.
    row
        Its row of CARA's results table.
    """

    conjunction_id: str
    conjunction: Conjunction
    row: Mapping[str, Cell]


def load_alfano_inputs(cache: Path | None = None) -> list[AlfanoInput]:
    """Every Alfano (2009) case: CARA's results table and the ``.xls`` sheets.

    Raises
    ------
    CaraFormatError
        If the results table has no row for a case.
    """
    results = {
        int(column_number(row, "Case")): row
        for row in cara.load_csv(cara.fetch(cara.ALFANO_2009_RESULTS, cache=cache))
    }
    inputs: list[AlfanoInput] = []
    for number, file in enumerate(cara.ALFANO_2009_TCA, start=1):
        if number not in results:
            raise CaraFormatError(f"Alfano results have no row for case {number}")
        result = results[number]
        primary, secondary = alfano_objects(
            cara.load_xls_rows(cara.fetch(file, cache=cache))
        )
        conjunction = Conjunction(primary, secondary, column_number(result, "HBR"))
        inputs.append(AlfanoInput(number, conjunction, result))
    return inputs


def load_alfano_windows(cache: Path | None = None) -> dict[int, float]:
    """Half-width [s] of each Alfano case's window, by case number:
    ``A09_FinalTime_s`` of CARA's supplementary table."""
    return {
        int(column_number(row, "CaseNumber")): column_number(row, "A09_FinalTime_s")
        for row in cara.load_xlsx(cara.fetch(cara.ALFANO_2009_SUPPLEMENT, cache=cache))
    }


def load_pc_method_inputs(cache: Path | None = None) -> list[PcMethodInput]:
    """Every conjunction of the CARA Pc method test set: table row and CDM.

    Raises
    ------
    CaraFormatError
        If a conjunction of the table has no pinned CDM.
    """
    inputs: list[PcMethodInput] = []
    for row in cara.load_xlsx(cara.fetch(cara.PC_METHOD_TABLE, cache=cache)):
        conjunction_id = str(column_value(row, "Conjunction_ID"))
        if conjunction_id not in cara.PC_METHOD_CDMS:
            raise CaraFormatError(f"no pinned CDM for conjunction {conjunction_id}")
        path = cara.fetch(cara.PC_METHOD_CDMS[conjunction_id], cache=cache)
        conjunction = cdm.load(path).conjunction(
            combined_hard_body_radius=column_number(row, "HBR_m")
        )
        inputs.append(PcMethodInput(conjunction_id, conjunction, row))
    return inputs


def alfano_objects(rows: Sequence[Sequence[Cell]]) -> tuple[ObjectState, ObjectState]:
    """Primary and secondary from the rows of a ``CaseN data at TCA.xls`` sheet.

    Layout (1-based rows): primary position 5, velocity 9, covariance 13-18;
    secondary position 23, velocity 27, covariance 31-36. ECI, m, m/s.
    Covariances are repaired (symmetrised) if needed; see
    :attr:`~keepout.covariance.Covariance.repairs`.

    Raises
    ------
    CaraFormatError
        If the sheet does not have that layout.
    """
    try:
        pieces = [
            (_vector(rows, r), _vector(rows, v), _matrix(rows, c))
            for r, v, c in ((5, 9, 13), (23, 27, 31))
        ]
    except (IndexError, ValueError) as exc:
        raise CaraFormatError(f"unexpected Alfano sheet layout: {exc}") from exc
    primary, secondary = (
        _gcrf_object(name, r, v, make_covariance(matrix, "GCRF", NOMINAL_EPOCH))
        for name, (r, v, matrix) in zip(("PRIMARY", "SECONDARY"), pieces, strict=True)
    )
    return primary, secondary


def _vector(rows: Sequence[Sequence[Cell]], row: int) -> FloatArray:
    return np.array(rows[row - 1][:3], dtype=np.float64)


def _matrix(rows: Sequence[Sequence[Cell]], first_row: int) -> FloatArray:
    return np.array(
        [rows[index - 1][:6] for index in range(first_row, first_row + 6)],
        dtype=np.float64,
    )


def _gcrf_object(
    name: str, r: npt.ArrayLike, v: npt.ArrayLike, covariance: Covariance
) -> ObjectState:
    """An object at :data:`NOMINAL_EPOCH`."""
    return ObjectState(name, StateVector(NOMINAL_EPOCH, r, v, "GCRF"), covariance)


def column_value(row: Mapping[str, Cell], name: str) -> Cell:
    """The cell of column ``name`` of a CARA table row.

    Raises
    ------
    CaraFormatError
        If the table has no such column.
    """
    try:
        return row[name]
    except KeyError as exc:
        raise CaraFormatError(f"CARA table has no column {name!r}") from exc


def column_number(row: Mapping[str, Cell], name: str) -> float:
    """The number in column ``name`` of a CARA table row.

    Raises
    ------
    CaraFormatError
        If the table has no such column or the cell is not a number.
    """
    value = column_value(row, name)
    error = f"column {name!r}: expected a number, got {value!r}"
    if not isinstance(value, str | SupportsFloat):
        raise CaraFormatError(error)
    try:
        return float(value)
    except ValueError as exc:
        raise CaraFormatError(error) from exc
