"""Validation cases built from NASA CARA reference data."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from astropy.time import Time

from keepout import cdm
from keepout._types import FloatArray, StateVector
from keepout.conjunction import Conjunction
from keepout.covariance import Covariance, make_covariance
from keepout.errors import CaraFormatError
from keepout.objects import ObjectState
from keepout.sources import cara

ALFANO_EPOCH = Time("2000-01-01T00:00:00", scale="utc")
"""Nominal epoch for the Alfano cases; their states are inertial vectors at
TCA with no calendar date, and 2-D Pc does not depend on it."""


@dataclass(frozen=True, eq=False)
class ValidationCase:
    """One conjunction with a CARA reference Pc.

    Parameters
    ----------
    suite
        Name of the case collection, e.g. ``"Alfano (2009)"``.
    case_id
        Identifier within the suite.
    conjunction
        The conjunction to evaluate.
    reference
        CARA's 2-D Pc.
    paper
        Value published with the test case (Alfano 2009), if any.
    absolute_tolerance
        Overrides the default criterion with ``|Pc - reference| <= tolerance``.
    """

    suite: str
    case_id: str
    conjunction: Conjunction
    reference: float
    paper: float | None = None
    absolute_tolerance: float | None = None


def _vector(rows: Sequence[Sequence[Any]], row: int) -> FloatArray:
    return np.array(rows[row - 1][:3], dtype=np.float64)


def _matrix(rows: Sequence[Sequence[Any]], first_row: int) -> FloatArray:
    return np.array(
        [rows[index - 1][:6] for index in range(first_row, first_row + 6)],
        dtype=np.float64,
    )


def _gcrf_object(
    name: str, r: npt.ArrayLike, v: npt.ArrayLike, covariance: Covariance
) -> ObjectState:
    """An object at :data:`ALFANO_EPOCH`."""
    return ObjectState(name, StateVector(ALFANO_EPOCH, r, v, "GCRF"), covariance)


def alfano_objects(rows: Sequence[Sequence[Any]]) -> tuple[ObjectState, ObjectState]:
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
        _gcrf_object(name, r, v, make_covariance(matrix, "GCRF", ALFANO_EPOCH))
        for name, (r, v, matrix) in zip(("PRIMARY", "SECONDARY"), pieces, strict=True)
    )
    return primary, secondary


def _field(row: Mapping[str, Any], name: str) -> Any:
    try:
        return row[name]
    except KeyError as exc:
        raise CaraFormatError(f"CARA table has no column {name!r}") from exc


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise CaraFormatError(f"expected a number, got {value!r}") from exc


def load_alfano_cases(cache: Path | None = None) -> list[ValidationCase]:
    """Alfano (2009) cases with a CARA 2-D reference (cases 1-11).

    Case 12 has zero relative velocity; CARA gives no 2-D value for it.
    """
    results = {
        int(_float(_field(row, "Case"))): row
        for row in cara.load_csv(cara.fetch(cara.ALFANO_2009_RESULTS, cache=cache))
    }
    cases: list[ValidationCase] = []
    for number, file in enumerate(cara.ALFANO_2009_TCA, start=1):
        if number not in results:
            raise CaraFormatError(f"Alfano results have no row for case {number}")
        result = results[number]
        reference = _float(_field(result, "CARA_Pc2D"))
        if math.isnan(reference):
            continue
        rows = cara.load_xls_rows(cara.fetch(file, cache=cache))
        primary, secondary = alfano_objects(rows)
        radius = _float(_field(result, "HBR"))
        cases.append(
            ValidationCase(
                "Alfano (2009)",
                str(number),
                Conjunction(primary, secondary, radius),
                reference,
                paper=_float(_field(result, "A09_PcLinear100")),
            )
        )
    return cases


def load_pc_method_cases(cache: Path | None = None) -> list[ValidationCase]:
    """The 53 real conjunctions of the CARA Pc method test set.

    The reference is CARA's ``Pc2D``: states moved to the true TCA before
    integrating. keepout projects the relative position onto the encounter
    plane, which is the same straight-line closest approach.
    """
    table = cara.load_xlsx(cara.fetch(cara.PC_METHOD_TABLE, cache=cache))
    cases: list[ValidationCase] = []
    for row in table:
        conjunction_id = str(_field(row, "Conjunction_ID"))
        if conjunction_id not in cara.PC_METHOD_CDMS:
            raise CaraFormatError(f"no pinned CDM for conjunction {conjunction_id}")
        path = cara.fetch(cara.PC_METHOD_CDMS[conjunction_id], cache=cache)
        conjunction = Conjunction.from_cdm(
            cdm.load(path), hard_body_radius=_float(_field(row, "HBR_m"))
        )
        cases.append(
            ValidationCase(
                "CARA Pc method test set",
                conjunction_id,
                conjunction,
                _float(_field(row, "Pc2D")),
            )
        )
    return cases


def load_non_pd_case(cache: Path | None = None) -> ValidationCase:
    """CDM with a non-positive-semi-definite covariance: Pc 0 within 1e-10.

    Reference and tolerance as in CARA's ``Pc2D_Foster_UnitTest.m``
    (``testNonPositiveDefinite``, HBR 20 m).
    """
    path = cara.fetch(cara.NON_PD_CDM, cache=cache)
    conjunction = Conjunction.from_cdm(cdm.load(path), hard_body_radius=20.0)
    return ValidationCase(
        "CARA unit tests",
        "non-PD covariance CDM",
        conjunction,
        0.0,
        absolute_tolerance=1e-10,
    )


def similar_to_3d_case() -> ValidationCase:
    """Inline case ``testSimilarTo3D`` of CARA's ``PcElrod_UnitTest.m``.

    Numbers transcribed from ``DistributedMatlab/ProbabilityOfCollision/
    UnitTests/PcElrod_UnitTest.m`` lines 208-232 at the pinned commit
    (ECI, m, m/s, HBR 20 m); the reference has 8 significant digits.
    """
    position_covariances = (
        [
            [4.976545641899520e04, 5.787130862568278e04, 3.370410320935015e03],
            [5.787130862568278e04, 6.730377643610841e04, 3.926542932121541e03],
            [3.370410320935015e03, 3.926542932121541e03, 2.461403197221289e02],
        ],
        [
            [4.246862551076427e04, 2.066374367781032e05, -5.011108933888592e03],
            [2.066374367781032e05, 1.005854717283451e06, -2.434876491048039e04],
            [-5.011108933888592e03, -2.434876491048039e04, 6.131274993037449e02],
        ],
    )
    states = (
        (
            [-9.841950433215101e05, 3.932342044549424e05, 6.991223682230414e06],
            [4.883696742000000e03, 5.689086045000000e03, 3.665361590000000e02],
        ),
        (
            [-9.839696058965517e05, 3.936845951174244e05, 6.991219291625473e06],
            [1.509562687000000e03, 7.372938617000000e03, -1.492509430000000e02],
        ),
    )
    primary, secondary = (
        _gcrf_object(
            name, r, v, Covariance.from_position(position, "GCRF", ALFANO_EPOCH)
        )
        for name, (r, v), position in zip(
            ("PRIMARY", "SECONDARY"), states, position_covariances, strict=True
        )
    )
    return ValidationCase(
        "CARA unit tests",
        "Elrod testSimilarTo3D",
        Conjunction(primary, secondary, 20.0),
        1.0281653e-02,
    )


def load_all(cache: Path | None = None) -> list[ValidationCase]:
    """Every validation case, downloading CARA files as needed."""
    cara.prefetch(cara.MANIFEST, cache=cache)
    return [
        *load_alfano_cases(cache),
        *load_pc_method_cases(cache),
        load_non_pd_case(cache),
        similar_to_3d_case(),
    ]
