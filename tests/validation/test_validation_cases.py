import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from builders import DIAGONAL_COVARIANCE

from keepout.errors import CaraFormatError
from keepout.sources import cara
from keepout.validation.cases import (
    alfano_objects,
    load_alfano_cases,
    load_non_pd_case,
    load_pc_method_cases,
    similar_to_3d_case,
)

B1_KVN = Path(__file__).parents[1] / "cdm" / "data" / "b1_example.kvn"


def alfano_rows(asymmetry: float = 0.0) -> list[list[Any]]:
    """A 36-row sheet in the layout of CARA's ``CaseN data at TCA.xls``."""
    rows: list[list[Any]] = [[""] * 6 for _ in range(36)]
    rows[4][:3] = [7000e3, 0.0, 0.0]
    rows[8][:3] = [0.0, 7500.0, 0.0]
    rows[22][:3] = [7000e3 + 10.0, 0.0, 0.0]
    rows[26][:3] = [0.0, 0.0, 7500.0]
    for first in (12, 30):
        matrix = DIAGONAL_COVARIANCE.copy()
        matrix[0, 1] = asymmetry
        for offset in range(6):
            rows[first + offset] = list(matrix[offset])
    return rows


def test_alfano_sheet_gives_both_objects() -> None:
    primary, secondary = alfano_objects(alfano_rows())
    np.testing.assert_array_equal(primary.state.r, [7000e3, 0.0, 0.0])
    np.testing.assert_array_equal(secondary.state.v, [0.0, 0.0, 7500.0])
    assert primary.covariance.matrix[1, 1] == 400.0
    assert primary.covariance.repairs == secondary.covariance.repairs == ()


def test_alfano_asymmetric_covariance_is_repaired_with_a_diagnostic() -> None:
    primary, secondary = alfano_objects(alfano_rows(asymmetry=1e-3))
    for item in (primary, secondary):
        (repair,) = item.covariance.repairs
        assert repair.startswith("symmetrised covariance")


def test_alfano_sheet_with_the_wrong_layout_is_rejected() -> None:
    with pytest.raises(CaraFormatError, match="unexpected Alfano sheet layout"):
        alfano_objects(alfano_rows()[:20])


def test_transcribed_similar_to_3d_case_is_well_formed() -> None:
    case = similar_to_3d_case()
    assert case.conjunction.hard_body_radius == 20.0
    assert case.reference == 1.0281653e-02
    assert case.conjunction.diagnostics == ()


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"HBR_m": 10.0, "Pc2D": 1e-4}, "no column 'Conjunction_ID'"),
        ({"Conjunction_ID": "unknown", "HBR_m": 10.0}, "no pinned CDM for"),
    ],
)
def test_malformed_pc_method_table_is_a_format_error(
    monkeypatch: pytest.MonkeyPatch, row: dict[str, Any], message: str
) -> None:
    monkeypatch.setattr(cara, "fetch", lambda file, cache=None: file.path)
    monkeypatch.setattr(cara, "load_xlsx", lambda path: [row])
    with pytest.raises(CaraFormatError, match=message):
        load_pc_method_cases()


def offline_alfano(
    monkeypatch: pytest.MonkeyPatch, results: list[dict[str, Any]]
) -> None:
    """Serve ``results`` as the Alfano results table and one sheet per case."""
    monkeypatch.setattr(cara, "fetch", lambda file, cache=None: Path(file.path))
    monkeypatch.setattr(cara, "load_csv", lambda path: results)
    monkeypatch.setattr(cara, "load_xls_rows", lambda path: alfano_rows())


def alfano_result(case: int, reference: Any = 1e-4) -> dict[str, Any]:
    return {"Case": case, "CARA_Pc2D": reference, "HBR": 10.0, "A09_PcLinear100": 2e-4}


def test_alfano_cases_are_built_from_the_results_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    offline_alfano(monkeypatch, [alfano_result(n) for n in range(1, 13)])
    cases = load_alfano_cases()
    assert [c.case_id for c in cases] == [str(n) for n in range(1, 13)]
    assert cases[0].conjunction.hard_body_radius == 10.0
    assert (cases[0].reference, cases[0].paper) == (1e-4, 2e-4)


def test_alfano_case_without_a_cara_value_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = [alfano_result(n) for n in range(1, 12)] + [alfano_result(12, math.nan)]
    offline_alfano(monkeypatch, results)
    assert [c.case_id for c in load_alfano_cases()] == [str(n) for n in range(1, 12)]


@pytest.mark.parametrize(
    ("results", "message"),
    [
        ([alfano_result(n) for n in range(2, 13)], "no row for case 1"),
        ([alfano_result(1, "n/a")], "expected a number, got 'n/a'"),
        ([{"Case": 1}], "no column 'CARA_Pc2D'"),
    ],
)
def test_malformed_alfano_results_are_a_format_error(
    monkeypatch: pytest.MonkeyPatch, results: list[dict[str, Any]], message: str
) -> None:
    offline_alfano(monkeypatch, results)
    with pytest.raises(CaraFormatError, match=message):
        load_alfano_cases()


def test_non_pd_case_uses_cara_reference_and_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cara, "fetch", lambda file, cache=None: B1_KVN)
    case = load_non_pd_case()
    assert case.conjunction.hard_body_radius == 20.0
    assert (case.reference, case.absolute_tolerance) == (0.0, 1e-10)
