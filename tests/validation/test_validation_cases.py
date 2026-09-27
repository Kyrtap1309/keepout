from typing import Any

import numpy as np
import pytest

from keepout.errors import CaraFormatError
from keepout.validation.cases import alfano_objects, similar_to_3d_case


def alfano_rows(asymmetry: float = 0.0) -> list[list[Any]]:
    """A 36-row sheet in the layout of CARA's ``CaseN data at TCA.xls``."""
    rows: list[list[Any]] = [[""] * 6 for _ in range(36)]
    rows[4][:3] = [7000e3, 0.0, 0.0]
    rows[8][:3] = [0.0, 7500.0, 0.0]
    rows[22][:3] = [7000e3 + 10.0, 0.0, 0.0]
    rows[26][:3] = [0.0, 0.0, 7500.0]
    for first in (12, 30):
        matrix = np.diag([100.0, 400.0, 25.0, 1e-2, 4e-2, 1e-2])
        matrix[0, 1] = asymmetry
        for offset in range(6):
            rows[first + offset] = list(matrix[offset])
    return rows


def test_alfano_sheet_gives_both_objects() -> None:
    primary, secondary, diagnostics = alfano_objects(alfano_rows())
    np.testing.assert_array_equal(primary.state.r, [7000e3, 0.0, 0.0])
    np.testing.assert_array_equal(secondary.state.v, [0.0, 0.0, 7500.0])
    assert primary.covariance.matrix[1, 1] == 400.0
    assert diagnostics == ()


def test_alfano_asymmetric_covariance_is_repaired_with_a_diagnostic() -> None:
    _, _, diagnostics = alfano_objects(alfano_rows(asymmetry=1e-3))
    assert len(diagnostics) == 2
    assert diagnostics[0].startswith("PRIMARY: symmetrised covariance")


def test_alfano_sheet_with_the_wrong_layout_is_rejected() -> None:
    with pytest.raises(CaraFormatError, match="unexpected Alfano sheet layout"):
        alfano_objects(alfano_rows()[:20])


def test_transcribed_similar_to_3d_case_is_well_formed() -> None:
    case = similar_to_3d_case()
    assert case.conjunction.hard_body_radius == 20.0
    assert case.reference == 1.0281653e-02
    assert case.conjunction.diagnostics == ()
