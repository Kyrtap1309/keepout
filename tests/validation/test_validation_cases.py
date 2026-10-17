import dataclasses
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from builders import (
    B1_KVN,
    DIAGONAL_COVARIANCE,
    LEO_R,
    LEO_V,
    case_with_reference,
    known,
    object_state,
)

from keepout.errors import CaraFormatError
from keepout.objects import ObjectState
from keepout.sources import cara
from keepout.validation import cases as cases_module
from keepout.validation.cases import (
    MC_MIN_PC,
    MC_SAMPLES,
    MC_SUITE_ALPHA,
    PC_METHOD_SET,
    alfano_3d_cases,
    alfano_cases,
    alfano_monte_carlo_cases,
    alfano_objects,
    elrod_similar_to_3d_case,
    hall_similar_to_2d_case,
    load_alfano_inputs,
    load_alfano_windows,
    load_hall_2d3d_cases,
    load_max_pc_cases,
    load_non_pd_case,
    load_pc_method_inputs,
    monte_carlo_options,
    pc_method_3d_cases,
    pc_method_cases,
    pc_method_monte_carlo_cases,
)
from keepout.validation.criteria import HALL, HALL_PC_SET, MONTE_CARLO
from keepout.validation.evaluation import evaluate


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
    assert known(primary.covariance).matrix[1, 1] == 400.0
    assert (
        known(primary.covariance).repairs == known(secondary.covariance).repairs == ()
    )


def test_alfano_asymmetric_covariance_is_repaired_with_a_diagnostic() -> None:
    primary, secondary = alfano_objects(alfano_rows(asymmetry=1e-3))
    for item in (primary, secondary):
        (repair,) = known(item.covariance).repairs
        assert repair.startswith("symmetrised covariance")


def test_alfano_sheet_with_the_wrong_layout_is_rejected() -> None:
    with pytest.raises(CaraFormatError, match="unexpected Alfano sheet layout"):
        alfano_objects(alfano_rows()[:20])


def test_transcribed_similar_to_3d_case_is_well_formed() -> None:
    case = elrod_similar_to_3d_case()
    assert case.conjunction.hard_body_radius == 20.0
    assert case.conjunction.diagnostics == ()
    # The transcribed states reproduce CARA's 8-digit reference.
    outcomes = evaluate([case])
    assert [outcome.result.method for outcome in outcomes] == ["foster", "elrod"]
    assert all(outcome.passed for outcome in outcomes)


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
        load_pc_method_inputs()


@pytest.fixture
def offline_b1(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve the B1 example CDM for every CARA file."""
    monkeypatch.setattr(cara, "fetch", lambda file, cache=None: B1_KVN)


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
    cases = alfano_cases(load_alfano_inputs())
    assert [c.case_id for c in cases] == [str(n) for n in range(1, 13)]
    assert cases[0].conjunction.hard_body_radius == 10.0
    assert (cases[0].reference, cases[0].paper) == (1e-4, 2e-4)


def test_alfano_case_without_a_cara_value_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = [alfano_result(n) for n in range(1, 12)] + [alfano_result(12, math.nan)]
    offline_alfano(monkeypatch, results)
    cases = alfano_cases(load_alfano_inputs())
    assert [c.case_id for c in cases] == [str(n) for n in range(1, 12)]


@pytest.mark.parametrize(
    ("results", "message"),
    [
        ([alfano_result(n) for n in range(2, 13)], "no row for case 1"),
        ([{"Case": n} for n in range(1, 13)], "no column 'HBR'"),
    ],
)
def test_malformed_alfano_results_are_a_format_error(
    monkeypatch: pytest.MonkeyPatch, results: list[dict[str, Any]], message: str
) -> None:
    offline_alfano(monkeypatch, results)
    with pytest.raises(CaraFormatError, match=message):
        load_alfano_inputs()


@pytest.mark.parametrize(
    ("results", "message"),
    [
        (
            [alfano_result(n, "n/a") for n in range(1, 13)],
            "column 'CARA_Pc2D': expected a number, got 'n/a'",
        ),
        (
            [
                {k: v for k, v in alfano_result(n).items() if k != "CARA_Pc2D"}
                for n in range(1, 13)
            ],
            "no column 'CARA_Pc2D'",
        ),
    ],
)
def test_malformed_alfano_reference_is_a_format_error(
    monkeypatch: pytest.MonkeyPatch, results: list[dict[str, Any]], message: str
) -> None:
    offline_alfano(monkeypatch, results)
    alfano = load_alfano_inputs()
    with pytest.raises(CaraFormatError, match=message):
        alfano_cases(alfano)


@pytest.mark.usefixtures("offline_b1")
def test_non_pd_case_uses_cara_reference_and_tolerance() -> None:
    case = load_non_pd_case()
    assert case.conjunction.hard_body_radius == 20.0
    assert (case.reference, case.criterion.absolute) == (0.0, 1e-10)


def offline_alfano_3d(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve 12 Alfano cases with 3-D and Monte Carlo references and windows."""
    results = [
        {
            "Case": n,
            "HBR": 10.0,
            "CARA_Pc3D": 1e-3 * n,
            "CARA_PcMC": 2e-3 * n,
            "CARA_PcMCLo95": 1e-3 * n,
            "CARA_PcMCHi95": 3e-3 * n,
        }
        for n in range(1, 13)
    ]
    windows = [{"CaseNumber": n, "A09_FinalTime_s": 100.0 * n} for n in range(1, 13)]
    monkeypatch.setattr(cara, "fetch", lambda file, cache=None: Path(file.path))
    monkeypatch.setattr(cara, "load_csv", lambda path: results)
    monkeypatch.setattr(cara, "load_xlsx", lambda path: windows)
    monkeypatch.setattr(cara, "load_xls_rows", lambda path: alfano_rows())


def test_alfano_3d_cases_use_the_supplementary_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    offline_alfano_3d(monkeypatch)
    cases = alfano_3d_cases(load_alfano_inputs(), load_alfano_windows())
    assert [c.case_id for c in cases] == [str(n) for n in range(1, 13)]
    ninth = cases[8]
    assert ninth.methods == ("hall3d",)
    assert ninth.criterion is HALL
    assert ninth.reference == pytest.approx(9e-3)
    assert ninth.options.get("window") == (-900.0, 900.0)


def test_alfano_monte_carlo_cases_carry_the_cara_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    offline_alfano_3d(monkeypatch)
    first = alfano_monte_carlo_cases(load_alfano_inputs(), load_alfano_windows())[0]
    assert first.methods == ("montecarlo",)
    assert first.criterion is MONTE_CARLO
    assert first.interval == (1e-3, 3e-3)
    assert first.options.get("window") == (-100.0, 100.0)
    assert first.options.get("samples") == 200_000  # 400 / 2e-3


def test_monte_carlo_options_are_bounded_and_reproducible() -> None:
    assert monte_carlo_options("s", "1", 0.5)["samples"] == MC_SAMPLES[0]
    assert monte_carlo_options("s", "1", 1e-5)["samples"] == MC_SAMPLES[1]
    assert monte_carlo_options("s", "1", 1e-3) == monte_carlo_options("s", "1", 1e-3)
    assert monte_carlo_options("s", "1", 1e-3) != monte_carlo_options("s", "2", 1e-3)


def test_a_case_refuses_options_its_methods_do_not_take() -> None:
    case = case_with_reference(1e-3)
    with pytest.raises(
        ValueError, match=r"test/case: foster takes no options \['window'\]"
    ):
        dataclasses.replace(case, options={"window": (-1.0, 1.0)})
    with pytest.raises(ValueError, match=r"hall3d takes no options \['seed'\]"):
        dataclasses.replace(
            case,
            methods=("hall3d", "montecarlo"),
            options={"window": (-1.0, 1.0), "seed": 1},
        )
    both = dataclasses.replace(
        case, methods=("hall3d", "montecarlo"), options={"window": (-1.0, 1.0)}
    )
    assert both.options == {"window": (-1.0, 1.0)}


def test_alfano_tables_must_cover_every_case(monkeypatch: pytest.MonkeyPatch) -> None:
    offline_alfano_3d(monkeypatch)
    monkeypatch.setattr(cara, "load_xlsx", lambda path: [])
    alfano = load_alfano_inputs()
    with pytest.raises(CaraFormatError, match="supplement has no row for case 1"):
        alfano_3d_cases(alfano, load_alfano_windows())


def test_pc_method_3d_and_monte_carlo_cases(
    monkeypatch: pytest.MonkeyPatch, offline_b1: None
) -> None:
    row = {
        "Conjunction_ID": "A",
        "HBR_m": 20.0,
        "Nc3D": 2e-4,
        "Pc3DCoppola": 1e-20,
        "PcSDMC": 3e-4,
        "PcSDMCLo": 2e-4,
        "PcSDMCHi": 4e-4,
    }
    rare = {**row, "Conjunction_ID": "B", "PcSDMC": 0.5 * MC_MIN_PC}
    monkeypatch.setattr(cara, "PC_METHOD_CDMS", {"A": B1_KVN, "B": B1_KVN})
    monkeypatch.setattr(cara, "load_xlsx", lambda path: [row, rare])
    pc_method = load_pc_method_inputs()
    three_d = pc_method_3d_cases(pc_method)
    assert [c.case_id for c in three_d] == ["A", "B"]
    assert three_d[0].criterion is HALL_PC_SET
    assert three_d[0].informative == ("Coppola 3-D Pc", 1e-20)
    (only,) = pc_method_monte_carlo_cases(pc_method)
    assert only.case_id == "A"
    assert only.interval == (2e-4, 4e-4)


def test_pc_method_2d_cases(monkeypatch: pytest.MonkeyPatch, offline_b1: None) -> None:
    row = {"Conjunction_ID": "A", "HBR_m": 20.0, "Pc2D": 3e-4}
    monkeypatch.setattr(cara, "PC_METHOD_CDMS", {"A": B1_KVN})
    monkeypatch.setattr(cara, "load_xlsx", lambda path: [row])
    (case,) = pc_method_cases(load_pc_method_inputs())
    assert (case.suite, case.case_id, case.reference) == (PC_METHOD_SET, "A", 3e-4)
    assert case.conjunction.hard_body_radius == 20.0
    assert case.methods == ("foster", "elrod")


@pytest.mark.usefixtures("offline_b1")
def test_hall_2d3d_cases_share_one_conjunction() -> None:
    two_d, three_d = load_hall_2d3d_cases()
    assert two_d.conjunction is three_d.conjunction
    assert two_d.conjunction.hard_body_radius == 20.0
    assert (two_d.reference, three_d.reference) == (2.2660816e-20, 1.9271201e-05)
    assert three_d.methods == ("hall3d",)
    assert three_d.criterion is HALL


def test_transcribed_hall_similar_to_2d_case_is_well_formed() -> None:
    case = hall_similar_to_2d_case()
    two_d = elrod_similar_to_3d_case().conjunction
    assert case.conjunction.diagnostics == ()
    # The transcribed 6x6 covariances reproduce CARA's reference to its 1e-3.
    [outcome] = evaluate([case])
    assert outcome.result.method == "hall3d"
    assert outcome.passed
    np.testing.assert_array_equal(
        case.conjunction.primary.state.r, two_d.primary.state.r
    )
    np.testing.assert_allclose(
        known(case.conjunction.secondary.covariance).position,
        known(two_d.secondary.covariance).position,
    )


def test_max_pc_cases_follow_cara_unit_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[bool, bool]] = []

    def objects(
        path: Path, *, known: tuple[bool, bool] = (True, True)
    ) -> tuple[ObjectState, ObjectState]:
        seen.append(known)
        primary = object_state(LEO_R, LEO_V, np.eye(3))
        secondary = object_state(LEO_R + np.array([30.0, 0, 0]), -LEO_V, np.eye(3))
        return primary, (
            secondary if known[1] else ObjectState("S", secondary.state, None)
        )

    monkeypatch.setattr(cara, "fetch", lambda file, cache=None: Path(file.path))
    monkeypatch.setattr(cases_module, "legacy_objects", objects)
    found = load_max_pc_cases()
    assert [c.methods for c in found] == [("frisbee",)] * 2 + [("dilution",)] * 4
    assert seen[0] == (True, False)
    assert found[0].criterion.absolute == 5e-7
    assert found[2].criterion.relative == 5e-3
    assert found[0].conjunction.unknown_covariance == ("secondary",)


def test_monte_carlo_intervals_share_the_suite_false_failure_chance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = dataclasses.replace(case_with_reference(1e-3), methods=("montecarlo",))
    monkeypatch.setattr(cara, "prefetch", lambda *_, **__: None)
    loaders = ("load_alfano_inputs", "load_alfano_windows", "load_pc_method_inputs")
    for loader in loaders:
        monkeypatch.setattr(cases_module, loader, lambda _: [])
    monkeypatch.setattr(cases_module, "alfano_monte_carlo_cases", lambda *_: [case] * 3)
    monkeypatch.setattr(cases_module, "pc_method_monte_carlo_cases", lambda _: [case])
    loaded = cases_module.load_monte_carlo_cases()
    assert [c.options.get("confidence") for c in loaded] == [
        1.0 - MC_SUITE_ALPHA / 4
    ] * 4
