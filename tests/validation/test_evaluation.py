import pytest
from builders import case_with_reference

from keepout.validation.evaluation import (
    ABSOLUTE_TOLERANCE,
    RELATIVE_TOLERANCE,
    SMALL_PC,
    Verdict,
    evaluate,
    judge,
)


def test_relative_criterion_at_and_above_the_small_pc_threshold() -> None:
    case = case_with_reference(SMALL_PC)
    inside = SMALL_PC * (1 + 0.5 * RELATIVE_TOLERANCE)
    assert judge(case, inside) == pytest.approx(
        Verdict(0.5 * RELATIVE_TOLERANCE, True, True)
    )
    assert not judge(case, SMALL_PC * (1 + 2 * RELATIVE_TOLERANCE)).passed


def test_absolute_criterion_below_the_small_pc_threshold() -> None:
    case = case_with_reference(0.5 * SMALL_PC)
    error, relative, passed = judge(case, 0.5 * SMALL_PC + 0.5 * ABSOLUTE_TOLERANCE)
    assert (relative, passed) == (False, True)
    assert error == pytest.approx(0.5 * ABSOLUTE_TOLERANCE)
    assert not judge(case, 0.5 * SMALL_PC + 2 * ABSOLUTE_TOLERANCE).passed


def test_case_tolerance_overrides_the_default() -> None:
    case = case_with_reference(0.0, absolute_tolerance=1e-10)
    assert judge(case, 5e-11) == Verdict(5e-11, False, True)
    assert not judge(case, 2e-10).passed


def test_evaluate_runs_every_method_and_keeps_the_full_result() -> None:
    case = case_with_reference(0.0, absolute_tolerance=1e-10)
    outcomes = evaluate([case])
    assert [o.method for o in outcomes] == ["foster", "elrod"]
    assert all(o.case is case and o.passed for o in outcomes)
    assert all("encounter_duration_s" in o.result.details for o in outcomes)


def test_outcome_verdict_follows_its_value() -> None:
    (foster, _) = evaluate([case_with_reference(0.5)])
    assert foster.value == 0.0
    assert foster.verdict == Verdict(1.0, True, False)
    assert not foster.passed
