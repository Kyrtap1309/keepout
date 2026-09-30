import pytest
from builders import case_with_reference

from keepout.pc.result import PcResult
from keepout.validation.criteria import HALL, MONTE_CARLO, TWO_D, Criterion
from keepout.validation.evaluation import Verdict, evaluate, judge

SMALL = TWO_D.small
RELATIVE = TWO_D.relative or 0.0
ABSOLUTE = TWO_D.absolute or 0.0


def result(value: float, **details: float) -> PcResult:
    return PcResult(value, "foster", (), details)


def test_relative_criterion_at_and_above_the_small_pc_threshold() -> None:
    case = case_with_reference(SMALL)
    inside = SMALL * (1 + 0.5 * RELATIVE)
    assert judge(case, result(inside)) == pytest.approx(
        Verdict(0.5 * RELATIVE, True, True)
    )
    assert not judge(case, result(SMALL * (1 + 2 * RELATIVE))).passed


def test_absolute_criterion_below_the_small_pc_threshold() -> None:
    case = case_with_reference(0.5 * SMALL)
    error, relative, passed = judge(case, result(0.5 * SMALL + 0.5 * ABSOLUTE))
    assert (relative, passed) == (False, True)
    assert error == pytest.approx(0.5 * ABSOLUTE)
    assert not judge(case, result(0.5 * SMALL + 2 * ABSOLUTE)).passed


def test_case_criterion_replaces_the_default() -> None:
    case = case_with_reference(0.0, absolute_tolerance=1e-10)
    assert judge(case, result(5e-11)) == Verdict(5e-11, False, True)
    assert not judge(case, result(2e-10)).passed


def test_interval_criterion_uses_the_result_interval() -> None:
    case = case_with_reference(0.02)
    case = type(case)(
        case.suite, case.case_id, case.conjunction, 0.02, criterion=MONTE_CARLO
    )
    inside = result(0.021, ci_low=0.019, ci_high=0.023)
    assert judge(case, inside) == pytest.approx(Verdict(0.05, True, True))
    assert not judge(case, result(0.03, ci_low=0.025, ci_high=0.035)).passed


@pytest.mark.parametrize(
    ("criterion", "text"),
    [
        (TWO_D, "relative error at most 1e-06 (absolute error at most 1e-15"),
        (HALL, "relative error at most 0.001"),
        (Criterion(absolute=5e-7), "absolute error at most 5e-07"),
        (MONTE_CARLO, "within keepout's confidence interval"),
    ],
)
def test_criteria_describe_themselves(criterion: Criterion, text: str) -> None:
    assert text in criterion.describe()


def test_evaluate_runs_the_case_methods_and_keeps_the_full_result() -> None:
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
