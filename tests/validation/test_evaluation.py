import dataclasses

import pytest
from builders import case_with_reference

from keepout.pc.result import PcResult
from keepout.validation.cases import ValidationCase
from keepout.validation.criteria import HALL, MONTE_CARLO, TWO_D, Criterion
from keepout.validation.evaluation import (
    MethodSummary,
    Outcome,
    Verdict,
    evaluate,
    judge,
    run,
    summarise,
)

SMALL = TWO_D.small
RELATIVE = TWO_D.relative or 0.0
ABSOLUTE = TWO_D.absolute or 0.0


def result(value: float, **details: float) -> PcResult:
    return PcResult(value, "foster", (), details)


def test_relative_criterion_at_and_above_the_small_pc_threshold() -> None:
    case = case_with_reference(SMALL)
    inside = SMALL * (1 + 0.5 * RELATIVE)
    assert judge(case, result(inside)) == pytest.approx(
        Verdict(0.5 * RELATIVE, True, True), rel=1e-9, abs=0.0
    )
    assert not judge(case, result(SMALL * (1 + 2 * RELATIVE))).passed


def test_absolute_criterion_below_the_small_pc_threshold() -> None:
    case = case_with_reference(0.5 * SMALL)
    error, relative, passed = judge(case, result(0.5 * SMALL + 0.5 * ABSOLUTE))
    assert (relative, passed) == (False, True)
    assert error == pytest.approx(0.5 * ABSOLUTE, rel=1e-9, abs=0.0)
    assert not judge(case, result(0.5 * SMALL + 2 * ABSOLUTE)).passed


def test_case_criterion_replaces_the_default() -> None:
    case = case_with_reference(0.0, absolute_tolerance=1e-10)
    assert judge(case, result(5e-11)) == Verdict(5e-11, False, True)
    assert not judge(case, result(2e-10)).passed


def test_interval_criterion_uses_the_result_interval() -> None:
    case = dataclasses.replace(case_with_reference(0.02), criterion=MONTE_CARLO)
    inside = PcResult(0.021, "montecarlo", interval=(0.019, 0.023))
    assert judge(case, inside) == pytest.approx(Verdict(0.05, True, True))
    outside = PcResult(0.03, "montecarlo", interval=(0.025, 0.035))
    assert not judge(case, outside).passed


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
    assert [o.result.method for o in outcomes] == ["foster", "elrod"]
    assert all(o.case is case and o.passed for o in outcomes)
    assert all("encounter_duration_s" in o.result.details for o in outcomes)


def test_run_passes_the_case_options_to_its_method() -> None:
    case = dataclasses.replace(
        case_with_reference(0.0),
        methods=("hall3d", "montecarlo"),
        options={"window": (-1.0, 1.0)},
    )
    hall = run(case, "hall3d")
    assert (hall.details["window_start_s"], hall.details["window_end_s"]) == (-1, 1)
    only = dataclasses.replace(
        case, methods=("montecarlo",), options={"window": (-2.0, 2.0), "seed": 1}
    )
    assert run(only, "montecarlo").details["window_start_s"] == -2.0


def test_evaluate_in_processes_gives_the_same_outcomes_in_order() -> None:
    cases = [
        case_with_reference(0.0, absolute_tolerance=1e-10),
        case_with_reference(0.5),
    ]
    serial = evaluate(cases)
    parallel = evaluate(cases, workers=2)

    def summary(outcomes: list[Outcome]) -> list[tuple[object, ...]]:
        return [
            (o.case.reference, o.result.method, o.result.value, dict(o.result.details))
            for o in outcomes
        ]

    assert summary(parallel) == summary(serial)


def test_evaluate_needs_a_worker() -> None:
    with pytest.raises(ValueError, match="workers must be at least 1"):
        evaluate([], workers=0)


def test_outcome_verdict_follows_its_value() -> None:
    (foster, _) = evaluate([case_with_reference(0.5)])
    assert foster.result.value == 0.0
    assert foster.verdict == Verdict(1.0, True, False)
    assert not foster.passed


def test_summarise_groups_by_suite_then_method_in_order() -> None:
    def case(suite: str, reference: float, case_id: str) -> ValidationCase:
        return dataclasses.replace(
            case_with_reference(reference, case_id=case_id), suite=suite
        )

    outcomes = evaluate(
        [
            case("b", 1e-3, "relative"),
            case("a", 0.0, "absolute"),
            dataclasses.replace(
                case("b", 2e-3, "loose"), criterion=Criterion(relative=2.0)
            ),
        ]
    )
    assert summarise(outcomes) == (
        MethodSummary("b", "foster", 1, 2, 1.0),
        MethodSummary("b", "elrod", 1, 2, 1.0),
        MethodSummary("a", "foster", 1, 1, None),
        MethodSummary("a", "elrod", 1, 1, None),
    )
    assert summarise([]) == ()
