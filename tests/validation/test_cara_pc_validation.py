import dataclasses
from collections import Counter

import pytest

from keepout.validation.cara_inputs import (
    load_alfano_inputs,
    load_alfano_windows,
    load_pc_method_inputs,
)
from keepout.validation.cases import (
    ALFANO,
    ALFANO_3D,
    ALFANO_MONTE_CARLO,
    MAX_PC_UNIT_TESTS,
    PC_METHOD_SET,
    PC_METHOD_SET_3D,
    PC_METHOD_SET_MONTE_CARLO,
    UNIT_TESTS,
    ValidationCase,
    alfano_monte_carlo_cases,
    load_all,
    pc_method_monte_carlo_cases,
)
from keepout.validation.evaluation import Outcome, evaluate

pytestmark = pytest.mark.validation

QUICK_MONTE_CARLO = {
    ALFANO_MONTE_CARLO: "3",
    PC_METHOD_SET_MONTE_CARLO: "000025994_conj_000037558_20210324_151047",
}
"""One fast-encounter case per Monte Carlo suite, run with few trials on
every push; tools/validation_report.py --monte-carlo runs them all nightly."""


@pytest.fixture(scope="module")
def cases() -> list[ValidationCase]:
    return load_all()


@pytest.fixture(scope="module")
def outcomes(cases: list[ValidationCase]) -> list[Outcome]:
    # Hall 3D-Nc takes minutes over every case; tools/validation_report.py
    # --check (same CI job) judges it. Here: 2-D and maximum Pc.
    return evaluate(c for c in cases if "hall3d" not in c.methods)


def test_every_case_is_loaded(cases: list[ValidationCase]) -> None:
    assert Counter(case.suite for case in cases) == {
        ALFANO: 11,
        ALFANO_3D: 12,
        PC_METHOD_SET: 53,
        PC_METHOD_SET_3D: 53,
        UNIT_TESTS: 5,
        MAX_PC_UNIT_TESTS: 6,
    }


def test_every_method_meets_its_criterion(outcomes: list[Outcome]) -> None:
    failed = [
        f"{o.case.suite} {o.case.case_id} {o.result.method}: {o.result.value} "
        f"vs {o.case.reference}"
        for o in outcomes
        if not o.passed
    ]
    assert failed == []


def test_non_pd_covariance_is_repaired_and_reported(
    cases: list[ValidationCase],
) -> None:
    (case,) = [c for c in cases if c.case_id == "non-PD covariance CDM"]
    position, velocity = case.conjunction.diagnostics
    assert position.startswith("secondary (")
    assert "clipped 1 negative position-block eigenvalue" in position
    # Clipping the position block leaves the conditional velocity covariance
    # slightly negative too; it is repaired and reported as well.
    assert velocity.startswith("secondary (")
    assert "velocity covariance conditional on position" in velocity


def test_slow_alfano_encounters_carry_the_2d_warning(outcomes: list[Outcome]) -> None:
    warned = {
        o.case.case_id
        for o in outcomes
        if o.case.suite == ALFANO
        and any(
            d.startswith("2-D Pc assumptions questionable")
            for d in o.result.diagnostics
        )
    }
    # Case 3 is fast; case 5 lasts under 1% of an orbit (2-D ~ 3-D there).
    assert warned == {str(n) for n in (1, 2, 4, 6, 7, 8, 9, 10, 11)}


def test_quick_monte_carlo_contains_the_cara_value() -> None:
    chosen = [
        dataclasses.replace(case, options={**case.options, "samples": 20_000})
        for case in (
            *alfano_monte_carlo_cases(load_alfano_inputs(), load_alfano_windows()),
            *pc_method_monte_carlo_cases(load_pc_method_inputs()),
        )
        if case.case_id.startswith(QUICK_MONTE_CARLO.get(case.suite, "-"))
    ]
    assert len(chosen) == 2
    failed = [o.case.case_id for o in evaluate(chosen) if not o.passed]
    assert failed == []
