import dataclasses

import pytest

from keepout.validation.cases import (
    ValidationCase,
    load_alfano_monte_carlo_cases,
    load_all,
    load_pc_method_monte_carlo_cases,
)
from keepout.validation.evaluation import Outcome, evaluate

pytestmark = pytest.mark.validation

QUICK_MONTE_CARLO = {
    "Alfano (2009), Monte Carlo": "3",
    "CARA Pc method test set, Monte Carlo": "000025994_conj_000037558_20210324_151047",
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
    return evaluate(c for c in cases if "hall" not in c.methods)


def test_every_case_is_loaded(cases: list[ValidationCase]) -> None:
    suites = [case.suite for case in cases]
    assert suites.count("Alfano (2009)") == 11
    assert suites.count("Alfano (2009), 3-D") == 12
    assert suites.count("CARA Pc method test set") == 53
    assert suites.count("CARA Pc method test set, 3-D") == 53
    assert suites.count("CARA unit tests") == 5
    assert suites.count("CARA unit tests, maximum Pc") == 6


def test_every_method_meets_its_criterion(outcomes: list[Outcome]) -> None:
    failed = [
        f"{o.case.suite} {o.case.case_id} {o.method}: {o.value} vs {o.case.reference}"
        for o in outcomes
        if not o.passed
    ]
    assert failed == []


def test_non_pd_covariance_is_repaired_and_reported(
    cases: list[ValidationCase],
) -> None:
    (case,) = [c for c in cases if c.case_id == "non-PD covariance CDM"]
    (diagnostic,) = case.conjunction.diagnostics
    assert diagnostic.startswith("secondary (")
    assert "clipped" in diagnostic


def test_slow_alfano_encounters_carry_the_2d_warning(outcomes: list[Outcome]) -> None:
    warned = {
        o.case.case_id
        for o in outcomes
        if o.case.suite == "Alfano (2009)"
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
            *load_alfano_monte_carlo_cases(),
            *load_pc_method_monte_carlo_cases(),
        )
        if case.case_id.startswith(QUICK_MONTE_CARLO.get(case.suite, "-"))
    ]
    assert len(chosen) == 2
    failed = [o.case.case_id for o in evaluate(chosen) if not o.passed]
    assert failed == []
