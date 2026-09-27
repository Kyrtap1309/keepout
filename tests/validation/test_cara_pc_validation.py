import pytest

from keepout.validation.cases import ValidationCase, load_all
from keepout.validation.evaluation import Outcome, evaluate

pytestmark = pytest.mark.validation


@pytest.fixture(scope="module")
def cases() -> list[ValidationCase]:
    return load_all()


@pytest.fixture(scope="module")
def outcomes(cases: list[ValidationCase]) -> list[Outcome]:
    return evaluate(cases)


def test_every_case_is_loaded(cases: list[ValidationCase]) -> None:
    suites = [case.suite for case in cases]
    assert suites.count("Alfano (2009)") == 11
    assert suites.count("CARA Pc method test set") == 53
    assert suites.count("CARA unit tests") == 2


def test_every_method_meets_the_spec_tolerance(outcomes: list[Outcome]) -> None:
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
