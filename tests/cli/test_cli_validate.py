import dataclasses
import json
from pathlib import Path

import pytest
from builders import case_with_reference
from cli_builders import keepout
from inline_snapshot import snapshot

from keepout.cli.output import ExitCode
from keepout.pc.result import PcResult
from keepout.sources.cara import CARA_COMMIT
from keepout.validation import cases as loading
from keepout.validation import evaluation
from keepout.validation.cases import ValidationCase
from keepout.validation.criteria import Criterion
from keepout.validation.evaluation import Outcome

PASSING = case_with_reference(0.0, case_id="zero", absolute_tolerance=1e-10)
FAILING = case_with_reference(0.5, case_id="half")


class Loader:
    """A stand-in for a case loader: returns ``cases``, records the cache."""

    def __init__(self, *cases: ValidationCase) -> None:
        self.cases = list(cases)
        self.caches: list[Path | None] = []

    def __call__(self, cache: Path | None = None) -> list[ValidationCase]:
        self.caches.append(cache)
        return self.cases


def use(
    monkeypatch: pytest.MonkeyPatch, *cases: ValidationCase, monte_carlo: bool = False
) -> Loader:
    loader = Loader(*cases)
    name = "load_monte_carlo_cases" if monte_carlo else "load_all"
    monkeypatch.setattr(loading, name, loader)
    return loader


def test_validate_prints_the_summary_and_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = dataclasses.replace(
        case_with_reference(1e-3), suite="other", criterion=Criterion(relative=1.0)
    )
    use(monkeypatch, PASSING, relative)
    result = keepout("validate", "--workers", "1")
    assert result.exit_code == ExitCode.OK
    assert result.stderr == ""
    assert result.stdout.replace(CARA_COMMIT[:12], "<commit>") == snapshot("""\
Validation against NASA CARA
Suite  Method  Cases passed  Worst relative error
─────────────────────────────────────────────────
test   foster  1 / 1                            -
test   elrod   1 / 1                            -
other  foster  1 / 1                      1.0e+00
other  elrod   1 / 1                      1.0e+00
4 of 4 comparisons passed (CARA commit <commit>).
""")


def test_validate_fails_with_status_1_and_names_the_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use(monkeypatch, PASSING, FAILING)
    result = keepout("validate", "--workers", "1")
    assert result.exit_code == ExitCode.ERROR
    assert "FAIL test / half / foster: 0.000000e+00 vs CARA 5.000000e-01" in (
        result.stdout
    )
    assert "FAIL test / half / elrod" in result.stdout
    assert "2 of 4 comparisons passed" in result.stdout
    assert result.stderr == "Error: 2 of 4 comparisons failed\n"


def test_validate_json(monkeypatch: pytest.MonkeyPatch) -> None:
    use(monkeypatch, PASSING, FAILING)
    result = keepout("validate", "--workers", "1", "--json")
    assert result.exit_code == ExitCode.ERROR
    data = json.loads(result.stdout)
    assert data["cara_commit"] == CARA_COMMIT
    assert (data["monte_carlo"], data["comparisons"], data["failed"]) == (False, 4, 2)
    assert data["summary"] == [
        {
            "suite": "test",
            "method": "foster",
            "passed": 1,
            "total": 2,
            "worst_relative_error": 1.0,
        },
        {
            "suite": "test",
            "method": "elrod",
            "passed": 1,
            "total": 2,
            "worst_relative_error": 1.0,
        },
    ]
    assert data["outcomes"][2] == {
        "suite": "test",
        "case_id": "half",
        "method": "foster",
        "reference": 0.5,
        "value": 0.0,
        "error": 1.0,
        "relative": True,
        "passed": False,
        "ci": None,
    }
    assert [o["case_id"] for o in data["outcomes"]] == ["zero"] * 2 + ["half"] * 2


def test_validate_monte_carlo_runs_those_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monte_carlo = use(monkeypatch, PASSING, monte_carlo=True)
    main = use(monkeypatch)
    result = keepout("validate", "--monte-carlo", "--workers", "1", "--json")
    assert result.exit_code == ExitCode.OK
    assert json.loads(result.stdout)["monte_carlo"] is True
    assert (len(monte_carlo.caches), main.caches) == (1, [])
    text = keepout("validate", "--monte-carlo", "--workers", "1")
    assert text.stdout.startswith("Monte Carlo validation against NASA CARA\n")


def test_validate_reports_the_interval_of_monte_carlo_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use(monkeypatch, PASSING)

    def evaluate(cases: list[ValidationCase], *, workers: int) -> list[Outcome]:
        result = PcResult(1e-3, "montecarlo", interval=(5e-4, 2e-3))
        return [Outcome(cases[0], result)]

    monkeypatch.setattr(evaluation, "evaluate", evaluate)
    data = json.loads(keepout("validate", "--json").stdout)
    assert data["outcomes"][0]["ci"] == [5e-4, 2e-3]


@pytest.mark.parametrize(("args", "workers"), [((), None), (("--workers", "3"), 3)])
def test_validate_workers_and_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    args: tuple[str, ...],
    workers: int | None,
) -> None:
    loader = use(monkeypatch)
    seen: list[int | None] = []

    def evaluate(cases: list[ValidationCase], *, workers: int | None) -> list[Outcome]:
        seen.append(workers)
        return []

    monkeypatch.setattr(evaluation, "evaluate", evaluate)
    result = keepout("validate", *args, "--cache-dir", str(tmp_path))
    assert result.exit_code == ExitCode.OK
    assert seen == [workers]
    assert loader.caches == [tmp_path]


def test_validate_needs_a_worker() -> None:
    result = keepout("validate", "--workers", "0")
    assert result.exit_code == ExitCode.USAGE
    assert "0 is not in the range x>=1" in result.stderr
