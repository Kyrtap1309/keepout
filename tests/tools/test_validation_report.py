import dataclasses
from pathlib import Path

import pytest
import validation_report as report
from builders import case_with_reference

from keepout.pc.result import PcResult
from keepout.validation.criteria import MONTE_CARLO, BoundCriterion
from keepout.validation.evaluation import Outcome, evaluate


def test_summary_and_page_report_failures() -> None:
    passing = case_with_reference(0.0, absolute_tolerance=1.0)
    failing = case_with_reference(0.5, case_id="bad")
    record = report.Record.of(evaluate([passing, failing]))
    assert "| CARA unit tests | foster | 1 / 2 |" in report.render_summary(record.rows)
    page = report.render_page(record)
    assert page.startswith("# Validation against NASA CARA")
    assert "| bad | 5 |" in page
    assert "**FAIL**" in page
    assert "## Not compared" in page


def test_paper_values_are_compared_with_the_paper_method() -> None:
    case = case_with_reference(0.0, absolute_tolerance=1.0)
    with_paper = dataclasses.replace(case, paper=1e-3)
    page = report.render_page(report.Record.of(evaluate([with_paper])))
    assert "| Alfano paper | vs paper |" in page
    assert "| 1.0000e-03 | 1.0e+00 **FAIL** |" in page


def empty_readme(tmp_path: Path) -> Path:
    readme = tmp_path / "README.md"
    readme.write_text(f"{report.MAIN.start}\n{report.MAIN.end}\n", encoding="utf-8")
    return readme


def test_main_writes_then_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = case_with_reference(0.0, absolute_tolerance=1.0)
    monkeypatch.setattr(report, "load_all", lambda: [case])
    page, readme = tmp_path / "validation.md", empty_readme(tmp_path)
    args = ["--page", str(page), "--readme", str(readme)]
    assert report.main(["--check", *args]) == 1
    assert report.main(args) == 0
    assert "| CARA unit tests | foster | 1 / 1 |" in readme.read_text(encoding="utf-8")
    assert report.main(["--check", *args]) == 0


def test_main_fails_when_a_case_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(report, "load_all", lambda: [case_with_reference(0.5)])
    readme = empty_readme(tmp_path)
    status = report.main(["--page", str(tmp_path / "v.md"), "--readme", str(readme)])
    assert status == 1
    assert "FAIL CARA unit tests case foster" in capsys.readouterr().out


def test_a_case_without_a_paper_value_keeps_the_columns() -> None:
    with_paper = dataclasses.replace(
        case_with_reference(0.0, absolute_tolerance=1.0), paper=1e-3
    )
    without = case_with_reference(0.0, case_id="plain", absolute_tolerance=1.0)
    page = report.render_page(report.Record.of(evaluate([with_paper, without])))
    lines = page.splitlines()
    (header,) = [line for line in lines if line.startswith("| Case |")]
    (row,) = [line for line in lines if line.startswith("| plain |")]
    assert row.count("|") == header.count("|")
    assert row.endswith(" - | - |")


def monte_carlo_outcome(shift: float = 0.0, *, case_id: str = "mc") -> Outcome:
    """A passing Monte Carlo row, its value and interval moved by ``shift``.

    The interval's half-width is 4e-4, so shifts below 1e-4 reproduce it.
    """
    case = dataclasses.replace(
        case_with_reference(2e-3, case_id=case_id),
        methods=("montecarlo",),
        criterion=MONTE_CARLO,
        interval=(1.9e-3, 2.1e-3),
    )
    interval = (1.8e-3 + shift, 2.6e-3 + shift)
    result = PcResult(
        2.2e-3 + shift, "montecarlo", (), {"samples": 1e5}, interval=interval
    )
    return Outcome(case, result)


def test_monte_carlo_rows_show_both_intervals_and_the_trials() -> None:
    record = report.Record.of([monte_carlo_outcome()])
    page = report.render_page(record, report.MONTE_CARLO)
    assert page.startswith("# Monte Carlo validation against NASA CARA")
    assert "| CARA 95% interval |" in page
    assert "| [1.8000e-03, 2.6000e-03] | 100000 |" in page
    assert "within keepout's confidence interval" in page


def test_monte_carlo_mode_writes_its_own_page_and_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(report, "load_monte_carlo_cases", list)
    monkeypatch.setattr(report, "evaluate", lambda cases, **_: [monte_carlo_outcome()])
    readme = tmp_path / "README.md"
    readme.write_text(
        f"{report.MONTE_CARLO.start}\n{report.MONTE_CARLO.end}\n", encoding="utf-8"
    )
    page = tmp_path / "mc.md"
    args = ["--monte-carlo", "--page", str(page), "--readme", str(readme)]
    assert report.main(args) == 0
    assert "| CARA unit tests | montecarlo | 1 / 1 |" in readme.read_text(
        encoding="utf-8"
    )
    assert page.read_text(encoding="utf-8").startswith("# Monte Carlo validation")


def test_record_round_trips_through_json() -> None:
    case = dataclasses.replace(
        case_with_reference(0.0, absolute_tolerance=1.0),
        paper=1e-3,
        informative=("CARA 3-D", 2e-3),
    )
    record = report.Record.of([*evaluate([case]), monte_carlo_outcome()])
    assert report.Record.loads(record.dumps()) == record


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('"method": "foster"', '"method": "coppola"', "method"),
        ('"passed": true', '"passed": "yes"', "passed"),
        ('"samples": null', '"samples": null, "extra": 1', "extra"),
    ],
)
def test_a_record_of_the_wrong_shape_is_refused(
    old: str, new: str, message: str
) -> None:
    case = case_with_reference(0.0, absolute_tolerance=1.0)
    text = report.Record.of(evaluate([case])).dumps().replace(" ", "")
    old, new = old.replace(" ", ""), new.replace(" ", "")
    assert old in text
    with pytest.raises(ValueError, match=f"results is not a RecordJson: .*{message}"):
        report.Record.loads(text.replace(old, new, 1), "results")


def test_a_pair_that_is_not_two_numbers_is_refused() -> None:
    case = dataclasses.replace(
        case_with_reference(0.0, absolute_tolerance=1.0), interval=(1e-3, 2e-3)
    )
    text = report.Record.of(evaluate([case])).dumps()
    with pytest.raises(ValueError, match="expected a pair of numbers"):
        report.Record.loads(text.replace("0.002\n", "0.002,\n    0.003\n", 1))


def differences(previous: Outcome, current: Outcome) -> list[str]:
    old, new = report.Record.of([previous]), report.Record.of([current])
    return new.differences(old)


def test_a_result_within_the_reproducibility_is_reproduced() -> None:
    assert differences(monte_carlo_outcome(), monte_carlo_outcome(9e-5)) == []


def test_a_result_beyond_the_reproducibility_is_not_reproduced() -> None:
    (difference,) = differences(monte_carlo_outcome(), monte_carlo_outcome(1.1e-4))
    assert difference.startswith("CARA unit tests mc montecarlo: 2.2")


def test_a_changed_verdict_is_not_reproduced() -> None:
    passing = case_with_reference(0.5, absolute_tolerance=1.0)
    failing = dataclasses.replace(passing, criterion=BoundCriterion(absolute=0.1))
    old = report.Record.of(evaluate([passing]))
    new = report.Record.of(evaluate([failing]))
    assert new.differences(old) == [
        f"CARA unit tests case {method}: CARA data or verdict changed"
        for method in ("foster", "elrod")
    ]


def test_changed_cases_are_not_reproduced() -> None:
    assert differences(monte_carlo_outcome(), monte_carlo_outcome(case_id="other")) == [
        "the cases or methods changed"
    ]


def test_another_cara_commit_is_not_reproduced() -> None:
    record = report.Record.of([monte_carlo_outcome()])
    other = dataclasses.replace(record, cara_commit="0" * 40)
    assert other.differences(record) == [
        f"CARA commit {record.cara_commit} -> {'0' * 40}"
    ]


def run_monte_carlo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shift: float, *flags: str
) -> int:
    monkeypatch.setattr(report, "load_monte_carlo_cases", list)
    outcomes = [monte_carlo_outcome(shift)]
    monkeypatch.setattr(report, "evaluate", lambda _, **__: outcomes)
    readme = tmp_path / "README.md"
    if not readme.exists():
        readme.write_text(
            f"{report.MONTE_CARLO.start}\n{report.MONTE_CARLO.end}\n", encoding="utf-8"
        )
    page = tmp_path / "mc.md"
    return report.main(
        ["--monte-carlo", "--page", str(page), "--readme", str(readme), *flags]
    )


def test_a_reproducing_run_keeps_the_published_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert run_monte_carlo(tmp_path, monkeypatch, 0.0) == 0
    published = (tmp_path / "mc.json").read_text(encoding="utf-8")
    assert run_monte_carlo(tmp_path, monkeypatch, 9e-5, "--check") == 0
    assert run_monte_carlo(tmp_path, monkeypatch, 9e-5) == 0
    assert (tmp_path / "mc.json").read_text(encoding="utf-8") == published


def test_a_run_that_does_not_reproduce_fails_the_check_then_replaces_the_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_monte_carlo(tmp_path, monkeypatch, 0.0) == 0
    capsys.readouterr()
    assert run_monte_carlo(tmp_path, monkeypatch, 1.1e-4, "--check") == 1
    assert "not reproduced: CARA unit tests mc montecarlo" in capsys.readouterr().out
    assert run_monte_carlo(tmp_path, monkeypatch, 1.1e-4) == 0
    assert run_monte_carlo(tmp_path, monkeypatch, 1.1e-4, "--check") == 0
