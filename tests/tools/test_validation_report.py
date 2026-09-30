import dataclasses
from pathlib import Path

import pytest
import validation_report as report
from builders import case_with_reference

from keepout.pc.result import PcResult
from keepout.validation.criteria import MONTE_CARLO
from keepout.validation.evaluation import Outcome, evaluate


def test_summary_and_page_report_failures() -> None:
    passing = case_with_reference(0.0, absolute_tolerance=1.0)
    failing = case_with_reference(0.5, case_id="bad")
    outcomes = evaluate([passing, failing])
    assert "| test | foster | 1 / 2 |" in report.render_summary(outcomes)
    page = report.render_page(outcomes)
    assert page.startswith("# Validation against NASA CARA")
    assert "| bad | 5 |" in page
    assert "**FAIL**" in page
    assert "## Not compared" in page


def test_paper_values_are_compared_with_the_paper_method() -> None:
    case = case_with_reference(0.0, absolute_tolerance=1.0)
    with_paper = dataclasses.replace(case, paper=1e-3)
    page = report.render_page(evaluate([with_paper]))
    assert "| Alfano paper | vs paper |" in page
    assert "| 1.000000000e-03 | 1.0e+00 **FAIL** |" in page


def test_replace_summary_rewrites_only_the_marked_block() -> None:
    readme = f"intro\n{report.SUMMARY_START}\nold\n{report.SUMMARY_END}\nrest\n"
    assert report.replace_summary(readme, "new") == (
        f"intro\n{report.SUMMARY_START}\nnew\n{report.SUMMARY_END}\nrest\n"
    )


def test_replace_summary_needs_the_markers() -> None:
    with pytest.raises(ValueError, match="README needs"):
        report.replace_summary("no markers", "new")


def empty_readme(tmp_path: Path) -> Path:
    readme = tmp_path / "README.md"
    readme.write_text(
        f"{report.SUMMARY_START}\n{report.SUMMARY_END}\n", encoding="utf-8"
    )
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
    assert "| test | foster | 1 / 1 |" in readme.read_text(encoding="utf-8")
    assert report.main(["--check", *args]) == 0


def test_main_fails_when_a_case_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(report, "load_all", lambda: [case_with_reference(0.5)])
    readme = empty_readme(tmp_path)
    status = report.main(["--page", str(tmp_path / "v.md"), "--readme", str(readme)])
    assert status == 1
    assert "FAIL test case foster" in capsys.readouterr().out


def test_a_case_without_a_paper_value_keeps_the_columns() -> None:
    with_paper = dataclasses.replace(
        case_with_reference(0.0, absolute_tolerance=1.0), paper=1e-3
    )
    without = case_with_reference(0.0, case_id="plain", absolute_tolerance=1.0)
    page = report.render_page(evaluate([with_paper, without]))
    lines = page.splitlines()
    (header,) = [line for line in lines if line.startswith("| Case |")]
    (row,) = [line for line in lines if line.startswith("| plain |")]
    assert row.count("|") == header.count("|")
    assert row.endswith(" - | - |")


def monte_carlo_outcome() -> Outcome:
    case = dataclasses.replace(
        case_with_reference(2e-3, case_id="mc"),
        methods=("montecarlo",),
        criterion=MONTE_CARLO,
        interval=(1.9e-3, 2.1e-3),
    )
    result = PcResult(
        2.2e-3, "montecarlo", (), {"ci_low": 1.8e-3, "ci_high": 2.6e-3, "samples": 1e5}
    )
    return Outcome(case, result)


def test_monte_carlo_rows_show_both_intervals_and_the_trials() -> None:
    page = report.render_page([monte_carlo_outcome()], report.MONTE_CARLO)
    assert page.startswith("# Monte Carlo validation against NASA CARA")
    assert "| CARA 95% interval |" in page
    assert "| [1.8000e-03, 2.6000e-03] | 100000 |" in page
    assert "within keepout's confidence interval" in page


def test_monte_carlo_mode_writes_its_own_page_and_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(report, "load_monte_carlo", lambda: [])
    monkeypatch.setattr(report, "evaluate", lambda cases: [monte_carlo_outcome()])
    readme = tmp_path / "README.md"
    readme.write_text(
        f"{report.MONTE_CARLO.start}\n{report.MONTE_CARLO.end}\n", encoding="utf-8"
    )
    page = tmp_path / "mc.md"
    args = ["--monte-carlo", "--page", str(page), "--readme", str(readme)]
    assert report.main(args) == 0
    assert "| test | montecarlo | 1 / 1 |" in readme.read_text(encoding="utf-8")
    assert page.read_text(encoding="utf-8").startswith("# Monte Carlo validation")
