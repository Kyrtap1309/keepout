from pathlib import Path

import pytest
from pc_report_builders import case_with_reference

from keepout.validation import __main__ as cli
from keepout.validation.report import (
    SUMMARY_END,
    SUMMARY_START,
    evaluate,
    judge,
    render_page,
    render_summary,
    replace_summary,
)


def test_relative_criterion_above_1e_10() -> None:
    case = case_with_reference(1e-3)
    assert judge(case, 1e-3 * (1 + 5e-7)) == pytest.approx((5e-7, True, True))
    assert judge(case, 1e-3 * (1 + 2e-6))[2] is False


def test_absolute_criterion_below_1e_10() -> None:
    case = case_with_reference(1e-20)
    error, relative, passed = judge(case, 5e-16)
    assert (relative, passed) == (False, True)
    assert error == pytest.approx(5e-16)
    assert judge(case, 2e-15)[2] is False


def test_case_tolerance_overrides_the_default() -> None:
    case = case_with_reference(0.0, absolute_tolerance=1e-10)
    assert judge(case, 5e-11) == (5e-11, False, True)


def test_evaluate_runs_both_methods() -> None:
    case = case_with_reference(0.0, absolute_tolerance=1e-10)
    outcomes = evaluate([case])
    assert [o.method for o in outcomes] == ["foster", "elrod"]


def test_summary_and_page_report_failures() -> None:
    passing = case_with_reference(0.0, absolute_tolerance=1.0)
    failing = case_with_reference(0.5, case_id="bad")
    outcomes = evaluate([passing, failing])
    assert "| foster | 1 / 2 |" in render_summary(outcomes)
    page = render_page(outcomes)
    assert page.startswith("# Validation against NASA CARA")
    assert "| bad | 5 |" in page
    assert "**FAIL**" in page
    assert "## Not compared" in page


def test_replace_summary_rewrites_only_the_marked_block() -> None:
    readme = f"intro\n{SUMMARY_START}\nold\n{SUMMARY_END}\nrest\n"
    assert replace_summary(readme, "new") == (
        f"intro\n{SUMMARY_START}\nnew\n{SUMMARY_END}\nrest\n"
    )


def test_replace_summary_needs_the_markers() -> None:
    with pytest.raises(ValueError, match="README needs"):
        replace_summary("no markers", "new")


def test_main_writes_then_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = case_with_reference(0.0, absolute_tolerance=1.0)
    monkeypatch.setattr(cli, "load_all", lambda: [case])
    page, readme = tmp_path / "validation.md", tmp_path / "README.md"
    readme.write_text(f"{SUMMARY_START}\n{SUMMARY_END}\n", encoding="utf-8")
    args = ["--page", str(page), "--readme", str(readme)]
    assert cli.main(["--check", *args]) == 1
    assert cli.main(args) == 0
    assert "| foster | 1 / 1 |" in readme.read_text(encoding="utf-8")
    assert cli.main(["--check", *args]) == 0


def test_main_fails_when_a_case_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "load_all", lambda: [case_with_reference(0.5)])
    readme = tmp_path / "README.md"
    readme.write_text(f"{SUMMARY_START}\n{SUMMARY_END}\n", encoding="utf-8")
    status = cli.main(["--page", str(tmp_path / "v.md"), "--readme", str(readme)])
    assert status == 1
    assert "FAIL test case foster" in capsys.readouterr().out
