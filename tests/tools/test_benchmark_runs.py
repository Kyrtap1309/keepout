import sys
from collections.abc import Sequence
from pathlib import Path

import benchmark_runs
import pytest


class Fake:
    """Measurement processes answering with numbered runs per tree, or
    failing when a tree has none, recording the calls."""

    def __init__(self, head: list[int] | None, base: list[int] | None) -> None:
        self.runs = {"head": head, "base": base}
        self.calls: list[str] = []

    def __call__(self, command: Sequence[str]) -> int:
        side = command[0]
        self.calls.append(side)
        runs = self.runs[side]
        if runs is None:
            raise benchmark_runs.TimingFailed(f"ModuleNotFoundError: {side}")
        return runs.pop(0)


def test_alternate_alternates_the_trees() -> None:
    fake = Fake([1, 2, 3], [4, 5, 6])
    comparison = benchmark_runs.alternate(["head"], ["base"], repeat=3, run=fake)
    assert fake.calls == ["head", "base", "base", "head", "head", "base"]
    assert comparison == benchmark_runs.Comparison((1, 2, 3), (4, 5, 6))


def test_alternate_stops_at_a_failure_of_the_base() -> None:
    fake = Fake([1, 2], None)
    comparison = benchmark_runs.alternate(["head"], ["base"], repeat=2, run=fake)
    assert fake.calls == ["head", "base"]
    assert comparison == benchmark_runs.Comparison(
        (1,), (), "ModuleNotFoundError: base"
    )


def test_alternate_fails_when_this_tree_cannot_run() -> None:
    fake = Fake(None, [1])
    with pytest.raises(benchmark_runs.TimingFailed, match="head"):
        benchmark_runs.alternate(["head"], ["base"], repeat=1, run=fake)


def test_run_process_returns_the_standard_output() -> None:
    command = [sys.executable, "-c", "print('measured')"]
    assert benchmark_runs.run_process(command) == "measured\n"


def test_run_process_reports_the_exit_status() -> None:
    with pytest.raises(benchmark_runs.TimingFailed, match="exit status 3"):
        benchmark_runs.run_process([sys.executable, "-c", "raise SystemExit(3)"])


def test_run_process_reports_the_end_of_standard_error() -> None:
    script = "import sys; print('\\n'.join(map(str, range(9))), file=sys.stderr); 1/0"
    with pytest.raises(benchmark_runs.TimingFailed) as raised:
        benchmark_runs.run_process([sys.executable, "-c", script])
    assert str(raised.value).endswith("ZeroDivisionError: division by zero")
    assert "\n0\n" not in str(raised.value)


def test_run_process_reports_a_missing_program(tmp_path: Path) -> None:
    missing = str(tmp_path / "missing" / "python")
    with pytest.raises(benchmark_runs.TimingFailed, match="No such file"):
        benchmark_runs.run_process([missing])


def test_write_step_summary_appends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = tmp_path / "summary.md"
    summary.write_text("earlier\n", encoding="utf-8")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    benchmark_runs.write_step_summary("text")
    assert summary.read_text(encoding="utf-8") == "earlier\ntext\n"


def test_write_step_summary_without_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.chdir(tmp_path)
    benchmark_runs.write_step_summary("text")
    assert list(tmp_path.iterdir()) == []


def test_skipped_lines_quote_the_reason() -> None:
    assert benchmark_runs.skipped_lines("no keepout") == [
        "The base cannot run the benchmark, so there is no comparison:",
        "",
        "```text",
        "no keepout",
        "```",
    ]


@pytest.mark.parametrize(
    ("change", "line", "regression"),
    [
        (0.25, "CPU time change: +25.0% (limit +20%): slower than allowed.", True),
        (0.2, "CPU time change: +20.0% (limit +20%): within the limit.", False),
        (-0.1, "CPU time change: -10.0% (limit +20%): within the limit.", False),
    ],
)
def test_verdict_line(change: float, line: str, regression: bool) -> None:
    assert benchmark_runs.verdict_line("CPU time change:", change, 0.2) == (
        line,
        regression,
    )


def test_cpu_change_compares_medians() -> None:
    change = benchmark_runs.cpu_change([1.0, 1.2, 9.0], [1.0, 0.5, 1.0])
    assert change == pytest.approx(0.2)
