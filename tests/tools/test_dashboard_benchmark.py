import json
import sqlite3
import stat
import sys
from collections.abc import Sequence
from pathlib import Path

import dashboard_benchmark as benchmark
import dashboard_timing as timing
import pytest

from keepout.settings import Settings
from keepout.store import open_store

EVENTS = 20
CDMS = 100


@pytest.fixture(scope="module")
def small_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A store of 20 events and 100 CDMs, built once (read-only)."""
    path = tmp_path_factory.mktemp("bench") / "bench.db"
    return benchmark.build_store(path, events=EVENTS, cdms=CDMS)


def _rows(path: Path, query: str) -> list[tuple[object, ...]]:
    connection = sqlite3.connect(path)
    try:
        return connection.execute(query).fetchall()
    finally:
        connection.close()


def _counts(path: Path) -> list[int]:
    """The number of CDMs of each event, fewest first."""
    rows = _rows(path, "SELECT count(*) FROM cdms GROUP BY event_id")
    return sorted(int(str(count)) for (count,) in rows)


def test_build_store_counts(small_store: Path) -> None:
    assert _rows(small_store, "SELECT count(*) FROM events") == [(EVENTS,)]
    assert _rows(small_store, "SELECT count(*) FROM cdms") == [(CDMS,)]
    # Every CDM has its Pc for the benchmark's settings: no event is "not
    # computed", and each event has its share of the CDMs.
    assert _rows(small_store, "SELECT count(*) FROM results") == [(CDMS,)]
    assert _counts(small_store) == [CDMS // EVENTS] * EVENTS
    with open_store(small_store, create=False) as store:
        page = store.summaries(Settings(), as_of=timing.AS_OF, include_past=True)
    assert page.total == EVENTS
    assert all(item.assessment is not None for item in page.items)


def test_build_store_spreads_the_remainder(tmp_path: Path) -> None:
    path = benchmark.build_store(tmp_path / "odd.db", events=3, cdms=7)
    assert _counts(path) == [2, 2, 3]


def test_build_store_is_deterministic(small_store: Path, tmp_path: Path) -> None:
    again = benchmark.build_store(tmp_path / "again.db", events=EVENTS, cdms=CDMS)
    query = "SELECT message_id, content FROM cdms ORDER BY message_id"
    assert _rows(again, query) == _rows(small_store, query)
    query = "SELECT id FROM events ORDER BY id"
    assert _rows(again, query) == _rows(small_store, query)


def test_build_store_needs_a_cdm_per_event(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one CDM per event"):
        benchmark.build_store(tmp_path / "s.db", events=3, cdms=2)


def test_measure_returns_positive(small_store: Path) -> None:
    timings = benchmark.measure(small_store, repeat=2)
    assert timings.list_s > 0.0
    assert timings.event_s is not None
    assert timings.event_s > 0.0
    assert timings.list_cpu > 0.0
    assert timings.event_cpu is not None


def test_timing_main_prints_one_measurement(
    small_store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert timing.main(["time", str(small_store), "--repeat", "1"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert set(output) == {"list_wall", "list_cpu", "event_wall", "event_cpu"}
    assert output["list_wall"] > 0.0


def test_timing_main_builds_a_store(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "built.db"
    args = ["build", str(path), "--events", "2", "--cdms", "4"]
    assert timing.main(args) == 0
    assert _rows(path, "SELECT count(*) FROM cdms") == [(4,)]
    assert "built" in capsys.readouterr().out


def run(list_cpu: float, event_cpu: float | None = 1.0) -> benchmark.Run:
    return benchmark.Run(
        list_wall=list_cpu,
        list_cpu=list_cpu,
        event_wall=event_cpu,
        event_cpu=event_cpu,
    )


class Fake:
    """Timing processes answering with given CPU times of the list, per tree."""

    def __init__(
        self,
        head: list[benchmark.Run] | None,
        base: list[benchmark.Run] | None,
    ) -> None:
        self.runs = {"head": head, "base": base}
        self.calls: list[str] = []
        self.failure = "ModuleNotFoundError: keepout.dashboard"

    def __call__(self, command: Sequence[str]) -> benchmark.Run:
        side = command[0]
        self.calls.append(side)
        runs = self.runs[side]
        if runs is None:
            raise benchmark.TimingFailed(self.failure)
        return runs.pop(0)


def test_compare_flags_regression() -> None:
    fake = Fake([run(1.3), run(1.25), run(1.2)], [run(1.0), run(1.0), run(1.0)])
    comparison = benchmark.compare(["head"], ["base"], repeat=3, run=fake)
    assert fake.calls == ["head", "base", "base", "head", "head", "base"]
    text, regression = benchmark.render_comparison(comparison, "setting", 0.2)
    assert regression
    assert "Events list: CPU time change +25.0% (limit +20%): slower than allowed." in (
        text
    )
    assert "Event page: CPU time change +0.0% (limit +20%): within the limit." in text


def test_compare_flags_a_slower_event_page() -> None:
    fake = Fake([run(1.0, 2.0)], [run(1.0, 1.0)])
    comparison = benchmark.compare(["head"], ["base"], repeat=1, run=fake)
    text, regression = benchmark.render_comparison(comparison, "setting", 0.2)
    assert regression
    assert "Event page: CPU time change +100.0%" in text


def test_compare_within_the_limit() -> None:
    fake = Fake([run(1.1), run(1.0)], [run(1.0), run(1.0)])
    comparison = benchmark.compare(["head"], ["base"], repeat=2, run=fake)
    text, regression = benchmark.render_comparison(comparison, "setting", 0.2)
    assert not regression
    assert "| base | 1.000 s | 1.000 - 1.000 s | 1.000 s | 1.000 - 1.000 s |" in text
    assert "| this | 1.050 s | 1.000 - 1.100 s | 1.000 s | 1.000 - 1.000 s |" in text


def test_compare_skips_the_event_page_of_a_base_without_it() -> None:
    fake = Fake([run(1.0, 9.0)], [run(1.0, None)])
    comparison = benchmark.compare(["head"], ["base"], repeat=1, run=fake)
    text, regression = benchmark.render_comparison(comparison, "setting", 0.2)
    assert not regression
    assert "Event page: the base has no event page data path" in text
    assert "| base | 1.000 s | 1.000 - 1.000 s | - | - |" in text


def test_compare_skips_a_base_that_cannot_run() -> None:
    fake = Fake([run(1.0)], None)
    comparison = benchmark.compare(["head"], ["base"], repeat=2, run=fake)
    assert fake.calls == ["head", "base"]
    text, regression = benchmark.render_comparison(comparison, "setting", 0.2)
    assert not regression
    assert "The base cannot run the benchmark" in text
    assert "ModuleNotFoundError: keepout.dashboard" in text


def test_compare_fails_when_this_tree_cannot_run() -> None:
    fake = Fake(None, [run(1.0)])
    with pytest.raises(benchmark.TimingFailed):
        benchmark.compare(["head"], ["base"], repeat=1, run=fake)


def test_run_timing_reports_failures() -> None:
    with pytest.raises(benchmark.TimingFailed, match="exit status 3"):
        benchmark.run_timing([sys.executable, "-c", "raise SystemExit(3)"])
    with pytest.raises(benchmark.TimingFailed, match="No such file"):
        benchmark.run_timing(["/nonexistent/python"])


def fake_python(tmp_path: Path, cpu: float) -> str:
    """An executable printing one measurement whatever it is asked (and
    nothing when asked to build a store)."""
    path = tmp_path / "fake-python"
    measurement = {
        "list_wall": cpu,
        "list_cpu": cpu,
        "event_wall": cpu,
        "event_cpu": cpu,
    }
    path.write_text(
        f"#!{sys.executable}\nimport sys\n"
        f"if 'time' in sys.argv: print({json.dumps(json.dumps(measurement))})\n"
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def compare_arguments(base_python: str) -> list[str]:
    return [
        "compare",
        "--base-python",
        base_python,
        "--events",
        "4",
        "--cdms",
        "8",
        "--repeat",
        "1",
    ]


def test_compare_main_against_a_faster_base_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    step_summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(step_summary))
    args = compare_arguments(fake_python(tmp_path, 1e-6))
    assert benchmark.main(args) == 1
    out = capsys.readouterr().out
    assert out.startswith("Dashboard benchmark: 4 events, 8 CDMs, 1 run per tree.")
    assert "slower than allowed" in out
    assert step_summary.read_text(encoding="utf-8") == out


def test_compare_main_skips_a_base_that_cannot_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    args = compare_arguments(str(tmp_path / "missing" / "python"))
    assert benchmark.main(args) == 0
    assert "The base cannot run the benchmark" in capsys.readouterr().out


def test_run_meets_the_targets(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        benchmark, "measure", lambda path, *, repeat: timing.Timings(0.2, 0.1, 0.5, 0.4)
    )
    assert benchmark.main(["run", "--events", "2", "--cdms", "4"]) == 0
    out = capsys.readouterr().out
    assert "Events list (first page): 0.200 s (target < 1 s): met." in out
    assert "Event page: 0.500 s (target < 2 s): met." in out


@pytest.mark.parametrize(
    ("list_s", "event_s"), [(1.0, 0.5), (0.5, 2.0)], ids=["list", "event"]
)
def test_run_fails_when_a_target_is_missed(
    list_s: float,
    event_s: float,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        benchmark,
        "measure",
        lambda path, *, repeat: timing.Timings(list_s, list_s, event_s, event_s),
    )
    assert benchmark.main(["run", "--events", "2", "--cdms", "4"]) == 1
    assert "missed" in capsys.readouterr().out


def test_timing_main_refuses_a_keepout_without_the_dashboard(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(timing, "missing_dashboard", lambda: "no dashboard here")
    with pytest.raises(SystemExit) as raised:
        timing.main(["time", "store.db"])
    assert raised.value.code == 2
    assert "no dashboard here" in capsys.readouterr().err


def test_this_keepout_has_the_dashboard() -> None:
    assert timing.missing_dashboard() == ""
    assert timing.has_event_page()


def test_measure_without_an_event_page(
    small_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(timing, "has_event_page", lambda: False)
    timings = timing.measure(small_store, repeat=1)
    assert (timings.event_s, timings.event_cpu) == (None, None)
    assert timing.output_of(timings)["event_wall"] is None


def test_compare_main_warns_when_the_base_fails_otherwise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    args = compare_arguments(str(tmp_path / "missing" / "python"))
    assert benchmark.main(args) == 0
    out = capsys.readouterr().out
    assert "::warning::The merge base cannot be timed: " in out
    assert "No such file" in out


def test_compare_main_warns_with_the_exception_of_a_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A base that fails with a traceback: the warning names the exception
    # (its last line), not the traceback's first frame.
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    path = tmp_path / "broken-python"
    path.write_text(f"#!{sys.executable}\nraise RuntimeError('database is locked')\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    assert benchmark.main(compare_arguments(str(path))) == 0
    (warning,) = [
        line for line in capsys.readouterr().out.splitlines() if "::warning::" in line
    ]
    assert warning.endswith("RuntimeError: database is locked")


@pytest.mark.parametrize(
    ("reason", "shown"),
    [
        (
            "Traceback (most recent call last):\n  File x\nOSError: no\n\n",
            "OSError: no",
        ),
        ("one line", "one line"),
        ("  \n", "no reason given"),
    ],
)
def test_warning_names_the_last_line(reason: str, shown: str) -> None:
    assert benchmark._warning(reason) == (
        f"::warning::The merge base cannot be timed: {shown}"
    )


def test_compare_main_skips_a_base_without_the_dashboard_quietly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    path = tmp_path / "old-python"
    path.write_text(
        f"#!{sys.executable}\nimport sys\n"
        f"sys.exit('tools/dashboard_timing.py: {timing.NO_DASHBOARD}: "
        "this keepout has no keepout.dashboard')\n"
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    assert benchmark.main(compare_arguments(str(path))) == 0
    out = capsys.readouterr().out
    assert "The base cannot run the benchmark" in out
    assert "::warning::" not in out


def test_compare_main_reports_a_failure_of_this_tree(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def failing(command: Sequence[str]) -> str:
        raise benchmark.TimingFailed("StoreError: disk full")

    monkeypatch.setattr(benchmark, "run_process", failing)
    assert benchmark.main(compare_arguments("base-python")) == 1
    captured = capsys.readouterr()
    assert captured.err == "Error: this tree cannot be timed: StoreError: disk full\n"
    assert "Traceback" not in captured.err


def test_run_reports_a_build_failure(capsys: pytest.CaptureFixture[str]) -> None:
    assert benchmark.main(["run", "--events", "3", "--cdms", "2"]) == 1
    assert capsys.readouterr().err.startswith("Error: at least one CDM per event")
