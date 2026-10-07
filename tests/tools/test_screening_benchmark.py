import copy
import csv
import gzip
import io
import json
import shutil
import stat
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import screening_benchmark as benchmark
import screening_snapshot as snapshot
from astropy.time import Time
from screening_timing import Timing, timed

from keepout.omm import Omm, load
from keepout.screening import DEFAULT_VOLUME, ScreeningResult

FIXTURE = load(snapshot.DATA / snapshot.FIXTURE)
PRIMARIES = ["63216", "58509", "44476"]
"""LEO objects of the test fixture with approaches in the week."""


def data_dir(tmp_path: Path) -> Path:
    """A data directory whose snapshot is the test fixture."""
    data = tmp_path / "data"
    data.mkdir()
    shutil.copy(snapshot.DATA / snapshot.FIXTURE, data / snapshot.SNAPSHOT)
    shutil.copy(snapshot.DATA / snapshot.MANIFEST, data / snapshot.MANIFEST)
    return data


def arguments(tmp_path: Path) -> list[str]:
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{benchmark.START}\n{benchmark.END}\n", encoding="utf-8")
    return [
        "--data",
        str(data_dir(tmp_path)),
        "--results",
        str(tmp_path / "docs" / "screening-benchmark.json"),
        "--readme",
        str(readme),
        "--primaries",
        *PRIMARIES,
        "--workers",
        "1",
        "--repeat",
        "1",
    ]


def written(tmp_path: Path) -> benchmark.Summary:
    path = tmp_path / "docs" / "screening-benchmark.json"
    summary: benchmark.Summary = json.loads(path.read_text(encoding="utf-8"))
    return summary


def rewrite(tmp_path: Path, summary: benchmark.Summary) -> None:
    """Replace the results file and the README block by ``summary``'s."""
    path = tmp_path / "docs" / "screening-benchmark.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    readme = tmp_path / "README.md"
    block = benchmark.render_summary(summary)
    readme.write_text(
        f"intro\n{benchmark.START}\n{block}\n{benchmark.END}\n", encoding="utf-8"
    )


@pytest.fixture(scope="module")
def benchmarked() -> tuple[list[benchmark.Found], list[benchmark.Measured]]:
    """One real benchmark of :data:`PRIMARIES` in the fixture, for ``main``."""
    by_number = {omm.catalog_number: omm for omm in FIXTURE}
    return benchmark.benchmark(
        [by_number[int(number)] for number in PRIMARIES],
        FIXTURE,
        benchmark.window_start("2026-10-06"),
        workers=1,
        repeat=1,
    )


def test_main_writes_then_checks_the_data_not_the_timings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    benchmarked: tuple[list[benchmark.Found], list[benchmark.Measured]],
) -> None:
    # main screens as the fixture says; its runs take that benchmark's
    # results, so only the writing and checking runs here.
    calls: list[tuple[list[int], int, str, dict[str, Any]]] = []

    def fake(
        primaries: Sequence[Omm], catalog: Sequence[Omm], start: Time, **kwargs: Any
    ) -> tuple[list[benchmark.Found], list[benchmark.Measured]]:
        numbers = [omm.catalog_number for omm in primaries]
        calls.append((numbers, len(catalog), start.isot, kwargs))
        return copy.deepcopy(benchmarked)

    monkeypatch.setattr(benchmark, "benchmark", fake)
    args = arguments(tmp_path)
    assert benchmark.main(["--check", *args]) == 1
    assert benchmark.main(args) == 0
    assert benchmark.main(["--check", *args]) == 0
    screened = ([63216, 58509, 44476], 222, "2026-10-06T00:00:00.000")
    assert calls == [(*screened, {"workers": 1, "repeat": 1})] * 3
    summary = written(tmp_path)
    assert summary["settings"] == {
        "primaries": [63216, 58509, 44476],
        "objects": 222,
        "retrieved": "2026-10-06",
        "start": "2026-10-06 00:00",
        "days": 7.0,
        "workers": 1,
    }
    assert [f["volume"] for f in summary["found"]] == list(benchmark.VOLUMES)
    default = summary["found"][0]
    assert default["screened"] == 217
    assert default["skipped"] == {"stale": 5}
    assert default["approaches"] > 0
    timings = summary["timings"]
    assert timings is not None
    assert timings["runs"] == 1
    assert timings["cpu_model"] == benchmark.cpu_model()
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert (
        "| Volume | Screened | Left out | Approaches | Wall time | CPU time |" in readme
    )
    assert (
        f"| 5 km sphere (default) | 217 | 5 stale | {default['approaches']} |" in readme
    )
    assert "for 7 days from 2026-10-06 00:00 UTC, in one process" in readme
    assert "Times: 1 run on" in readme

    # Other timings: still up to date.
    faster = copy.deepcopy(summary)
    assert faster["timings"] is not None
    for measured in faster["timings"]["volumes"]:
        measured["wall"], measured["cpu"] = 0.1, 0.2
    faster["timings"]["cpu_model"] = "another CPU"
    rewrite(tmp_path, faster)
    assert benchmark.main(["--check", *args]) == 0
    assert "| 0.1 s | 0.2 s |" in (tmp_path / "README.md").read_text(encoding="utf-8")

    # Other data: out of date.
    changed = copy.deepcopy(faster)
    changed["found"][0]["approaches"] += 1
    rewrite(tmp_path, changed)
    assert benchmark.main(["--check", *args]) == 1


def test_runs_that_find_different_results_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = iter(range(10))

    def found(volume: str, result: ScreeningResult) -> benchmark.Found:
        return {
            "volume": volume,
            "screened": 1,
            "skipped": {},
            "approaches": next(counts),
        }

    monkeypatch.setattr(benchmark, "found", found)
    start = benchmark.window_start("2026-10-06")
    with pytest.raises(RuntimeError, match="the runs found different results"):
        benchmark.benchmark(
            FIXTURE[:1],
            FIXTURE[:5],
            start,
            workers=1,
            repeat=2,
            volumes={"5 km": DEFAULT_VOLUME},
        )


def test_benchmark_takes_the_median_of_the_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = iter([Timing(3.0, 30.0), Timing(1.0, 10.0), Timing(2.0, 20.0)])

    def fake(*args: Any, **kwargs: Any) -> tuple[ScreeningResult, Timing]:
        result, _ = timed(*args, **kwargs)
        return result, next(times)

    monkeypatch.setattr(benchmark, "timed", fake)
    start = benchmark.window_start("2026-10-06")
    found, measured = benchmark.benchmark(
        FIXTURE[:1],
        FIXTURE[:5],
        start,
        workers=1,
        repeat=3,
        volumes={"v": DEFAULT_VOLUME},
    )
    assert measured == [{"volume": "v", "wall": 2.0, "cpu": 20.0}]
    assert found[0]["volume"] == "v"


def test_render_without_timings() -> None:
    summary: benchmark.Summary = {
        "settings": {
            "primaries": [1, 2],
            "objects": 10,
            "retrieved": "2026-10-06",
            "start": "2026-10-06 00:00",
            "days": 7.0,
            "workers": 4,
        },
        "found": [{"volume": "v", "screened": 10, "skipped": {}, "approaches": 3}],
        "timings": None,
    }
    text = benchmark.render_summary(summary)
    assert text.startswith(
        "Screening 2 LEO primaries against the 10 element sets of the frozen "
        "public catalog (CelesTrak, retrieved 2026-10-06) for 7 days from "
        "2026-10-06 00:00 UTC, with 4 worker processes"
    )
    assert "| v | 10 | - | 3 | - | - |" in text
    assert text.endswith("Not measured yet.")


def test_cpu_model_reads_the_systems_name(tmp_path: Path) -> None:
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor\t: 0\nmodel name\t: Some CPU @ 3 GHz\n")
    assert benchmark.cpu_model(cpuinfo) == "Some CPU @ 3 GHz"
    cpuinfo.write_text("processor\t: 0\n")
    assert benchmark.cpu_model(cpuinfo)
    assert benchmark.cpu_model(tmp_path / "missing")


@pytest.mark.parametrize(
    "argv",
    [
        ["--repeat", "0"],
        ["--workers", "-1"],
        ["compare", "--base-python", "p", "--every", "0"],
    ],
)
def test_counts_must_be_positive(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as raised:
        benchmark.main(argv)
    assert raised.value.code == 2
    assert "must be at least 1" in capsys.readouterr().err


def test_the_primaries_are_in_the_frozen_catalog() -> None:
    text = gzip.decompress((snapshot.DATA / snapshot.SNAPSHOT).read_bytes()).decode()
    numbers = {int(row["NORAD_CAT_ID"]) for row in csv.DictReader(io.StringIO(text))}
    assert len(set(benchmark.PRIMARIES)) == 20
    assert set(benchmark.PRIMARIES) <= numbers


def test_reduced_catalog_keeps_every_kth_object_and_the_primaries() -> None:
    primary = FIXTURE[5].catalog_number
    reduced = benchmark.reduced_catalog(FIXTURE, [primary], 4)
    assert reduced == [
        omm for index, omm in enumerate(FIXTURE) if index % 4 == 0 or index == 5
    ]


def run(cpu: float) -> benchmark.Run:
    return benchmark.Run(wall=cpu + 0.5, cpu=cpu, approaches=7)


class Fake:
    """Timed runs from lists of CPU times, recording the calls."""

    def __init__(self, head: list[float] | None, base: list[float] | None) -> None:
        self.cpu = {"head": head, "base": base}
        self.calls: list[str] = []

    def __call__(self, command: Sequence[str]) -> benchmark.Run:
        side = command[0]
        self.calls.append(side)
        times = self.cpu[side]
        if times is None:
            raise benchmark.TimingFailed("ModuleNotFoundError: keepout.screening")
        return run(times.pop(0))


def test_compare_alternates_and_takes_medians() -> None:
    fake = Fake([1.0, 1.1, 5.0, 1.05], [1.0, 1.0, 0.95, 4.0])
    comparison = benchmark.compare(["head"], ["base"], repeat=4, run=fake)
    assert fake.calls == ["head", "base", "base", "head"] * 2
    assert comparison.change == pytest.approx(1.075 / 1.0 - 1.0)
    text, regression = benchmark.render_comparison(comparison, "setting", 0.2)
    assert not regression
    assert "| base | 1.00 s | 0.95 - 4.00 s | 1.50 s | 7 |" in text
    assert "| this | 1.08 s | 1.00 - 5.00 s | 1.58 s | 7 |" in text
    assert text.endswith("CPU time change: +7.5% (limit +20%): within the limit.")


def test_compare_finds_a_regression() -> None:
    fake = Fake([1.3, 1.25, 1.2], [1.0, 1.0, 1.0])
    comparison = benchmark.compare(["head"], ["base"], repeat=3, run=fake)
    text, regression = benchmark.render_comparison(comparison, "setting", 0.2)
    assert regression
    assert "+25.0% (limit +20%): slower than allowed." in text


def test_compare_skips_a_base_that_cannot_run() -> None:
    fake = Fake([1.0, 1.0], None)
    comparison = benchmark.compare(["head"], ["base"], repeat=2, run=fake)
    assert fake.calls == ["head", "base"]
    assert comparison.base == ()
    text, regression = benchmark.render_comparison(comparison, "setting", 0.2)
    assert not regression
    assert "The base cannot run the benchmark" in text
    assert "ModuleNotFoundError: keepout.screening" in text


def test_compare_fails_when_this_tree_cannot_run() -> None:
    fake = Fake(None, [1.0])
    with pytest.raises(benchmark.TimingFailed):
        benchmark.compare(["head"], ["base"], repeat=1, run=fake)


def test_run_timing_reports_failures() -> None:
    with pytest.raises(benchmark.TimingFailed, match="exit status 3"):
        benchmark.run_timing([sys.executable, "-c", "raise SystemExit(3)"])
    with pytest.raises(benchmark.TimingFailed, match="No such file"):
        benchmark.run_timing(["/nonexistent/python"])


def compare_arguments(tmp_path: Path, base_python: str) -> list[str]:
    return [
        "compare",
        "--base-python",
        base_python,
        "--data",
        str(data_dir(tmp_path)),
        "--primaries",
        *PRIMARIES,
        "--every",
        "1",
        "--days",
        "1",
        "--repeat",
        "1",
    ]


def fake_python(tmp_path: Path, cpu: float) -> str:
    """An executable printing one measurement whatever it is asked."""
    path = tmp_path / "fake-python"
    measurement = {"wall": cpu, "cpu": cpu, "approaches": 1, "screened": 1}
    path.write_text(
        f"#!{sys.executable}\nprint({json.dumps(json.dumps(measurement))})\n"
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def test_compare_main_against_a_faster_base_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    step_summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(step_summary))
    args = compare_arguments(tmp_path, fake_python(tmp_path, 1e-6))
    assert benchmark.main(args) == 1
    out = capsys.readouterr().out
    assert out.startswith(
        "Screening benchmark: 3 primaries x 222 objects x 1 day, 5 km sphere, "
        "1 process, 1 run per tree."
    )
    assert "slower than allowed" in out
    assert step_summary.read_text(encoding="utf-8") == out


def test_compare_main_skips_a_base_that_cannot_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    args = compare_arguments(tmp_path, str(tmp_path / "missing" / "python"))
    assert benchmark.main(args) == 0
    assert "The base cannot run the benchmark" in capsys.readouterr().out
