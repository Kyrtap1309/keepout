import copy
import csv
import gzip
import io
import json
import os
import platform
import shutil
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
import screening_benchmark as benchmark
import screening_snapshot as snapshot
import snapshot_files
import spacetrack_snapshot
from astropy.time import Time
from benchmark_runs import Comparison
from generated import README, json_text, read_json
from report_builders import (
    assert_writes_then_checks,
    fake_python,
    fixture_data,
    readme_with_markers,
)
from screening_data import fixture_by_number, load_fixture
from screening_timing import Timing, timed

from keepout.omm import Omm
from keepout.omm.celestrak import dumps_csv
from keepout.screening import DEFAULT_VOLUME, ScreeningResult

FIXTURE = load_fixture()
CELESTRAK = benchmark.SOURCES["celestrak"]
SPACETRACK = benchmark.SOURCES["spacetrack"]
PRIMARIES = ["63216", "58509", "44476"]
"""LEO objects of the test fixture with approaches in the week."""

OFFLINE_SITE = """\
import socket

_connect = socket.socket.connect


def _refuse(self, address):
    if self.family != socket.AF_UNIX:
        raise OSError("tests must not access the network")
    return _connect(self, address)


socket.socket.connect = _refuse
"""
"""``sitecustomize`` of the measurement processes: no connection but local
ones (as ``--disable-socket --allow-unix-socket`` for the tests)."""


@pytest.fixture(autouse=True)
def offline_subprocesses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the processes ``compare`` and ``run_timing`` start offline.

    The suite's network guards hold in the test process only: Python runs
    ``sitecustomize`` from ``PYTHONPATH`` in each new process, which here
    refuses connections, and astropy reads ``ASTROPY_CONFIG_DIR``, which here
    turns off IERS downloads.
    """
    site = tmp_path / "offline-site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(OFFLINE_SITE, encoding="utf-8")
    config = tmp_path / "offline-astropy"
    config.mkdir()
    (config / "astropy.cfg").write_text(
        "[utils.iers.iers]\nauto_download = False\n", encoding="utf-8"
    )
    path = os.environ.get("PYTHONPATH")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [str(site), path])))
    monkeypatch.setenv("ASTROPY_CONFIG_DIR", str(config))


def test_measurement_processes_are_offline() -> None:
    connect = "import socket; socket.create_connection(('192.0.2.1', 9), timeout=1)"
    with pytest.raises(benchmark.TimingFailed, match="must not access the network"):
        benchmark.run_timing([sys.executable, "-c", connect])
    iers = (
        "from astropy.utils import iers; "
        "assert not iers.conf.auto_download; raise SystemExit(3)"
    )
    with pytest.raises(benchmark.TimingFailed, match="exit status 3"):
        benchmark.run_timing([sys.executable, "-c", iers])


def spacetrack_data(tmp_path: Path) -> Path:
    """A Space-Track data directory whose catalog is the test fixture."""
    data = tmp_path / "spacetrack-data"
    data.mkdir()
    csv_text = dumps_csv(FIXTURE).encode("utf-8")
    (data / spacetrack_snapshot.SNAPSHOT).write_bytes(gzip.compress(csv_text))
    manifest = spacetrack_snapshot.read_manifest(spacetrack_snapshot.DATA)
    manifest["download"]["retrieved"] = "2026-10-15T16:28:59.977Z"
    (data / snapshot_files.MANIFEST).write_text(json.dumps(manifest))
    return data


def arguments(tmp_path: Path) -> list[str]:
    """The fixture as the catalog and a README with the public catalog's
    markers in ``tmp_path``."""
    fixture_data(tmp_path)
    readme_with_markers(tmp_path, CELESTRAK.start, CELESTRAK.end)
    return arguments_of(tmp_path)


def arguments_of(folder: Path) -> list[str]:
    """The arguments of a run on the files of ``folder``."""
    return [
        "--data",
        str(folder / "data"),
        "--results",
        str(folder / "docs" / "screening-benchmark.json"),
        "--readme",
        str(folder / "README.md"),
        "--primaries",
        *PRIMARIES,
        "--workers",
        "1",
        "--repeat",
        "1",
    ]


def written(tmp_path: Path) -> benchmark.Summary:
    path = tmp_path / "docs" / "screening-benchmark.json"
    return read_json(path, benchmark.Summary)


def rewrite(tmp_path: Path, summary: benchmark.Summary) -> None:
    """Replace the results file and the README block by ``summary``'s."""
    path = tmp_path / "docs" / "screening-benchmark.json"
    path.write_text(json_text(summary), encoding="utf-8")
    block = benchmark.render_summary(summary)
    readme_with_markers(tmp_path, CELESTRAK.start, CELESTRAK.end, block)


Benchmarked = tuple[list[benchmark.Found], list[benchmark.Measured]]
Call = tuple[list[int], int, str, dict[str, Any]]


@pytest.fixture(scope="module")
def benchmarked() -> Benchmarked:
    """One real benchmark of :data:`PRIMARIES` in the fixture, for ``main``."""
    return benchmark.benchmark(
        [fixture_by_number()[int(number)] for number in PRIMARIES],
        FIXTURE,
        benchmark.window_start("2026-10-06"),
        workers=1,
        repeat=1,
    )


def replaying(
    benchmarked: Benchmarked, calls: list[Call]
) -> Callable[..., Benchmarked]:
    """A ``benchmark`` recording its calls and answering ``benchmarked``: main
    screens as the fixture says, without running the screenings again."""

    def fake(
        primaries: Sequence[Omm], catalog: Sequence[Omm], start: Time, **kwargs: Any
    ) -> Benchmarked:
        numbers = [omm.catalog_number for omm in primaries]
        calls.append((numbers, len(catalog), start.isot, kwargs))
        return copy.deepcopy(benchmarked)

    return fake


@pytest.fixture(scope="module")
def written_once(
    tmp_path_factory: pytest.TempPathFactory, benchmarked: Benchmarked
) -> tuple[Path, list[Call]]:
    """A directory where main wrote the benchmark (checking before and after),
    and the calls of its screenings."""
    folder = tmp_path_factory.mktemp("benchmark")
    calls: list[Call] = []
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(benchmark, "benchmark", replaying(benchmarked, calls))
        assert_writes_then_checks(benchmark.main, arguments(folder))
    return folder, calls


@pytest.fixture
def copied(
    written_once: tuple[Path, list[Call]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    benchmarked: Benchmarked,
) -> list[str]:
    """The arguments of a run on a copy of :func:`written_once`'s files."""
    shutil.copytree(written_once[0], tmp_path, dirs_exist_ok=True)
    monkeypatch.setattr(benchmark, "benchmark", replaying(benchmarked, []))
    return arguments_of(tmp_path)


def test_main_writes_then_checks(written_once: tuple[Path, list[Call]]) -> None:
    folder, calls = written_once
    screened = ([63216, 58509, 44476], 222, "2026-10-06T00:00:00.000")
    assert calls == [(*screened, {"workers": 1, "repeat": 1})] * 3
    summary = written(folder)
    assert summary["settings"] == {
        "source": "celestrak",
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
    readme = (folder / "README.md").read_text(encoding="utf-8")
    assert (
        "| Volume | Screened | Left out | Approaches | Wall time | CPU time |" in readme
    )
    assert (
        f"| 5 km sphere (default) | 217 | 5 stale | {default['approaches']} |" in readme
    )
    assert "for 7 days from 2026-10-06 00:00 UTC, in one process" in readme
    assert "Times: 1 run on" in readme


def test_check_passes_with_other_timings(copied: list[str], tmp_path: Path) -> None:
    faster = written(tmp_path)
    assert faster["timings"] is not None
    for measured in faster["timings"]["volumes"]:
        measured["wall"], measured["cpu"] = 0.1, 0.2
    faster["timings"]["cpu_model"] = "another CPU"
    rewrite(tmp_path, faster)
    assert benchmark.main(["--check", *copied]) == 0
    assert "| 0.1 s | 0.2 s |" in (tmp_path / "README.md").read_text(encoding="utf-8")


def test_check_fails_with_other_data(copied: list[str], tmp_path: Path) -> None:
    changed = written(tmp_path)
    changed["found"][0]["approaches"] += 1
    rewrite(tmp_path, changed)
    assert benchmark.main(["--check", *copied]) == 1


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
            "source": "celestrak",
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


def test_main_benchmarks_the_spacetrack_catalog_in_its_own_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    benchmarked: tuple[list[benchmark.Found], list[benchmark.Measured]],
) -> None:
    calls: list[tuple[int, str]] = []

    def fake(
        primaries: Sequence[Omm], catalog: Sequence[Omm], start: Time, **kwargs: Any
    ) -> tuple[list[benchmark.Found], list[benchmark.Measured]]:
        calls.append((len(catalog), start.isot))
        return copy.deepcopy(benchmarked)

    monkeypatch.setattr(benchmark, "benchmark", fake)
    readme = tmp_path / "README.md"
    public = f"{CELESTRAK.start}\nthe public catalog's block\n{CELESTRAK.end}"
    readme.write_text(
        f"intro\n{public}\n{SPACETRACK.start}\n{SPACETRACK.end}\n", encoding="utf-8"
    )
    results = tmp_path / "docs" / "screening-benchmark-spacetrack.json"
    args = [
        "--source",
        "spacetrack",
        "--data",
        str(spacetrack_data(tmp_path)),
        "--results",
        str(results),
        "--readme",
        str(readme),
        "--primaries",
        *PRIMARIES,
        "--workers",
        "1",
        "--repeat",
        "1",
    ]
    assert benchmark.main(args) == 0
    assert benchmark.main(["--check", *args]) == 0
    assert calls == [(222, "2026-10-15T00:00:00.000")] * 2
    summary: benchmark.Summary = json.loads(results.read_text(encoding="utf-8"))
    assert summary["settings"] == {
        "source": "spacetrack",
        "primaries": [63216, 58509, 44476],
        "objects": 222,
        "retrieved": "2026-10-15",
        "start": "2026-10-15 00:00",
        "days": 7.0,
        "workers": 1,
    }
    text = readme.read_text(encoding="utf-8")
    assert public in text
    assert (
        "Screening 3 LEO primaries against the 222 element sets of the frozen "
        "full catalog (Space-Track, retrieved 2026-10-15) for 7 days from "
        "2026-10-15 00:00 UTC, in one process (`tools/screening_benchmark.py "
        "--source spacetrack`):"
    ) in text


@pytest.mark.parametrize("name", list(benchmark.SOURCES))
def test_each_source_defaults_to_its_own_committed_files(name: str) -> None:
    source = benchmark.SOURCES[name]
    results = json.loads(source.results.read_text(encoding="utf-8"))
    assert results["settings"]["source"] == name
    readme = README.read_text(encoding="utf-8")
    assert source.start in readme
    assert source.end in readme


def test_the_spacetrack_source_reads_the_catalog_and_its_retrieval_day(
    tmp_path: Path,
) -> None:
    catalog, retrieved = SPACETRACK.read(spacetrack_data(tmp_path))
    assert catalog == FIXTURE
    assert retrieved == "2026-10-15"


def test_an_unknown_source_is_a_usage_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        benchmark.main(["--source", "elsewhere"])
    assert raised.value.code == 2
    assert "argument --source: invalid choice: 'elsewhere'" in capsys.readouterr().err


def test_cpu_model_reads_the_systems_name(tmp_path: Path) -> None:
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor\t: 0\nmodel name\t: Some CPU @ 3 GHz\n")
    assert benchmark.cpu_model(cpuinfo) == "Some CPU @ 3 GHz"


@pytest.mark.parametrize(
    ("processor", "machine", "expected"),
    [
        ("Intel64 Family 6", "AMD64", "Intel64 Family 6"),
        ("", "arm64", "arm64"),
        ("", "", "unknown CPU"),
    ],
)
@pytest.mark.parametrize("cpuinfo", ["processor\t: 0\n", None])
def test_cpu_model_falls_back_to_the_platforms_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    processor: str,
    machine: str,
    expected: str,
    cpuinfo: str | None,
) -> None:
    # Without a model name in /proc/cpuinfo (or without the file: macOS,
    # Windows).
    monkeypatch.setattr(platform, "processor", lambda: processor)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    path = tmp_path / "cpuinfo"
    if cpuinfo is not None:
        path.write_text(cpuinfo)
    assert benchmark.cpu_model(path) == expected


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


@pytest.mark.parametrize(
    "path",
    [
        snapshot.DATA / snapshot.SNAPSHOT,
        spacetrack_snapshot.DATA / spacetrack_snapshot.SNAPSHOT,
    ],
    ids=["celestrak", "spacetrack"],
)
def test_the_primaries_are_in_the_frozen_catalogs(path: Path) -> None:
    text = gzip.decompress(path.read_bytes()).decode()
    numbers = {int(row["NORAD_CAT_ID"]) for row in csv.DictReader(io.StringIO(text))}
    assert set(benchmark.PRIMARIES) <= numbers


def test_the_primaries_are_twenty_distinct_satellites() -> None:
    # "20 primaries" is the performance target the docstrings and README state.
    stated = 20
    assert len(set(benchmark.PRIMARIES)) == len(benchmark.PRIMARIES) == stated


def test_reduced_catalog_keeps_every_kth_object_and_the_primaries() -> None:
    primary = FIXTURE[5].catalog_number
    reduced = benchmark.reduced_catalog(FIXTURE, [primary], 4)
    assert reduced == [
        omm for index, omm in enumerate(FIXTURE) if index % 4 == 0 or index == 5
    ]


def run(cpu: float) -> benchmark.Run:
    return benchmark.Run(wall=cpu + 0.5, cpu=cpu, approaches=7)


def runs(*cpu: float) -> tuple[benchmark.Run, ...]:
    return tuple(run(value) for value in cpu)


def test_render_comparison_takes_medians() -> None:
    comparison = Comparison(runs(1.0, 1.1, 5.0, 1.05), runs(1.0, 1.0, 0.95, 4.0))
    text, regression = benchmark.render_comparison(comparison, "setting", 0.2)
    assert not regression
    assert "| base | 1.00 s | 0.95 - 4.00 s | 1.50 s | 7 |" in text
    assert "| this | 1.08 s | 1.00 - 5.00 s | 1.58 s | 7 |" in text
    assert text.endswith("CPU time change: +7.5% (limit +20%): within the limit.")


def test_render_comparison_finds_a_regression() -> None:
    comparison = Comparison(runs(1.3, 1.25, 1.2), runs(1.0, 1.0, 1.0))
    text, regression = benchmark.render_comparison(comparison, "setting", 0.2)
    assert regression
    assert "+25.0% (limit +20%): slower than allowed." in text


def test_render_comparison_of_a_skipped_base() -> None:
    comparison = Comparison(runs(1.0), (), "ModuleNotFoundError: keepout.screening")
    text, regression = benchmark.render_comparison(comparison, "setting", 0.2)
    assert not regression
    assert text == (
        "Screening benchmark: setting.\n\n"
        "The base cannot run the benchmark, so there is no comparison:\n\n"
        "```text\nModuleNotFoundError: keepout.screening\n```"
    )


def compare_arguments(tmp_path: Path, base_python: str) -> list[str]:
    return [
        "compare",
        "--base-python",
        base_python,
        "--data",
        str(fixture_data(tmp_path)),
        "--primaries",
        *PRIMARIES,
        "--every",
        "1",
        "--days",
        "1",
        "--repeat",
        "1",
    ]


def base_python(tmp_path: Path, cpu: float) -> str:
    """A base printing one measurement whatever it is asked."""
    measurement = {
        "wall": cpu,
        "cpu": cpu,
        "approaches": 1,
        "screened": 1,
        "objects": 1,
    }
    return fake_python(tmp_path, measurement)


def test_compare_main_against_a_faster_base_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    step_summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(step_summary))
    args = compare_arguments(tmp_path, base_python(tmp_path, 1e-6))
    assert benchmark.main(args) == 1
    out = capsys.readouterr().out
    assert out.startswith(
        "Screening benchmark: 3 primaries x 222 objects x 1 day, 5 km sphere, "
        "1 process, 1 run per tree."
    )
    assert "slower than allowed" in out
    assert step_summary.read_text(encoding="utf-8") == out


def test_compare_main_screens_the_chosen_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    commands: list[Sequence[str]] = []

    def recorded(command: Sequence[str]) -> benchmark.Run:
        commands.append(command)
        return run(1.0)

    monkeypatch.setattr(benchmark, "run_timing", recorded)
    args = compare_arguments(tmp_path, "base-python")
    args[args.index("--data") + 1] = str(spacetrack_data(tmp_path))
    assert benchmark.main(["--source", "spacetrack", *args]) == 0
    assert capsys.readouterr().out.startswith(
        "Screening benchmark: 3 primaries x 222 objects x 1 day,"
    )
    head, base = commands
    assert (head[0], base[0]) == (sys.executable, "base-python")
    assert head[1:] == base[1:]
    # The window starts on the Space-Track catalog's retrieval day.
    assert head[head.index("--start") + 1] == "2026-10-15T00:00:00.000"


def test_compare_main_skips_a_base_that_cannot_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    args = compare_arguments(tmp_path, str(tmp_path / "missing" / "python"))
    assert benchmark.main(args) == 0
    assert "The base cannot run the benchmark" in capsys.readouterr().out
