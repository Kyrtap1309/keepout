import json
import resource
import subprocess
import sys
from types import SimpleNamespace

import pytest
import screening_snapshot as snapshot
import screening_timing as timing
from screening_data import FIXTURE_START, fixture_by_number, load_fixture

from keepout.screening import DEFAULT_VOLUME, screen
from keepout.units import DAY

CATALOG = snapshot.DATA / snapshot.FIXTURE
START = FIXTURE_START.isot


def test_main_prints_the_measurement(capsys: pytest.CaptureFixture[str]) -> None:
    arguments = [str(CATALOG), "--primaries", "63216", "58509", "--start", START]
    assert timing.main([*arguments, "--days", "1"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert set(printed) == {"wall", "cpu", "approaches", "screened", "objects"}
    assert printed["wall"] > 0.0
    assert printed["cpu"] > 0.0
    catalog = load_fixture()
    by_number = fixture_by_number()
    expected = screen(
        [by_number[63216], by_number[58509]],
        catalog,
        FIXTURE_START,
        DAY,
        DEFAULT_VOLUME,
    )
    assert printed["approaches"] == len(expected.approaches) > 0
    assert printed["screened"] == expected.catalog_count
    assert printed["objects"] == len(catalog) == 222


def test_main_refuses_primaries_outside_the_catalog(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        timing.main([str(CATALOG), "--primaries", "63216", "1", "--start", START])
    assert raised.value.code == 2
    assert "primaries not in" in capsys.readouterr().err


def test_select_refuses_missing_primaries() -> None:
    catalog = load_fixture()
    assert timing.select(catalog, [63216]) == [
        omm for omm in catalog if omm.catalog_number == 63216
    ]
    with pytest.raises(ValueError, match=r"primaries not in the catalog: \[1\]"):
        timing.select(catalog, [63216, 1])


def children_cpu_time() -> float:
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return children.ru_utime + children.ru_stime


def test_cpu_time_counts_finished_children() -> None:
    before, children_before = timing.cpu_time(), children_cpu_time()
    subprocess.run(
        [sys.executable, "-c", "sum(i * i for i in range(3_000_000))"], check=True
    )
    children = children_cpu_time() - children_before
    assert children > 0.0
    # Counted whatever the machine's speed; 1 ns for rounding of the sums.
    assert timing.cpu_time() - before >= children - 1e-9


def test_timed_returns_the_screening_and_the_times_it_took(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clocks = {"wall": iter([100.0, 101.5]), "cpu": iter([10.0, 12.5])}
    monkeypatch.setattr(
        timing, "time", SimpleNamespace(perf_counter=lambda: next(clocks["wall"]))
    )
    monkeypatch.setattr(timing, "cpu_time", lambda: next(clocks["cpu"]))
    catalog = load_fixture()
    result, measured = timing.timed(
        catalog[:2], catalog, FIXTURE_START, 3600.0, DEFAULT_VOLUME, workers=1
    )
    assert result.start == FIXTURE_START
    assert result.duration == 3600.0
    assert measured == timing.Timing(wall=1.5, cpu=2.5)
