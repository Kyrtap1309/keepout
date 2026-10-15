import json
import resource
import subprocess
import sys

import pytest
import screening_snapshot as snapshot
import screening_timing as timing
from astropy.time import Time

from keepout._units import DAY
from keepout.omm import load
from keepout.screening import DEFAULT_VOLUME, screen

CATALOG = snapshot.DATA / snapshot.FIXTURE
START = "2026-10-06T00:00:00"


def test_main_prints_the_measurement(capsys: pytest.CaptureFixture[str]) -> None:
    arguments = [str(CATALOG), "--primaries", "63216", "58509", "--start", START]
    assert timing.main([*arguments, "--days", "1"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert set(printed) == {"wall", "cpu", "approaches", "screened", "objects"}
    assert printed["wall"] > 0.0
    assert printed["cpu"] > 0.0
    catalog = load(CATALOG)
    by_number = {omm.catalog_number: omm for omm in catalog}
    expected = screen(
        [by_number[63216], by_number[58509]],
        catalog,
        Time(START, scale="utc"),
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
    catalog = load(CATALOG)
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


def test_timed_returns_the_screening() -> None:
    catalog = load(CATALOG)
    start = Time(START, scale="utc")
    result, measured = timing.timed(
        catalog[:2], catalog, start, 3600.0, DEFAULT_VOLUME, workers=1
    )
    assert result.start == start
    assert result.duration == 3600.0
    assert measured.wall > 0.0
    assert measured.cpu >= 0.0
