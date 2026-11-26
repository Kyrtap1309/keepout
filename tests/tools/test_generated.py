import argparse
import datetime
import importlib
import json
import platform
from pathlib import Path
from typing import Literal, TypedDict

import pytest
from generated import (
    REPO,
    Machine,
    cpu_model,
    day_text,
    generated_notice,
    json_text,
    machine,
    machine_text,
    minute_text,
    parse_json,
    positive,
    read_json,
    read_timings,
    readme_with,
    replace_block,
    second_text,
    significant,
    threshold_settings,
    threshold_text,
    write_or_check,
)

from keepout.risk import Thresholds


def test_replace_block_rewrites_only_the_marked_block() -> None:
    text = "intro\n<s>\nold\n<e>\nrest\n"
    assert replace_block(text, "<s>", "<e>", "new") == "intro\n<s>\nnew\n<e>\nrest\n"


@pytest.mark.parametrize("text", ["no markers", "<e>\n<s>\n", "<s> only"])
def test_replace_block_needs_both_markers_in_order(text: str) -> None:
    with pytest.raises(ValueError, match="README needs <s> and <e>, in that order"):
        replace_block(text, "<s>", "<e>", "new")


@pytest.fixture
def files(tmp_path: Path) -> dict[Path, str]:
    """An up-to-date file and a missing one in a missing directory."""
    same = tmp_path / "same.md"
    same.write_text("same", encoding="utf-8")
    return {same: "same", tmp_path / "sub" / "stale.md": "fresh"}


def test_check_reports_a_stale_file_without_writing(
    files: dict[Path, str], capsys: pytest.CaptureFixture[str]
) -> None:
    same, stale = files
    before = same.stat().st_mtime_ns
    assert write_or_check(files, check=True, tool="tool.py") == 1
    assert capsys.readouterr().out == f"{stale} is out of date; run tool.py\n"
    assert not stale.exists()
    assert same.stat().st_mtime_ns == before


def test_check_shows_how_a_stale_text_file_differs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stale = tmp_path / "stale.json"
    stale.write_text('{\n  "a": 1,\n  "b": 2\n}\n', encoding="utf-8")
    fresh = '{\n  "a": 1,\n  "b": 3\n}\n'
    assert write_or_check({stale: fresh}, check=True, tool="tool.py") == 1
    assert capsys.readouterr().out == (
        f"{stale} is out of date; run tool.py\n"
        f"--- {stale} (committed)\n"
        f"+++ {stale} (generated)\n"
        "@@ -1,4 +1,4 @@\n"
        " {\n"
        '   "a": 1,\n'
        '-  "b": 2\n'
        '+  "b": 3\n'
        " }\n"
    )


def test_write_creates_stale_files_and_leaves_current_ones(
    files: dict[Path, str],
) -> None:
    same, stale = files
    before = same.stat().st_mtime_ns
    assert write_or_check(files, check=False, tool="tool.py") == 0
    assert stale.read_text(encoding="utf-8") == "fresh"
    assert same.stat().st_mtime_ns == before
    assert write_or_check(files, check=True, tool="tool.py") == 0


@pytest.mark.parametrize(("text", "value"), [("1", 1), ("12", 12)])
def test_a_positive_count_is_read(text: str, value: int) -> None:
    assert positive(text) == value


@pytest.mark.parametrize("text", ["0", "-3"])
def test_a_count_below_one_is_refused(text: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="must be at least 1"):
        positive(text)


WORKERS_ARGUMENTS = {
    "maneuver_report": ["--workers", "0"],
    "screening_benchmark": ["--workers", "0"],
    "screening_report": ["--workers", "0"],
    "spacetrack_snapshot": ["compare", "raw", "--workers", "0"],
    "synthetic_operator_report": ["--workers", "0"],
    "validation_report": ["--workers", "0"],
}
"""Each tool with a ``--workers`` option, and a command line giving it 0.

``screening_timing.py`` is left out: ``screening_benchmark.py compare`` runs it
with an older tree's keepout, so it imports nothing else from ``tools/``
(``positive`` included), and that tool always passes ``--workers 1``."""


def has_workers(text: str) -> bool:
    """Whether a tool's ``text`` adds ``--workers``: itself, or with
    :func:`generated.report_parser` (unless told not to)."""
    shared = "report_parser(" in text and "workers=False" not in text
    return '"--workers"' in text or shared


def test_every_tool_with_a_workers_option_is_listed() -> None:
    with_workers = {
        tool.stem
        for tool in (REPO / "tools").glob("*.py")
        if has_workers(tool.read_text(encoding="utf-8"))
    }
    assert with_workers - {"generated", "screening_timing"} == set(WORKERS_ARGUMENTS)


@pytest.mark.parametrize("name", sorted(WORKERS_ARGUMENTS))
def test_every_workers_option_takes_only_positive_counts(
    name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = importlib.import_module(name)
    with pytest.raises(SystemExit) as raised:
        tool.main(WORKERS_ARGUMENTS[name])
    assert raised.value.code == 2
    assert "argument --workers: must be at least 1" in capsys.readouterr().err


class Inner(TypedDict):
    name: Literal["a", "b"]
    sizes: list[float]


class Outer(TypedDict):
    count: int
    flag: bool
    inner: Inner | None
    by_name: dict[str, Inner]


GOOD: Outer = {
    "count": 1,
    "flag": True,
    "inner": {"name": "a", "sizes": [1.5, 2]},
    "by_name": {"x": {"name": "b", "sizes": []}},
}


@pytest.mark.parametrize("value", [GOOD, {**GOOD, "inner": None}])
def test_json_of_the_shape_is_parsed(value: Outer) -> None:
    assert parse_json(json_text(value), Outer, "text") == value


@pytest.mark.parametrize(
    "value",
    [
        [],
        {**GOOD, "count": 1.0},
        {**GOOD, "flag": 1},
        {**GOOD, "inner": {"name": "c", "sizes": []}},
        {**GOOD, "inner": {"name": "a", "sizes": ["1"]}},
        {**GOOD, "inner": {"name": "a"}},
        {**GOOD, "by_name": {"x": None}},
        {**GOOD, "extra": 0},
        {key: value for key, value in GOOD.items() if key != "flag"},
    ],
)
def test_json_of_another_shape_is_refused(value: object) -> None:
    with pytest.raises(ValueError, match=r"^text is not a Outer: "):
        parse_json(json_text(value), Outer, "text")


def test_text_that_is_not_json_is_refused() -> None:
    with pytest.raises(json.JSONDecodeError):
        parse_json("{", Outer, "text")


def test_read_json_gives_a_file_of_the_shape(tmp_path: Path) -> None:
    path = tmp_path / "good.json"
    path.write_text(json_text(GOOD), encoding="utf-8")
    assert read_json(path, Outer) == GOOD


def test_read_json_refuses_a_file_of_another_shape(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json_text({**GOOD, "count": "1"}), encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.json is not a Outer"):
        read_json(path, Outer)


def test_json_text_is_indented_with_a_final_newline() -> None:
    assert json_text({"a": [1]}) == '{\n  "a": [\n    1\n  ]\n}\n'


def test_readme_with_puts_the_block_in_the_file_text(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("intro\n<s>\nold\n<e>\n", encoding="utf-8")
    assert readme_with(readme, "<s>", "<e>", "new") == "intro\n<s>\nnew\n<e>\n"


def test_day_minute_and_second_text_cut_an_iso_time() -> None:
    iso = "2026-10-06T12:34:56.789"
    assert day_text(iso) == "2026-10-06"
    assert minute_text(iso) == "2026-10-06 12:34"
    assert second_text(iso) == "2026-10-06 12:34:56"


def test_significant_keeps_the_leading_digits() -> None:
    assert significant(0.012345) == 0.0123
    assert significant(123456.0, 2) == 120000.0


def test_generated_notice_names_the_tool() -> None:
    assert generated_notice("tools/x.py") == (
        "<!-- Generated by tools/x.py; do not edit. -->"
    )


def test_thresholds_are_recorded_and_written() -> None:
    thresholds = Thresholds()
    settings = threshold_settings(thresholds)
    assert settings == {
        "green_below": thresholds.green_below,
        "red_from": thresholds.red_from,
        "basis": thresholds.basis,
    }
    assert threshold_text(1e-7, 1e-4) == "green below 1e-07, red from 1e-04"


def test_cpu_model_reads_the_systems_name(tmp_path: Path) -> None:
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor\t: 0\nmodel name\t: Some CPU @ 3 GHz\n")
    assert cpu_model(cpuinfo) == "Some CPU @ 3 GHz"


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
    assert cpu_model(path) == expected


def test_the_machine_is_this_one_today() -> None:
    measured = machine()
    assert measured["cpu_model"] == cpu_model()
    assert measured["python"] == platform.python_version()
    assert measured["logical_cpus"] >= 1
    assert measured["date"] == datetime.datetime.now(datetime.UTC).date().isoformat()


def test_machine_text_names_where_and_when() -> None:
    measured: Machine = {
        "cpu_model": "Some CPU",
        "logical_cpus": 8,
        "python": "3.13.7",
        "date": "2026-11-02",
    }
    assert machine_text(measured) == (
        "2026-11-02, Some CPU (8 logical CPUs), Python 3.13.7"
    )


class Measurement(TypedDict):
    seconds: float


def test_read_timings_reads_only_the_timings(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    path.write_text(json_text({"other": [1, "a"], "timings": {"seconds": 1.5}}))
    assert read_timings(path, Measurement) == {"seconds": 1.5}


def test_read_timings_of_a_missing_file_or_null_timings_are_none(
    tmp_path: Path,
) -> None:
    path = tmp_path / "results.json"
    assert read_timings(path, Measurement) is None
    path.write_text(json_text({"timings": None}))
    assert read_timings(path, Measurement) is None


def test_read_timings_checks_the_timings(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    path.write_text(json_text({"timings": {"seconds": "slow"}}))
    with pytest.raises(ValueError, match=r"the timings of .* is not a Measurement"):
        read_timings(path, Measurement)


def test_read_timings_needs_timings(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    path.write_text(json_text({"other": 1}))
    with pytest.raises(ValueError, match="has no timings"):
        read_timings(path, Measurement)
