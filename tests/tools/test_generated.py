import argparse
import importlib
import json
from pathlib import Path
from typing import Literal, TypedDict

import pytest
from generated import (
    REPO,
    day_text,
    generated_notice,
    json_text,
    minute_text,
    parse_json,
    pc_text,
    plural,
    positive,
    read_json,
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


def test_every_tool_with_a_workers_option_is_listed() -> None:
    with_workers = {
        tool.stem
        for tool in (REPO / "tools").glob("*.py")
        if '"--workers"' in tool.read_text(encoding="utf-8")
    }
    assert with_workers - {"screening_timing"} == set(WORKERS_ARGUMENTS)


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


@pytest.mark.parametrize(
    ("count", "nouns", "text"),
    [(1, None, "1 run"), (2, None, "2 runs"), (0, "minima", "0 minima")],
)
def test_plural_names_one_or_many(count: int, nouns: str | None, text: str) -> None:
    noun = "minimum" if nouns else "run"
    assert plural(count, noun, nouns) == text


def test_pc_text_gives_three_digits_or_a_dash() -> None:
    assert pc_text(1.234e-5) == "1.23e-05"
    assert pc_text(None) == "-"


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
