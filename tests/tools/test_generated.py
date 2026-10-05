from pathlib import Path

import pytest
from generated import replace_block, write_or_check


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
