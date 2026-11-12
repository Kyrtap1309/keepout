"""The examples of ``docs/index.md`` that read the sample CDM, run as written:
each prints the ``text`` block that follows it on the page."""

import contextlib
import io
import itertools
import re
from pathlib import Path
from typing import Final

import pytest

REPO: Final = Path(__file__).parents[1]
PAGE: Final = REPO / "docs" / "index.md"
SAMPLE: Final = "tests/cdm/data/tracss_v2.kvn"


def examples() -> list[tuple[str, str]]:
    """Each ``python`` block that reads the sample CDM, with the ``text`` block
    right after it."""
    page = PAGE.read_text(encoding="utf-8")
    blocks = re.findall(r"^```(\w*)\n(.*?)^```$", page, re.DOTALL | re.MULTILINE)
    return [
        (code, shown)
        for (language, code), (next_language, shown) in itertools.pairwise(blocks)
        if language == "python" and next_language == "text" and SAMPLE in code
    ]


def test_the_page_has_the_examples() -> None:
    assert len(examples()) == 4


@pytest.mark.slow
@pytest.mark.timeout(300)
@pytest.mark.parametrize(("code", "shown"), examples())
def test_an_example_prints_what_the_page_shows(
    code: str, shown: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(REPO)
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        exec(compile(code, str(PAGE), "exec"), {"__name__": "docs_example"})  # noqa: S102
    assert printed.getvalue() == shown
