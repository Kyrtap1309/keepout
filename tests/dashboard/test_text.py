"""Words of the dashboard's pages: the countdown to TCA, why an event has
no risk level, and text from data shown verbatim in Markdown."""

import astropy.units as u
import pytest
from astropy.time import Time
from builders import series
from hypothesis import example, given
from hypothesis import strategies as st
from markdown_it import MarkdownIt
from markdown_it.token import Token

from keepout.dashboard.text import AFTER_TCA, FAILED, countdown, literal, unrated
from keepout.risk import assess

AS_OF = Time("2026-10-07T12:00:00", scale="utc")


@pytest.mark.parametrize(
    ("seconds", "words"),
    [
        (0.0, "in 0 min"),
        (25 * 60.0, "in 25 min"),
        (2 * 3600.0 - 1.0, "in 120 min"),
        (2 * 3600.0, "in 2 h"),
        (30 * 3600.0, "in 30 h"),
        (72 * 3600.0, "in 72 h"),
        (72 * 3600.0 + 60.0, "in 3.0 d"),
        (4.5 * 86400.0, "in 4.5 d"),
        (-25 * 60.0, "25 min ago"),
        (-2 * 3600.0, "2 h ago"),
        (-72 * 3600.0, "72 h ago"),
        (-4.5 * 86400.0, "4.5 d ago"),
    ],
)
def test_countdown_boundaries(seconds: float, words: str) -> None:
    assert countdown(AS_OF + seconds * u.s, AS_OF) == words


def test_unrated_when_the_latest_pc_failed() -> None:
    assessment = assess(series((2.0, 1e-5), (1.0, None)))
    assert assessment.level is None
    assert unrated(assessment) == FAILED


def test_unrated_when_issued_after_tca() -> None:
    assessment = assess(series((-0.5, 1e-5)))
    assert assessment.level is None
    assert unrated(assessment) == AFTER_TCA


@pytest.mark.parametrize(
    ("value", "shown"),
    [
        ("25544_46275_20261007T161003.846Z", "25544_46275_20261007T161003.846Z"),
        ("2020-001A", "2020-001A"),
        ("![](http://host/x.png)", r"\!\[\](http://host/x.png)"),
        ("[a](http://host)", r"\[a\](http://host)"),
        ("<img src=x>", r"\<img src=x\>"),
        ("$x$ *b* `c` ~d~ |e| #f &amp;", r"\$x\$ \*b\* \`c\` \~d\~ \|e\| \#f \&amp;"),
        ("_a_ a_b", r"\_a\_ a_b"),
        (r"a\b", r"a\\b"),
        ("- a", r"\- a"),
        ("+ a", r"\+ a"),
        ("1. a", r"1\. a"),
        ("2) a", r"2\) a"),
        ("a - b 1. c", "a - b 1. c"),
    ],
)
def test_literal_escapes_markdown(value: str, shown: str) -> None:
    assert literal(value) == shown


_MARKDOWN = MarkdownIt("commonmark").enable(["table", "strikethrough"])
"""CommonMark with GitHub's tables and strikethrough, as Streamlit reads
Markdown (its LaTeX ``$`` is in the examples above)."""

_PLAIN_BLOCKS = {"paragraph_open", "inline", "paragraph_close"}
"""The blocks of a line of plain text: one paragraph."""


def _text(tokens: list[Token]) -> str:
    """The text a paragraph of plain text shows; fail on any other markup."""
    shown = []
    for token in tokens:
        assert token.type in _PLAIN_BLOCKS, token.type
        for child in token.children or []:
            # No link, image, HTML, emphasis, code or line break.
            assert child.type == "text", child.type
            shown.append(child.content)
    return "".join(shown)


@given(
    st.text(
        # A line of text from data: no line breaks or control characters,
        # and no edge whitespace, which Markdown drops from a paragraph.
        st.characters(blacklist_categories=("Cc", "Cs", "Zl", "Zp")),
        min_size=1,
    ).map(str.strip)
)
@example("![x](http://host/x.png) <b>a</b> *b* `c` ~~d~~ | e | &amp; \\")
@example("# a")
@example("> a")
@example("1. a")
@example("---")
def test_literal_shows_any_text_as_plain_text(value: str) -> None:
    assert _text(_MARKDOWN.parse(literal(value))) == value
