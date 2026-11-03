"""Tests of the words keepout writes in messages and notes."""

import pytest

from keepout._text import before_tca, days_text, pc_text, plural
from keepout.units import DAY, HOUR


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


def test_days_text_and_before_tca_give_the_unit() -> None:
    assert days_text(1.5 * DAY) == "1.50 d"
    assert before_tca(6 * HOUR) == "6.0 h before TCA"
