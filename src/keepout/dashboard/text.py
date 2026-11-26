"""Words of the dashboard's pages: how far a TCA is, a Pc or why there is
none, an event's risk in words, the notes of a plan, and text from data
(CDMs, file names, the query) put in Markdown verbatim.

The pages' own words spell "manoeuvre" (:func:`british`); the names keepout
gives things elsewhere (``maneuver`` actions and jobs, the CLI, JSON) are
unchanged.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

from keepout._text import PC_FAILED
from keepout.maneuver import BurnDirection
from keepout.risk import RiskLevel
from keepout.units import DAY, HOUR, MINUTE, elapsed_seconds

if TYPE_CHECKING:
    from collections.abc import Sequence

    from astropy.time import Time

    from keepout.risk import Action, Assessment

__all__ = [
    "AFTER_TCA",
    "DAYS_FROM",
    "FAILED",
    "HOURS_FROM",
    "MISSING",
    "NOT_COMPUTED",
    "RISK_MARKS",
    "assessment_text",
    "british",
    "countdown",
    "literal",
    "merged_notes",
    "risk_text",
    "unrated",
]

HOURS_FROM: Final = 2 * HOUR
"""Shortest time [s] counted in hours; shorter ones are in minutes."""

DAYS_FROM: Final = 72 * HOUR
"""Time [s] beyond which a time is counted in days."""

NOT_COMPUTED: Final = "not computed"
"""The Pc, and the risk, of a CDM or event without a Pc for the settings."""

FAILED: Final = PC_FAILED
"""The risk of an event whose latest Pc before TCA could not be computed."""

AFTER_TCA: Final = "issued after TCA"
"""The risk of an event whose every CDM was issued after its TCA."""

MISSING: Final = "\N{EN DASH}"
"""A cell of a table without a value."""

RISK_MARKS: Final = {
    RiskLevel.GREEN: "🟢",
    RiskLevel.YELLOW: "🟡",
    RiskLevel.RED: "🔴",
}
"""The mark of each risk level in a table, before its name: colour and
words together."""


_MARKUP: Final = re.compile(r"[\\`*\[\]<>!$~|#{}&]|(?<![0-9A-Za-z])_|_(?![0-9A-Za-z])")
"""The characters Markdown (with Streamlit's LaTeX and GitHub's tables and
strikethrough) reads as markup anywhere in a line: an underscore only at a
word's edge (one inside a word, as in a ``MESSAGE_ID``, is text)."""

_LINE_START: Final = re.compile(r"^(\s*)([-+=]|\d+(?=[.)]))", re.MULTILINE)
"""What makes a line a list item or a heading underline."""


def literal(value: str) -> str:
    """``value`` as Markdown shows it verbatim: its markup characters
    backslash-escaped, so that text from data (a ``MESSAGE_ID``, a file
    name, the query) is shown as it is, never as an image, a link, HTML or
    a formula.

    A bare URL is still a link (GitHub's autolinks), which fetches nothing.
    """
    escaped = _MARKUP.sub(lambda match: "\\" + match[0], value)

    def mark(match: re.Match[str]) -> str:
        indent, start = match[1], match[2]
        return f"{indent}{start}\\" if start[0].isdigit() else f"{indent}\\{start}"

    return _LINE_START.sub(mark, escaped)


def countdown(tca: Time, as_of: Time) -> str:
    """The time from ``as_of`` to ``tca``: ``in 25 min`` under
    :data:`HOURS_FROM`, ``in 30 h`` up to :data:`DAYS_FROM`, ``in 4.5 d``
    beyond; ``... ago`` once past."""
    seconds = elapsed_seconds(as_of, tca)
    # To the millisecond: the difference of two Times is off by microseconds.
    span = round(abs(seconds), 3)
    if span < HOURS_FROM:
        words = f"{round(span / MINUTE)} min"
    elif span <= DAYS_FROM:
        words = f"{round(span / HOUR)} h"
    else:
        words = f"{span / DAY:.1f} d"
    return f"in {words}" if seconds >= 0 else f"{words} ago"


def british(words: str) -> str:
    """Words of keepout as the pages spell them: ``maneuver`` as
    ``manoeuvre``."""
    return words.replace("maneuver", "manoeuvre")


def risk_text(level: RiskLevel, action: Action | None = None) -> str:
    """A risk level in a table, its mark and name, and the action:
    ``🔴 red · manoeuvre``."""
    words = f"{RISK_MARKS[level]} {level.key}"
    return words if action is None else f"{words} · {british(action.label)}"


def assessment_text(assessment: Assessment | None) -> str:
    """An event's risk in a table (:func:`risk_text`); without a level, why
    (:func:`unrated`); without an assessment, :data:`NOT_COMPUTED`."""
    if assessment is None:
        return NOT_COMPUTED
    if assessment.level is None:
        return unrated(assessment)
    return risk_text(assessment.level, assessment.decision.action)


def merged_notes(notes: Sequence[str]) -> list[str]:
    """A plan's notes, each said once: a note of the baseline and of burns
    (``prograde: ...``) becomes one, naming them, ``... (baseline,
    prograde)``."""
    directions = {str(direction) for direction in BurnDirection}
    sources: dict[str, list[str]] = {}
    for note in notes:
        head, colon, rest = note.partition(": ")
        if colon and head in directions:
            sources.setdefault(rest, []).append(head)
        else:
            sources.setdefault(note, []).append("baseline")
    merged: list[str] = []
    for note, of in sources.items():
        if of == ["baseline"]:
            merged.append(note)
        elif len(of) == 1:
            merged.append(f"{of[0]}: {note}")
        else:
            merged.append(f"{note} ({', '.join(of)})")
    return merged


def unrated(assessment: Assessment) -> str:
    """Why an assessed event has no risk level
    (:attr:`~keepout.risk.Assessment.level` is ``None``): every CDM was
    issued after TCA, or the latest one before TCA has no Pc."""
    if assessment.trend.latest_observation is None:
        return AFTER_TCA
    return FAILED
