"""The changelog section of a version: the notes of its GitHub Release.

Usage::

    uv run python tools/changelog_section.py VERSION [--changelog PATH]

Prints the body under ``## [VERSION] - DATE`` in ``docs/changelog.md``
(Keep a Changelog format), without the heading, up to the next ``## ``
heading or the link references at the end, its links to ``docs/`` pages
made links to the documentation site. The release workflow runs it on the
tag's version.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final

from generated import REPO

if TYPE_CHECKING:
    from collections.abc import Sequence

CHANGELOG: Final = REPO / "docs/changelog.md"
"""The changelog."""

_DATE: Final = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
"""A release date, as Keep a Changelog writes it."""

_LINK: Final = re.compile(r"^\[[^\]]+\]: ", re.MULTILINE)
"""A link reference line, which ends the last section."""

SITE: Final = "https://kyrtap1309.github.io/keepout/"
"""The documentation site (``site_url`` in ``mkdocs.yml``)."""

_PAGE_LINK: Final = re.compile(r"\]\((?P<page>[a-z0-9_./-]+)\.md(?P<anchor>#[^)]*)?\)")
"""A Markdown link to a page of ``docs/`` by its relative path."""


def changelog_section(text: str, version: str) -> str:
    """The body of ``version``'s section of the changelog ``text``.

    Raises
    ------
    ValueError
        If there is no section for ``version``, its heading has no date
        (``YYYY-MM-DD`` still unset), or its body has nothing but headings.
    """
    heading = re.search(
        rf"^## \[{re.escape(version)}\](?: - (?P<date>\S+))?[ \t]*$", text, re.MULTILINE
    )
    if heading is None:
        raise ValueError(f"the changelog has no section for {version}")
    date = heading["date"]
    if date is None or not _DATE.fullmatch(date):
        raise ValueError(f"the section for {version} has no release date: {date}")
    rest = text[heading.end() :]
    ends = [
        match.start()
        for match in (re.search(r"^## ", rest, re.MULTILINE), _LINK.search(rest))
        if match is not None
    ]
    body = rest[: min(ends, default=len(rest))].strip()
    entries = [
        line for line in body.splitlines() if line.strip() and not line.startswith("#")
    ]
    if not entries:
        raise ValueError(f"the section for {version} is empty")
    return body


def absolute_links(body: str, site: str) -> str:
    """``body`` with each link to a ``docs/`` page made a link to that page
    on the documentation ``site``: a release's notes are read on GitHub,
    where the relative paths lead nowhere."""

    def page(match: re.Match[str]) -> str:
        return f"]({site}{match['page']}/{match['anchor'] or ''})"

    return _PAGE_LINK.sub(page, body)


def main(argv: Sequence[str] | None = None) -> int:
    """The entry point: print the section.

    Returns
    -------
    int
        0; a missing, undated or empty section raises.
    """
    parser = argparse.ArgumentParser(prog="tools/changelog_section.py")
    parser.add_argument("version")
    parser.add_argument("--changelog", type=Path, default=CHANGELOG)
    parser.add_argument("--site", default=SITE, help="where relative links lead")
    args = parser.parse_args(argv)
    body = changelog_section(args.changelog.read_text(encoding="utf-8"), args.version)
    print(absolute_links(body, args.site))
    return 0


if __name__ == "__main__":
    sys.exit(main())
