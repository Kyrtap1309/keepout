"""The changelog section a release's notes are made of."""

from pathlib import Path

import changelog_section as section
import pytest
from generated import REPO

import keepout

CHANGELOG = """# Changelog

Intro.

## [Unreleased]

## [1.1.0] - 2027-01-05

### Added

- Something.

## [1.0.0] - 2026-12-01

### Added

- Everything.

[Unreleased]: https://example.invalid/compare/v1.1.0...HEAD
[1.1.0]: https://example.invalid/releases/tag/v1.1.0
"""


def test_section_extracted() -> None:
    assert section.changelog_section(CHANGELOG, "1.1.0") == "### Added\n\n- Something."


def test_last_section_stops_at_the_links() -> None:
    assert section.changelog_section(CHANGELOG, "1.0.0") == "### Added\n\n- Everything."


def test_absent_version_fails() -> None:
    with pytest.raises(ValueError, match=r"no section for 2\.0\.0"):
        section.changelog_section(CHANGELOG, "2.0.0")


def test_empty_section_fails() -> None:
    with pytest.raises(ValueError, match="empty"):
        section.changelog_section(CHANGELOG.replace("- Something.", ""), "1.1.0")


def test_undated_section_fails() -> None:
    undated = CHANGELOG.replace("[1.1.0] - 2027-01-05", "[1.1.0] - YYYY-MM-DD")
    with pytest.raises(ValueError, match="date"):
        section.changelog_section(undated, "1.1.0")


def test_main_prints_the_section(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    path = tmp_path / "changelog.md"
    path.write_text(CHANGELOG, encoding="utf-8")
    assert section.main(["1.0.0", "--changelog", str(path)]) == 0
    assert capsys.readouterr().out == "### Added\n\n- Everything.\n"


def test_the_committed_changelog_has_the_version_s_dated_section() -> None:
    text = (REPO / "docs/changelog.md").read_text(encoding="utf-8")
    assert section.changelog_section(text, keepout.__version__).strip()


def test_relative_links_point_at_the_site() -> None:
    body = (
        "See [CDMs](cdm.md), [upgrading](store.md#upgrading) and [x](https://a.b/c.md)."
    )
    assert section.absolute_links(body, "https://site.invalid/keepout/") == (
        "See [CDMs](https://site.invalid/keepout/cdm/), "
        "[upgrading](https://site.invalid/keepout/store/#upgrading) and "
        "[x](https://a.b/c.md)."
    )


def test_main_prints_site_links(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    path = tmp_path / "changelog.md"
    path.write_text(
        CHANGELOG.replace("- Everything.", "- [CDMs](cdm.md)."), encoding="utf-8"
    )
    assert section.main(["1.0.0", "--changelog", str(path)]) == 0
    assert f"[CDMs]({section.SITE}cdm/)" in capsys.readouterr().out


def test_site_is_mkdocs_site_url() -> None:
    text = (REPO / "mkdocs.yml").read_text(encoding="utf-8")
    assert f"site_url: {section.SITE}\n" in text
