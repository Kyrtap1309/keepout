"""The walk-through of ``docs/cli.md``, run as written: each command's output
is the block that follows it on the page; likewise the JSON sample of
``docs/cli-json.md`` and the help of ``keepout dashboard`` in
``docs/dashboard.md``."""

import json
import math
import re
import shlex
from pathlib import Path
from typing import Any, Final

import pytest
from cli_builders import keepout

from keepout.cli.output import ExitCode
from keepout.events import ConjunctionEvent
from keepout.maneuver import ManeuverPlan

REPO: Final = Path(__file__).parents[2]
PAGE: Final = REPO / "docs" / "cli.md"


def documented(command: str, page_path: Path = PAGE) -> tuple[list[str], str]:
    """The arguments of the ``bash`` block starting ``keepout COMMAND`` on
    the page ``page_path``, and the ``text`` block after it."""
    page = page_path.read_text(encoding="utf-8")
    found = re.search(
        rf"```bash\n(keepout {command} .*?)\n```\n\n```text\n(.*?)```",
        page,
        re.DOTALL,
    )
    assert found is not None, f"no example of keepout {command} in {page_path}"
    line = found[1].replace("\\\n", " ")
    return shlex.split(line)[1:], found[2]


def run(arguments: list[str]) -> tuple[int, str]:
    """``keepout ARGUMENTS`` with shell globs expanded, and its output without
    trailing blanks (Rich pads table rows)."""
    expanded = [
        path
        for item in arguments
        for path in sorted(str(found) for found in Path().glob(item)) or [item]
    ]
    result = keepout(*expanded)
    lines = [line.rstrip() for line in result.stdout.splitlines()]
    return result.exit_code, "\n".join(lines) + "\n"


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_the_walk_through_prints_what_the_page_shows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # From a clone of the repository: its paths, and the CDMs in a fresh folder.
    for folder in ("examples", "tests"):
        (tmp_path / folder).symlink_to(REPO / folder)
    monkeypatch.chdir(tmp_path)
    screen, screened = documented("screen")
    assert run(screen) == (ExitCode.OK, screened)
    status, assessed = run(["assess", "cdms/*.kvn", "--hbr", "1000"])
    assert status == ExitCode.MANEUVER
    summary = "12 events from 12 CDMs: 2 maneuver, 4 prepare maneuver, 1 monitor, "
    assert f"{summary}5 ignore." in assessed
    maneuver, planned = documented("maneuver")
    assert run(maneuver) == (ExitCode.OK, planned)
    add, added = documented("store add")
    assert run(add) == (ExitCode.OK, added)

    # The stored event the page plans, queued and computed by the worker. Its
    # child process imports keepout afresh and plans on the full default grid.
    submit, queued = documented("maneuver --event")
    assert run(submit) == (ExitCode.OK, queued)
    worker, ran = documented("worker")
    assert run(worker) == (ExitCode.OK, ran)

    # Then shown from the store's cache: this process never plans it.
    def unplanned(event: ConjunctionEvent, **settings: Any) -> ManeuverPlan:
        raise AssertionError(f"{event.event_id} planned again, not read from the store")

    monkeypatch.setattr("keepout.maneuver.outcome.plan_maneuver", unplanned)
    page = PAGE.read_text(encoding="utf-8")
    stored = re.search(r"`(keepout maneuver --event [^`]*)`", page)
    assert stored is not None
    status, planned = run(shlex.split(stored[1])[1:])
    assert status == ExitCode.OK
    assert planned.startswith("Event 3befe6f1305d: 44476 / 55841, TCA 2026-10-08 12:03")
    assert "\nNot re-screened: nothing is known" in planned
    purge, purged = documented("store purge")
    assert run(purge) == (ExitCode.OK, purged)


JSON_PAGE: Final = REPO / "docs" / "cli-json.md"


def _close(found: object, shown: object) -> bool:
    """Whether ``found`` is the JSON value ``shown``, numbers to 1e-9
    relative (a Pc's last digits may differ between machines)."""
    if isinstance(shown, float) and isinstance(found, float):
        return math.isclose(found, shown, rel_tol=1e-9)
    if isinstance(shown, dict) and isinstance(found, dict):
        return found.keys() == shown.keys() and all(
            _close(found[key], shown[key]) for key in shown
        )
    if isinstance(shown, list) and isinstance(found, list):
        return len(found) == len(shown) and all(
            _close(a, b) for a, b in zip(found, shown, strict=True)
        )
    return found == shown


def test_the_json_sample_is_what_assess_prints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = JSON_PAGE.read_text(encoding="utf-8")
    found = re.search(
        r"`(keepout assess --json [^`]*)`.*?```json\n(.*?)```", page, re.DOTALL
    )
    assert found is not None, f"no JSON sample of keepout assess in {JSON_PAGE}"
    monkeypatch.chdir(REPO)
    result = keepout(*shlex.split(found[1])[1:])
    assert result.exit_code == ExitCode.MANEUVER
    assert _close(json.loads(result.stdout), json.loads(found[2]))


DASHBOARD_PAGE: Final = REPO / "docs" / "dashboard.md"


def test_the_dashboard_page_shows_its_help() -> None:
    arguments, shown = documented("dashboard", DASHBOARD_PAGE)
    assert arguments == ["dashboard", "--help"]
    status, printed = run(arguments)
    assert status == ExitCode.OK
    # The page leaves out the blank lines around the help.
    assert printed.strip("\n") == shown.rstrip("\n")
