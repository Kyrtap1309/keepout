"""The walk-through of ``docs/cli.md``, run as written: each command's output
is the block that follows it on the page."""

import os
import re
import shlex
from pathlib import Path
from typing import Any, Final

import pytest
from cli_builders import keepout

from keepout._workers import ONE_THREAD
from keepout.cli.output import ExitCode
from keepout.events import ConjunctionEvent
from keepout.maneuver import ManeuverPlan

REPO: Final = Path(__file__).parents[2]
PAGE: Final = REPO / "docs" / "cli.md"


def documented(command: str) -> tuple[list[str], str]:
    """The arguments of the ``bash`` block starting ``keepout COMMAND``, and
    the ``text`` block after it."""
    page = PAGE.read_text(encoding="utf-8")
    found = re.search(
        rf"```bash\n(keepout {command} .*?)\n```\n\n```text\n(.*?)```",
        page,
        re.DOTALL,
    )
    assert found is not None, f"no example of keepout {command} in {PAGE}"
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
    for name, value in ONE_THREAD.items():
        monkeypatch.setenv(name, os.environ.get(name, value))
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
