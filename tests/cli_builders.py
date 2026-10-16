"""Helpers of the command-line tests: run ``keepout`` in-process, fake the
web and Space-Track, and a fleet with approaches in the screening fixture."""

import io
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pytest
from screening_data import DATA, FIXTURE
from spacetrack_builders import IDENTITY, PASSWORD, FakeSpaceTrack
from typer.testing import CliRunner, Result

from keepout.cli.app import app
from keepout.omm.celestrak import dumps_csv
from keepout.sources import celestrak, spacetrack


def keepout(*args: str, env: Mapping[str, str | None] | None = None) -> Result:
    """Run ``keepout ARGS`` in-process; standard output and error kept apart.

    A bug (any exception but :class:`SystemExit`) is raised, not hidden in
    the result. ``env`` sets (or, with ``None``, unsets) environment
    variables for the run.
    """
    runner = CliRunner()
    return runner.invoke(app, list(args), env=env, catch_exceptions=False)


class _Response(io.BytesIO):
    """What ``urlopen`` returns: a body with status 200."""

    status = 200
    reason = "OK"


class FakeWeb:
    """Serves fixed bodies by URL in place of ``urllib.request.urlopen``.

    Records the URLs asked for; a URL it does not serve fails as an
    unreachable server does.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, pages: Mapping[str, bytes]):
        self.pages = dict(pages)
        self.urls: list[str] = []
        monkeypatch.setattr(urllib.request, "urlopen", self._open)

    def _open(self, request: urllib.request.Request, timeout: float) -> _Response:
        url = request.full_url
        self.urls.append(url)
        if url not in self.pages:
            raise OSError(f"cannot reach {url}")
        return _Response(self.pages[url])


def fake_space_track(
    monkeypatch: pytest.MonkeyPatch, answers: Mapping[str, bytes]
) -> FakeSpaceTrack:
    """A :class:`FakeSpaceTrack` serving ``answers`` (query URL to body),
    used by every Space-Track client the commands open."""
    server = FakeSpaceTrack(answers)
    monkeypatch.setattr(spacetrack, "UrllibTransport", lambda: server)
    return server


SPACE_TRACK_ENV: Final[Mapping[str, str | None]] = {
    spacetrack.IDENTITY_VARIABLE: IDENTITY,
    spacetrack.PASSWORD_VARIABLE: PASSWORD,
}
"""The fake server's account, as the commands read it."""

NO_SPACE_TRACK_ENV: Final[Mapping[str, str | None]] = {
    spacetrack.IDENTITY_VARIABLE: None,
    spacetrack.PASSWORD_VARIABLE: None,
}
"""No Space-Track account in the environment."""

FIXTURE_CSV: Final = DATA / FIXTURE
"""The screening fixture: about 200 objects of the 2026-10-06 catalog."""


def celestrak_pages() -> dict[str, bytes]:
    """CelesTrak's public groups: the fixture as ``active``, the others empty."""
    empty = dumps_csv(()).encode()
    pages = {celestrak.group_url(name): empty for name in celestrak.PUBLIC_GROUPS}
    pages[celestrak.group_url("active")] = FIXTURE_CSV.read_bytes()
    return pages


START: Final = "2026-10-06"
"""Start of the screening window of the tests."""

DAYS: Final = "3"
"""Length of the screening window of the tests, days."""

FLEET_TOML: Final = """\
catalog_hard_body_radius = 1.0

[[satellites]]
catalog_number = 52939
name = "FLEET ONE"
hard_body_radius = 5.0
maneuverable = true

[[satellites]]
catalog_number = 44476
hard_body_radius = 20.0
maneuverable = false
"""
"""A fleet with two approaches in the fixture over :data:`DAYS` from
:data:`START`: a LEO satellite crossing a Starlink at 6 km/s (green) and
Intelsat 39 drifting past Luch-5X (yellow)."""


def write_fleet(folder: Path, text: str = FLEET_TOML) -> Path:
    """``text`` written to ``folder/fleet.toml``."""
    path = folder / "fleet.toml"
    path.write_text(text, encoding="utf-8")
    return path
