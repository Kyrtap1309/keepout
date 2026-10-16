"""Helpers of the command-line tests: run ``keepout`` in-process, fake the web."""

import io
import urllib.request
from collections.abc import Mapping

import pytest
from typer.testing import CliRunner, Result

from keepout.cli.app import app


def keepout(*args: str, env: Mapping[str, str] | None = None) -> Result:
    """Run ``keepout ARGS`` in-process; standard output and error kept apart.

    A bug (any exception but :class:`SystemExit`) is raised, not hidden in
    the result.
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
