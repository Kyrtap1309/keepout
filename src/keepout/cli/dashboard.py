"""``keepout dashboard``: Streamlit run on the dashboard's app
(:mod:`keepout.dashboard.app`), on a store.

Neither the dashboard's modules nor Streamlit are imported here: the app's
file is found with :func:`importlib.util.find_spec`, and Streamlit runs it
in a process of its own. ``keepout.store`` is imported in the function only, to
check the store before Streamlit starts.
"""

from __future__ import annotations

import importlib.util
import ipaddress
import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Final, TypeAlias

import typer

from keepout._environment import (
    AS_OF_VARIABLE,
    MAX_UPLOAD_MB,
    SETTINGS_VARIABLE,
    STORE_VARIABLE,
)
from keepout.cli.options import SettingsFile, operator_settings, time_option
from keepout.cli.output import ExitCode, fail, isot
from keepout.cli.store import StorePath, open_cli_store

__all__ = ["DASHBOARD_HINT", "DEFAULT_HOST", "EXPOSED_WARNING", "dashboard_command"]

DASHBOARD_HINT: Final = (
    "keepout dashboard needs the dashboard extra: "
    'pip install "keepout[cli,dashboard] @ git+https://github.com/Kyrtap1309/keepout"'
)
"""What ``keepout dashboard`` prints (exit status 1) when the extra is
missing."""

_DASHBOARD_MODULES: Final = ("streamlit", "plotly")
"""The top-level packages of the ``dashboard`` extra."""

DEFAULT_HOST: Final = "127.0.0.1"
"""The address the dashboard listens on unless ``--host`` gives another:
this machine only."""

EXPOSED_WARNING: Final = (
    "Warning: the dashboard listens on {host}, which other machines may "
    "reach; it has no authentication (see the deployment docs)."
)
"""What ``keepout dashboard`` prints on standard error when ``--host`` is
not a loopback address."""

Host: TypeAlias = Annotated[
    str,
    typer.Option(
        "--host",
        metavar="ADDRESS",
        help="The IP address to listen on (default: this machine only).",
    ),
]
"""``--host``: the IP address the dashboard listens on."""

Port: TypeAlias = Annotated[
    int,
    typer.Option("--port", min=1, max=65535, metavar="N", help="The port to serve on."),
]
"""``--port``: the dashboard's TCP port."""

AsOf: TypeAlias = Annotated[
    str | None,
    typer.Option(
        "--as-of",
        metavar="TIME",
        help=(
            "See the data at this time, ISO 8601 UTC: the events whose latest "
            "TCA is before it are past (default: the clock, read on each page)."
        ),
    ),
]
"""``--as-of``: the time the data are seen at, read by
:func:`keepout._environment.parse_time`, as the dashboard reads it;
``None``: the clock."""


def _app_path() -> Path:
    """The dashboard's entry script, found without running it.

    Raises
    ------
    typer.Exit
        Without the ``dashboard`` extra: :data:`DASHBOARD_HINT`, status 1;
        or if this keepout has no dashboard app.
    """
    if any(importlib.util.find_spec(name) is None for name in _DASHBOARD_MODULES):
        fail(DASHBOARD_HINT)
    # Loads keepout.dashboard's __init__, which imports nothing, not the app.
    spec = importlib.util.find_spec("keepout.dashboard.app")
    if spec is None or spec.origin is None:
        fail("this keepout has no dashboard app: reinstall keepout")
    return Path(spec.origin)


def _host(value: str) -> str:
    """The ``--host`` address ``value``, warned about on standard error if
    it is not a loopback address.

    Raises
    ------
    typer.BadParameter
        If ``value`` is not an IP address.
    """
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        raise typer.BadParameter(
            f"{value!r} is not an IP address", param_hint="--host"
        ) from None
    if not address.is_loopback:
        typer.echo(EXPOSED_WARNING.format(host=value), err=True)
    return value


def dashboard_command(
    *,
    store: StorePath,
    settings: SettingsFile = None,
    host: Host = DEFAULT_HOST,
    port: Port = 8501,
    as_of: AsOf = None,
) -> None:
    """Run the dashboard on a store (needs the dashboard extra).

    Serves the dashboard on http://HOST:PORT (127.0.0.1, this machine only,
    by default) until stopped with Ctrl+C: the events, each event's risk
    and the maneuver planner. It reads the store and queues jobs for
    keepout worker; it computes no Pc or plan itself. The store must exist
    and be at this keepout's schema (keepout store upgrade). Exits with
    Streamlit's status.
    """
    address = _host(host)
    when = None if as_of is None else time_option(as_of, "--as-of")
    if settings is not None:
        operator_settings(settings)  # A bad file fails here, not in the browser.
    app_path = _app_path()
    open_cli_store(store, create=False).close()
    env = {**os.environ, STORE_VARIABLE: str(store.resolve())}
    # Only the options decide: variables left in the shell are not passed on.
    env.pop(SETTINGS_VARIABLE, None)
    env.pop(AS_OF_VARIABLE, None)
    if settings is not None:
        env[SETTINGS_VARIABLE] = str(settings.resolve())
    if when is not None:
        env[AS_OF_VARIABLE] = isot(when)
    args = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        address,
        "--server.port",
        str(port),
        "--server.maxUploadSize",
        str(MAX_UPLOAD_MB),
        "--browser.gatherUsageStats",
        "false",
        # No browser opened and no first-run e-mail prompt on the terminal.
        "--server.headless",
        "true",
        # An unexpected exception shows only its type in the browser; its
        # message and traceback stay on the terminal.
        "--client.showErrorDetails",
        "type",
    ]
    try:
        completed = subprocess.run(args, env=env, check=False)  # noqa: S603 (this interpreter, fixed arguments)
    except KeyboardInterrupt:
        # Ctrl+C stops Streamlit too: a stop, not an error.
        raise typer.Exit(ExitCode.OK) from None
    if completed.returncode:
        raise typer.Exit(completed.returncode)
