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
import signal
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final, TypeAlias

import typer

from keepout._environment import (
    AS_OF_VARIABLE,
    MAX_UPLOAD_MB,
    SETTINGS_VARIABLE,
    STORE_VARIABLE,
)
from keepout.cli.extras import install_hint
from keepout.cli.options import SettingsFile, operator_settings, time_option
from keepout.cli.output import ExitCode, fail, isot
from keepout.cli.store import StorePath, open_cli_store

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = ["DASHBOARD_HINT", "DEFAULT_HOST", "EXPOSED_WARNING", "dashboard_command"]

DASHBOARD_HINT: Final = (
    f"keepout dashboard needs the dashboard extra: {install_hint('cli', 'dashboard')}"
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
"""What ``keepout dashboard`` prints on standard error, as Streamlit starts,
when ``--host`` is not a loopback address."""

_STOP_TIMEOUT: Final = 10.0
"""How long Streamlit has to stop after Ctrl+C before it is killed [s]."""

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
        envvar=AS_OF_VARIABLE,
        metavar="TIME",
        help=(
            "See the data at this time, ISO 8601 UTC: the events whose latest "
            "TCA is before it are past (default: the clock, read on each page)."
        ),
    ),
]
"""``--as-of``: the time the data are seen at, read by
:func:`keepout._environment.parse_time`, as the dashboard reads it;
``None`` (unset or empty): the clock."""


def _app_path() -> Path:
    """The dashboard's entry script, found without running it; the command ends
    (status 1) without the ``dashboard`` extra or the app."""
    if any(importlib.util.find_spec(name) is None for name in _DASHBOARD_MODULES):
        fail(DASHBOARD_HINT)
    # Loads keepout.dashboard's __init__, which imports nothing, not the app.
    spec = importlib.util.find_spec("keepout.dashboard.app")
    if spec is None or spec.origin is None:
        fail("this keepout has no dashboard app: reinstall keepout")
    return Path(spec.origin)


def _host(value: str) -> str:
    """The ``--host`` address ``value``, checked.

    Raises
    ------
    typer.BadParameter
        If ``value`` is not an IP address, or is a scoped IPv6 one.
    """
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        raise typer.BadParameter(
            f"{value!r} is not an IP address", param_hint="--host"
        ) from None
    if isinstance(address, ipaddress.IPv6Address) and address.scope_id:
        # Streamlit's URL would carry the zone unescaped: not a valid URL.
        raise typer.BadParameter(
            f"{value!r} is a scoped IPv6 address: give it without %{address.scope_id}",
            param_hint="--host",
        )
    return value


def _warn_if_exposed(host: str) -> None:
    """:data:`EXPOSED_WARNING` on standard error unless ``host`` is a loopback
    address."""
    if not ipaddress.ip_address(host).is_loopback:
        typer.echo(EXPOSED_WARNING.format(host=host), err=True)


def _server_address(host: str) -> str:
    """The address the dashboard names in its URL: ``localhost`` when it
    listens on every address (``0.0.0.0``, ``::``), else ``host``, in
    brackets if it is an IPv6 address."""
    address = ipaddress.ip_address(host)
    if address.is_unspecified:
        return "localhost"
    return f"[{host}]" if address.version == 6 else host


def _run(args: Sequence[str], env: Mapping[str, str]) -> int:
    """Run ``args`` until it exits, passing SIGTERM (a container's stop) and
    Ctrl+C on to it; its exit status, 0 if Ctrl+C stopped it."""
    with subprocess.Popen(args, env=env) as child:  # noqa: S603 (this interpreter, fixed arguments)
        previous = signal.signal(signal.SIGTERM, lambda *_: child.terminate())
        try:
            return child.wait()
        except KeyboardInterrupt:
            _stop(child)
            return ExitCode.OK
        finally:
            signal.signal(signal.SIGTERM, previous)


def _stop(child: subprocess.Popen[bytes]) -> None:
    """Stop ``child`` after Ctrl+C: SIGTERM (Ctrl+C may have reached this
    process alone), then SIGKILL after :data:`_STOP_TIMEOUT` or a second
    Ctrl+C."""
    try:
        child.terminate()
        child.wait(timeout=_STOP_TIMEOUT)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        child.kill()
        child.wait()


def _as_of_name(ctx: typer.Context) -> str:
    """Where ``--as-of`` came from, for its errors: the option or its variable."""
    source = ctx.get_parameter_source("as_of")
    # The enum is Typer's vendored Click's: compared by name.
    if source is not None and source.name == "ENVIRONMENT":
        return AS_OF_VARIABLE
    return "--as-of"


def dashboard_command(
    ctx: typer.Context,
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
    when = None if as_of is None else time_option(as_of, _as_of_name(ctx))
    if settings is not None:
        operator_settings(settings)  # A bad file fails here, not in the browser.
    app_path = _app_path()
    open_cli_store(store, create=False).close()
    env = {**os.environ, STORE_VARIABLE: str(store.resolve())}
    # Only the options (--as-of also read from its variable) decide: the
    # shell's values are not passed on as they are.
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
        # Given, Streamlit prints this address instead of looking the
        # machine's public IP up over the network (as it does when it
        # listens on every address).
        "--browser.serverAddress",
        _server_address(address),
        # No browser opened and no first-run e-mail prompt on the terminal.
        "--server.headless",
        "true",
        # An unexpected exception shows only its type in the browser; its
        # message and traceback stay on the terminal.
        "--client.showErrorDetails",
        "type",
        # No Deploy button or developer options in the toolbar: an
        # operator's page, not an app being written.
        "--client.toolbarMode",
        "minimal",
    ]
    _warn_if_exposed(address)
    status = _run(args, env)
    if status:
        raise typer.Exit(status)
