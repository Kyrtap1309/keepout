"""keepout's command line, ``keepout`` (the ``cli`` extra).

The console script runs :func:`main`; the commands are in
:mod:`keepout.cli.app`. This module imports nothing from the extra, so
without it ``keepout`` says how to install it instead of failing with a
traceback. The library never imports this package.
"""

import os
import sys
from typing import Final

from keepout._workers import ONE_THREAD

__all__ = ["EXTRA_MODULES", "INSTALL_HINT", "main"]

INSTALL_HINT: Final = (
    "keepout's command line needs the 'cli' extra: pip install 'keepout[cli]'"
)
"""What ``keepout`` prints (exit status 1) when the extra is missing."""

EXTRA_MODULES: Final = frozenset({"typer", "rich"})
"""The top-level packages of the ``cli`` extra that the commands import."""


def main() -> None:
    """Run the ``keepout`` command on the command line arguments and exit
    with its status.

    Raises
    ------
    SystemExit
        Always: with :data:`INSTALL_HINT` if the extra is missing, else with
        the command's exit status (:class:`keepout.cli.output.ExitCode`).
    ModuleNotFoundError
        If a module outside the ``cli`` extra is missing (re-raised).
    """
    # One BLAS thread unless the user set one: the integrations of a burn
    # grid ran no faster on 20 threads, at 20 times the CPU time.
    for name, value in ONE_THREAD.items():
        os.environ.setdefault(name, value)
    try:
        from keepout.cli.app import run
    except ModuleNotFoundError as exc:
        if (exc.name or "").partition(".")[0] not in EXTRA_MODULES:
            raise
        sys.exit(INSTALL_HINT)
    run()
