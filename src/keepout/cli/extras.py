"""The extras the commands need, and the line that says how to install one.

Imports neither Typer nor the numerical libraries: :func:`keepout.cli.main`
uses it before it knows that the ``cli`` extra is there.
"""

from collections.abc import Collection
from typing import Final

__all__ = ["install_hint", "missing_extra"]

_REQUIREMENT: Final = "git+https://github.com/Kyrtap1309/keepout"
"""Where keepout is installed from."""


def install_hint(*extras: str) -> str:
    """The command that installs keepout with ``extras``:
    ``pip install "keepout[cli,store] @ git+https://..."``."""
    return f'pip install "keepout[{",".join(extras)}] @ {_REQUIREMENT}"'


def missing_extra(exc: ModuleNotFoundError, modules: Collection[str]) -> bool:
    """Whether ``exc`` is about one of ``modules`` (top-level packages of an
    extra), or a module of one."""
    return (exc.name or "").partition(".")[0] in modules
