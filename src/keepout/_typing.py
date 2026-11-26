"""Typing helpers without NumPy, for the modules that must not load it
(:mod:`keepout.errors`, the command line's ``--help``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

__all__ = ["override"]

if TYPE_CHECKING:
    # typing.override from Python 3.12, for the type checkers only, so
    # typing_extensions stays out of the dependencies.
    from typing_extensions import override
else:
    _Function = TypeVar("_Function")

    def override(method: _Function) -> _Function:
        """The marker of a method overriding its base class's (checked statically)."""
        return method
