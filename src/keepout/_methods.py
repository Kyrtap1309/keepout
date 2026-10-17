"""The names of the Pc methods, without importing the numerical libraries.

:mod:`keepout.pc` re-exports :data:`Method` and :data:`MethodChoice` and
describes each method in :data:`keepout.pc.METHODS`; the command line builds
its ``--method`` choices from :data:`METHOD_CHOICES`, so ``keepout --help``
need not load NumPy.
"""

from __future__ import annotations

from typing import Final, Literal, TypeAlias, get_args

__all__ = ["METHOD_CHOICES", "Method", "MethodChoice"]

Method: TypeAlias = Literal[
    "foster", "elrod", "hall3d", "montecarlo", "frisbee", "dilution", "alfano"
]
"""Pc methods: 2-D (``foster``, ``elrod``), 3-D (``hall3d``), Monte Carlo
(``montecarlo``) and maximum Pc (``frisbee``, ``dilution``, ``alfano``)."""

MethodChoice: TypeAlias = Literal["auto"] | Method
"""Methods :func:`~keepout.pc.compute_pc` accepts; ``"auto"`` picks one."""

METHOD_CHOICES: Final[tuple[MethodChoice, ...]] = ("auto", *get_args(Method))
"""Every :data:`MethodChoice`, ``"auto"`` first."""
