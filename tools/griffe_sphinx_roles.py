"""A griffe extension for the documentation: Sphinx roles become links.

keepout's docstrings cross-reference with Sphinx roles (``:func:`screen```,
``:class:`~keepout.pc.PcResult```), which read well in ``help()`` and in
editors. mkdocstrings does not understand them and would print them as they
are. This extension rewrites each role, once the package is loaded, into an
optional mkdocs-autorefs link: a link where the target is on a page, plain
code where it is not (a built-in, a private helper).

A target is written as in Sphinx:

- ``~a.b.c`` shows ``c`` and links to ``a.b.c``;
- ``title <a.b.c>`` shows ``title``;
- a name without dots, or whose first part is not ``keepout``, is resolved
  in the scope of the documented object, as Python would resolve it there.

mkdocs.yml loads it with ``extensions: [tools/griffe_sphinx_roles.py]``.
"""

from __future__ import annotations

import functools
import html
import re
import sys
from typing import TYPE_CHECKING, Final

import griffe

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Iterator

ROLE: Final = re.compile(
    r":(?:py:)?(?:func|class|meth|attr|mod|data|exc|obj|const|type):"
    r"`(?P<tilde>~?)(?P<text>[^`<]+?)(?:\s*<(?P<target>[^`>]+)>)?`"
)
"""A Sphinx cross-reference role, e.g. ``:func:`~keepout.pc.compute_pc```."""

PACKAGE: Final = "keepout"
"""The package whose names are taken as already absolute."""


def _resolve(obj: griffe.Object, name: str) -> str | None:
    """The full path of ``name`` seen from ``obj``, or ``None`` when Python
    would not find it there (a built-in, a typo)."""
    if name == PACKAGE or name.startswith(f"{PACKAGE}."):
        return name
    head, _, rest = name.partition(".")
    try:
        resolved = obj.resolve(head)
    except griffe.NameResolutionError:
        return None
    return f"{resolved}.{rest}" if rest else resolved


def _link(obj: griffe.Object, match: re.Match[str]) -> str:
    """The autorefs link that replaces one role."""
    text = match["text"].strip()
    target = (match["target"] or text).strip().lstrip("~")
    if match["tilde"]:
        text = text.rsplit(".", 1)[-1]
    code = f"<code>{html.escape(text)}</code>"
    identifier = _resolve(obj, target)
    if identifier is None:
        return code
    return f'<autoref identifier="{html.escape(identifier)}" optional>{code}</autoref>'


def _objects(obj: griffe.Object, seen: set[str]) -> Iterator[griffe.Object]:
    """``obj`` and every object defined under it, aliases left out."""
    if obj.path in seen:
        return
    seen.add(obj.path)
    yield obj
    for member in obj.members.values():
        if isinstance(member, griffe.Object):
            yield from _objects(member, seen)


class SphinxRoles(griffe.Extension):
    """Rewrites the Sphinx roles of every docstring of the loaded package."""

    @override
    def on_package(self, *, pkg: griffe.Module, **kwargs: object) -> None:
        """Rewrite the docstrings once the whole package is loaded, so that
        every name can be resolved."""
        for obj in _objects(pkg, set()):
            docstring = obj.docstring
            if docstring is not None and ":" in docstring.value:
                docstring.value = ROLE.sub(
                    functools.partial(_link, obj), docstring.value
                )
