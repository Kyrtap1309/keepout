"""The private keepout names the tools may import: an explicit allow-list.

import-linter checks the package's own imports, not ``tools/`` (not a
package), so this test reads every tool's imports itself. A private module
(``keepout._x`` or ``keepout.y._x``) is the package's to change; a tool that
needs one of its names says why here, and anything else goes through the
public API.
"""

import ast
from pathlib import Path
from typing import Final

import pytest
from generated import REPO

TOOLS: Final = sorted((REPO / "tools").glob("*.py"))

ALLOWED: Final = {
    # Plain-text helpers, so the pages word numbers, times and plurals as
    # the CLI does.
    ("keepout._text", "isot"),
    ("keepout._text", "minute"),
    ("keepout._text", "pc_text"),
    ("keepout._text", "plural"),
    ("keepout._text", "utc"),
    # The CLI's default of --workers and the one-thread environment, so the
    # tools screen and time as the CLI does.
    ("keepout._workers", "DEFAULT_WORKERS"),
    ("keepout._workers", "ONE_THREAD"),
    ("keepout._workers", "process_pool"),
    ("keepout._workers", "resolve_workers"),
    # The check a command-line radius gets, worded as the library words it.
    ("keepout._validation", "require_positive"),
    # The JSON value type the snapshot manifests are written as.
    ("keepout._types", "JsonValue"),
    # The source clients' download and atomic write, so the snapshot tools
    # save bodies as the clients' caches do.
    ("keepout.sources._http", "Downloader"),
    ("keepout.sources._http", "http_download"),
    ("keepout.sources._store", "store_bytes"),
    ("keepout.sources._store", "store_json"),
}
"""Each private (module, name) a tool may import, grouped by why."""


def private(module: str) -> bool:
    """Whether ``module`` is a private keepout module."""
    parts = module.split(".")
    return parts[0] == "keepout" and any(part.startswith("_") for part in parts)


def private_imports(path: Path) -> set[tuple[str, str]]:
    """The private keepout (module, name) pairs ``path`` imports; ``*`` for a
    whole module."""
    found: set[tuple[str, str]] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {(alias.name, "*") for alias in node.names if private(alias.name)}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                module = f"{node.module}.{alias.name}"
                if private(node.module):
                    found.add((node.module, alias.name))
                elif private(module):
                    found.add((module, "*"))
    return found


@pytest.mark.parametrize("path", TOOLS, ids=lambda path: path.name)
def test_a_tool_imports_only_allowed_private_names(path: Path) -> None:
    assert private_imports(path) - ALLOWED == set()


def test_every_allowed_private_name_is_used() -> None:
    used = set().union(*(private_imports(path) for path in TOOLS))
    assert ALLOWED - used == set()


def test_a_private_module_is_found_however_it_is_imported(tmp_path: Path) -> None:
    path = tmp_path / "tool.py"
    path.write_text(
        "import keepout._a\n"
        "from keepout import _b\n"
        "from keepout.c._d import e\n"
        "from keepout.f import g\n",
        encoding="utf-8",
    )
    assert private_imports(path) == {
        ("keepout._a", "*"),
        ("keepout._b", "*"),
        ("keepout.c._d", "e"),
    }
