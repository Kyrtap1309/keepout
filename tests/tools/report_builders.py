"""Setup shared by the tests of the report tools: a data directory holding the
test fixture, a README with a tool's markers, and the write-then-check run."""

import gzip
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path

import screening_snapshot as snapshot
import snapshot_files
from screening_data import load_fixture

from keepout.omm import Omm
from keepout.omm.celestrak import dumps_csv


def fixture_data(
    folder: Path,
    *,
    catalog: Sequence[Omm] | None = None,
    socrates: str | None = None,
) -> Path:
    """``folder/data``, a data directory as ``tools/screening_snapshot.py``
    writes one: ``catalog`` (the test fixture by default) as the snapshot, the
    SOCRATES text ``socrates`` as the excerpt if given, and the committed
    manifest."""
    data = folder / "data"
    data.mkdir()
    omms = load_fixture() if catalog is None else catalog
    (data / snapshot.SNAPSHOT).write_bytes(gzip.compress(dumps_csv(omms).encode()))
    if socrates is not None:
        (data / snapshot.SOCRATES).write_bytes(gzip.compress(socrates.encode()))
    shutil.copy(snapshot.DATA / snapshot_files.MANIFEST, data / snapshot_files.MANIFEST)
    return data


def readme_with_markers(folder: Path, start: str, end: str, block: str = "") -> Path:
    """``folder/README.md``: an intro line, then ``block`` between ``start``
    and ``end``."""
    readme = folder / "README.md"
    inside = f"{block}\n" if block else ""
    readme.write_text(f"intro\n{start}\n{inside}{end}\n", encoding="utf-8")
    return readme


def assert_writes_then_checks(
    main: Callable[[list[str]], int], args: Sequence[str]
) -> None:
    """``main`` with ``--check`` fails before writing, writes, then passes.

    ``--check`` goes last, so ``args`` may start with a subcommand.
    """
    assert main([*args, "--check"]) == 1
    assert main([*args]) == 0
    assert main([*args, "--check"]) == 0
