"""Shared by the report tools: the repository's paths, a report's command
line and files, command-line counts, JSON read back, README blocks, Markdown
tables, links, plain-text numbers and times, risk thresholds as settings,
the machine a timing ran on, the generated-page notice, reproducible SVGs
and writing or checking files and figures."""

from __future__ import annotations

import argparse
import datetime
import difflib
import json
import os
import platform
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, TypedDict, TypeVar

import matplotlib as mpl
from typeguard import CollectionCheckStrategy, TypeCheckError, check_type

from keepout._workers import DEFAULT_WORKERS
from keepout.risk import Basis

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from matplotlib.figure import Figure

    from keepout.risk import Thresholds

REPO: Final = Path(__file__).resolve().parents[1]
"""The repository's root directory."""

README: Final = REPO / "README.md"
"""The README, whose summary blocks the report tools write."""

CARA_CDM_ID: Final = "000038771_conj_000030802_20201216_182131_20201215_171306"
"""The CARA test CDM the manoeuvre report plans and the dashboard's seed can
add: Metop-B (38771) against 30802, in
:data:`keepout.sources.cara.PC_METHOD_CDMS`."""

CPUINFO: Final = Path("/proc/cpuinfo")
"""Where Linux names the CPU (:func:`cpu_model`)."""

_T = TypeVar("_T")


def positive(text: str) -> int:
    """A command-line count: an integer of at least 1.

    Raises
    ------
    argparse.ArgumentTypeError
        If the integer in ``text`` is less than 1.
    """
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def report_parser(
    tool: str,
    *,
    page: Path | None,
    images: Path | None = None,
    readme: bool = True,
    workers: bool = True,
    check: str = "fail if out of date",
) -> argparse.ArgumentParser:
    """The command line every report tool shares, for ``tool`` to add its
    own options to.

    Parameters
    ----------
    tool
        The tool, as its usage names it.
    page
        The default of ``--page``; its JSON goes next to it.
    images
        The default of ``--images``, the page's figures; no such option if
        ``None``.
    readme
        Add ``--readme``, the README holding the report's summary block.
    workers
        Add ``--workers``, the processes to screen with.
    check
        The help of ``--check``.
    """
    parser = argparse.ArgumentParser(prog=tool)
    parser.add_argument("--check", action="store_true", help=check)
    parser.add_argument(
        "--page", type=Path, default=page, help="the page; its JSON goes next to it"
    )
    if images is not None:
        parser.add_argument("--images", type=Path, default=images)
    if readme:
        parser.add_argument("--readme", type=Path, default=README)
    if workers:
        parser.add_argument(
            "--workers",
            type=positive,
            default=DEFAULT_WORKERS,
            help="processes to screen with",
        )
    return parser


def json_text(value: object) -> str:
    """``value`` as the tools write JSON files: indented by 2, with a final
    newline."""
    return json.dumps(value, indent=2) + "\n"


def parse_json(text: str, shape: type[_T], source: str) -> _T:
    """JSON ``text`` checked to be a ``shape`` (a TypedDict) throughout: every
    key there and no other, every list item, dict entry and value of the type
    it is declared with.

    Raises
    ------
    ValueError
        If ``text`` is not JSON, or not a ``shape``; the message names
        ``source``.
    """
    return _checked(json.loads(text), shape, source)


def _checked(value: object, shape: type[_T], source: str) -> _T:
    """A JSON ``value`` checked to be a ``shape`` (:func:`parse_json`).

    Raises
    ------
    ValueError
        If ``value`` is not a ``shape``; the message names ``source``.
    """
    try:
        return check_type(
            value, shape, collection_check_strategy=CollectionCheckStrategy.ALL_ITEMS
        )
    except TypeCheckError as exc:
        raise ValueError(f"{source} is not a {shape.__name__}: {exc}") from exc


def read_json(path: Path, shape: type[_T]) -> _T:
    """The JSON file at ``path``, checked to be a ``shape`` (:func:`parse_json`).

    Raises
    ------
    ValueError
        If the file is not JSON, or not a ``shape``.
    """
    return parse_json(path.read_text(encoding="utf-8"), shape, str(path))


def read_timings(path: Path, shape: type[_T]) -> _T | None:
    """The ``timings`` of the JSON file at ``path``, checked to be a
    ``shape`` (:func:`parse_json`), if the file exists and they are not
    null.

    The rest of the file is not read: a report whose other numbers change
    shape keeps the timings it committed.

    Raises
    ------
    ValueError
        If the file is not a JSON object with ``timings``, or they are not a
        ``shape``.
    """
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "timings" not in data:
        raise ValueError(f"{path} has no timings")
    timings: object = data["timings"]
    if timings is None:
        return None
    return _checked(timings, shape, f"the timings of {path}")


def replace_block(text: str, start: str, end: str, block: str) -> str:
    """``text`` with ``block`` between its ``start`` and ``end`` markers.

    Raises
    ------
    ValueError
        If the markers are missing or out of order.
    """
    first = text.find(start)
    last = text.find(end)
    if first < 0 or last < first:
        raise ValueError(f"README needs {start} and {end}, in that order")
    head = text[: first + len(start)]
    return f"{head}\n{block}\n{text[last:]}"


def readme_with(readme: Path, start: str, end: str, block: str) -> str:
    """The text of the file ``readme`` with ``block`` between its ``start``
    and ``end`` markers (:func:`replace_block`).

    Raises
    ------
    ValueError
        If the markers are missing or out of order.
    """
    return replace_block(readme.read_text(encoding="utf-8"), start, end, block)


def report_files(
    page: Path,
    summary: object,
    text: str,
    *,
    readme: Path | None = None,
    markers: tuple[str, str] = ("", ""),
    block: str = "",
) -> dict[Path, str]:
    """What a report writes: ``summary`` as JSON next to ``page``, the page's
    ``text`` and, with ``readme``, that README with ``block`` between its
    ``markers`` (:func:`readme_with`).

    Raises
    ------
    ValueError
        If ``readme`` lacks the markers or has them out of order.
    """
    files = {page.with_suffix(".json"): json_text(summary), page: text}
    if readme is not None:
        start, end = markers
        files[readme] = readme_with(readme, start, end, block)
    return files


def table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    """The lines of a Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines


def relative(path: Path, start: Path) -> str:
    """``path`` as a link from a file in ``start``."""
    return Path(os.path.relpath(path, start)).as_posix()


def significant(value: float, digits: int = 3) -> float:
    """``value`` rounded to ``digits`` significant digits: ``0.0123``."""
    return float(f"{value:.{digits}g}")


def day_text(iso: str) -> str:
    """The day of an ISO 8601 time: ``2025-01-02``."""
    return iso[:10]


def minute_text(iso: str) -> str:
    """An ISO 8601 time to the minute, with a space for the ``T``:
    ``2025-01-02 03:04``."""
    return iso[:16].replace("T", " ")


def second_text(iso: str) -> str:
    """An ISO 8601 time to the second, with a space for the ``T``:
    ``2025-01-02 03:04:05``."""
    return iso[:19].replace("T", " ")


class Machine(TypedDict):
    """Where and when a timing was measured."""

    cpu_model: str
    logical_cpus: int
    python: str
    date: str
    """The day, UTC, ISO 8601."""


def cpu_model(cpuinfo: Path = CPUINFO) -> str:
    """The CPU's model name, as the operating system gives it."""
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "model name":
                return value.strip()
    return platform.processor() or platform.machine() or "unknown CPU"


def machine() -> Machine:
    """The machine running the tool, and today's date."""
    return {
        "cpu_model": cpu_model(),
        "logical_cpus": os.cpu_count() or 1,
        "python": platform.python_version(),
        "date": datetime.datetime.now(datetime.UTC).date().isoformat(),
    }


def machine_text(measured: Machine) -> str:
    """Where and when a timing was measured, as the reports write it:
    ``2026-11-02, Some CPU (8 logical CPUs), Python 3.13.7``."""
    return (
        f"{measured['date']}, {measured['cpu_model']} "
        f"({measured['logical_cpus']} logical CPUs), Python {measured['python']}"
    )


def generated_notice(tool: str) -> str:
    """The comment heading a page that ``tool`` writes."""
    return f"<!-- Generated by {tool}; do not edit. -->"


class ThresholdSettings(TypedDict):
    """Risk thresholds as a report's settings record them."""

    green_below: float
    red_from: float
    basis: Basis


def threshold_settings(thresholds: Thresholds) -> ThresholdSettings:
    """``thresholds`` as a report's settings record them."""
    return {
        "green_below": thresholds.green_below,
        "red_from": thresholds.red_from,
        "basis": thresholds.basis,
    }


def threshold_text(green_below: float, red_from: float) -> str:
    """Risk thresholds as the reports write them:
    ``green below 1e-07, red from 1e-04``."""
    return f"green below {green_below:.0e}, red from {red_from:.0e}"


def save_svg(
    drawing: Figure,
    path: Path,
    *,
    facecolor: str = "auto",
    dpi: float | Literal["figure"] = "figure",
) -> None:
    """``drawing`` saved to ``path`` as a reproducible SVG: no date, fixed
    element ids and text as text. ``facecolor`` and ``dpi`` are
    ``savefig``'s (its defaults: the figure's)."""
    with mpl.rc_context({"svg.hashsalt": "keepout", "svg.fonttype": "none"}):
        drawing.savefig(
            path, format="svg", metadata={"Date": None}, facecolor=facecolor, dpi=dpi
        )


def _diff(path: Path, content: str) -> str:
    """How ``content`` differs from the text of ``path``, as a unified diff."""
    return "".join(
        difflib.unified_diff(
            path.read_text(encoding="utf-8").splitlines(keepends=True),
            content.splitlines(keepends=True),
            f"{path} (committed)",
            f"{path} (generated)",
        )
    )


def write_or_check(files: Mapping[Path, str | bytes], *, check: bool, tool: str) -> int:
    """The files whose contents differ, written, or with ``check`` reported
    (a text file that exists with its diff).

    Text is written as UTF-8, bytes as they are.

    Returns
    -------
    int
        1 if ``check`` found a file out of date, else 0.
    """
    status = 0
    for path, content in files.items():
        data = content.encode() if isinstance(content, str) else content
        if path.exists() and path.read_bytes() == data:
            continue
        if check:
            print(f"{path} is out of date; run {tool}")
            if isinstance(content, str) and path.exists():
                # Shown so a difference only another machine sees can be read
                # in that machine's log.
                print(_diff(path, content), end="")
            status = 1
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            print(f"wrote {path}")
    return status


def write_or_check_figures(
    figures: Mapping[Path, Callable[[], Figure]],
    *,
    check: bool,
    tool: str,
    facecolor: str = "auto",
    dpi: float | Literal["figure"] = "figure",
) -> int:
    """Each figure drawn and saved as a reproducible SVG at its path
    (:func:`save_svg`, with ``facecolor`` and ``dpi``), or with ``check``
    only required to exist: matplotlib versions write different SVG bytes
    for the same figure, so comparing them would fail on another setup.

    Returns
    -------
    int
        1 if ``check`` found a figure missing, else 0.
    """
    status = 0
    for path, draw in figures.items():
        if check:
            if not path.exists():
                print(f"{path} is missing; run {tool}")
                status = 1
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        save_svg(draw(), path, facecolor=facecolor, dpi=dpi)
        print(f"wrote {path}")
    return status
