"""Assess every event of ESA's Kelvins data and write the risk report.

Usage: ``uv run python tools/risk_report.py [--check] [--table PATH]``.
Downloads the Kelvins raw table (221 MB, once), assesses each of its 15 321
events with keepout's default thresholds, trend and decision point, and
writes ``docs/risk-kelvins.json`` (the numbers), ``docs/risk-kelvins.md``
(the page), plots of a few example events under ``docs/img/risk-kelvins/``
and the README summary block. ``--check`` fails instead of writing when the
numbers, the page or the README are out of date; the plots are then only
required to exist (SVG bytes vary with the matplotlib version).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import matplotlib as mpl
from generated import replace_block, write_or_check
from matplotlib.figure import Figure

from keepout.risk import (
    DAY,
    PC_FLOOR,
    Action,
    Assessment,
    DecisionConfig,
    PcObservation,
    RiskLevel,
    Thresholds,
    TrendConfig,
    assess,
)
from keepout.sources import kelvins
from keepout.sources.kelvins import KelvinsEvent

REPO: Final = Path(__file__).resolve().parents[1]
PAGE: Final = REPO / "docs/risk-kelvins.md"
IMAGES: Final = REPO / "docs/img/risk-kelvins"
START: Final = "<!-- risk-kelvins-summary:start -->"
END: Final = "<!-- risk-kelvins-summary:end -->"

THRESHOLDS: Final = Thresholds()
TREND: Final = TrendConfig()
DECISION: Final = DecisionConfig()

HIGH_RISK: Final = -6.0
"""The challenge's high-risk label: log10 Pc of an event's last CDM >= -6."""

ACTIONS: Final = ("ignore", "monitor", "monitor, prepare manoeuvre", "manoeuvre")
DIRECTIONS: Final = ("rising", "steady", "falling", "undetermined")
LEVELS: Final = ("green", "yellow", "red", "none")

LOWEST_AXIS: Final = 1e-12
"""The Pc axis reaches at least this low (lower if an event needs it)."""

COLOURS: Final[Mapping[RiskLevel, str]] = {
    RiskLevel.GREEN: "#2e7d32",
    RiskLevel.YELLOW: "#e6a700",
    RiskLevel.RED: "#c62828",
}


def action_name(assessment: Assessment) -> str:
    """The decision as one of :data:`ACTIONS`."""
    decision = assessment.decision
    if decision.prepare_manoeuvre:
        return "monitor, prepare manoeuvre"
    return decision.action.name.lower()


def level_name(level: RiskLevel | None) -> str:
    """A level as one of :data:`LEVELS` (``"none"`` for no Pc)."""
    return "none" if level is None else level.name.lower()


def assess_with_settings(observations: Iterable[PcObservation]) -> Assessment:
    """Assess observations with the report's :data:`THRESHOLDS`, :data:`TREND`
    and :data:`DECISION`, so the page, the numbers and the plots agree."""
    return assess(
        observations,
        thresholds=THRESHOLDS,
        trend_config=TREND,
        decision_config=DECISION,
    )


def at_decision_point(event: KelvinsEvent) -> RiskLevel | None:
    """Risk level of the latest CDM issued at or before the decision point."""
    known = [
        o for o in event.observations() if o.time_to_tca >= DECISION.decision_point
    ]
    return assess_with_settings(known).level if known else None


@dataclass(frozen=True)
class Example:
    """An example event chosen for a plot."""

    key: str
    title: str
    pick: Callable[[Assessment], bool]


EXAMPLES: Final = (
    Example(
        "manoeuvre",
        "Red within the decision point: manoeuvre",
        lambda a: a.decision.action is Action.MANOEUVRE,
    ),
    Example(
        "prepare",
        "Red before the decision point: monitor and prepare a manoeuvre",
        lambda a: a.decision.prepare_manoeuvre,
    ),
    Example(
        "rising",
        "Green but rising towards yellow: monitor",
        lambda a: (
            a.level is RiskLevel.GREEN
            and a.trend.direction == "rising"
            and a.decision.action is Action.MONITOR
        ),
    ),
    Example(
        "falling",
        "Falling to green: ignore",
        lambda a: a.decision.action is Action.IGNORE and a.trend.direction == "falling",
    ),
)
"""For each, the event it picks with the most CDMs before TCA (the lowest
``event_id`` among equals)."""


def choose(
    examples: Sequence[Example], assessed: Mapping[int, Assessment]
) -> dict[str, int]:
    """The event of each example that has one."""
    chosen: dict[str, int] = {}
    for example in examples:
        matching = [i for i, a in assessed.items() if example.pick(a)]
        if matching:
            chosen[example.key] = min(
                matching, key=lambda i: (-len(assessed[i].trend.points), i)
            )
    return chosen


def summarise(events: Sequence[KelvinsEvent]) -> dict[str, Any]:
    """The report's numbers."""
    assessed = {e.event_id: assess_with_settings(e.observations()) for e in events}
    actions = Counter(action_name(a) for a in assessed.values())
    directions = Counter(a.trend.direction for a in assessed.values())
    known: dict[str, Counter[str]] = {level: Counter() for level in LEVELS}
    labels: dict[str, Counter[str]] = {action: Counter() for action in ACTIONS}
    for event in events:
        assessment = assessed[event.event_id]
        final = level_name(assessment.level)
        known[level_name(at_decision_point(event))][final] += 1
        high = event.column("risk")[-1] >= HIGH_RISK
        labels[action_name(assessment)]["high" if high else "low"] += 1
    return {
        "doi": kelvins.DOI,
        "settings": {
            "green_below": THRESHOLDS.green_below,
            "red_from": THRESHOLDS.red_from,
            "basis": THRESHOLDS.basis,
            "trend_window": TREND.window,
            "steady_within": TREND.steady_within,
            "decision_point_days": DECISION.decision_point / DAY,
        },
        "events": len(events),
        "cdms": sum(len(event) for event in events),
        "actions": {action: actions[action] for action in ACTIONS},
        "directions": {d: directions[d] for d in DIRECTIONS},
        "decision_point_vs_final": {
            level: {final: known[level][final] for final in LEVELS} for level in LEVELS
        },
        "challenge_label": {
            action: {label: labels[action][label] for label in ("high", "low")}
            for action in ACTIONS
        },
        "examples": choose(EXAMPLES, assessed),
    }


def _share(count: int, total: int) -> str:
    if not total:
        return "-"
    percent = 100 * count / total
    return "<0.1%" if 0 < percent < 0.1 else f"{percent:.1f}%"


def _table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines


def image(event_id: int) -> str:
    """File name of an example plot."""
    return f"event-{event_id}.svg"


def render_summary(summary: Mapping[str, Any], images: str) -> str:
    """The README block: the decisions and the first example plot."""
    total = summary["events"]
    lines = [
        f"keepout's decision for each of the {total} events of ESA's Kelvins "
        "data, from ESA's Pc of every CDM ([report](docs/risk-kelvins.md)):",
        "",
        *_table(
            ("Decision", "Events", "Share"),
            [
                (action, count, _share(count, total))
                for action, count in summary["actions"].items()
            ],
        ),
    ]
    examples = summary["examples"]
    if examples:
        key, event_id = next(iter(examples.items()))
        lines += ["", f"![Kelvins event {event_id}: {key}]({images}/{image(event_id)})"]
    return "\n".join(lines)


def render_page(summary: Mapping[str, Any], images: str) -> str:
    """The report page, with the example plots under ``images``."""
    total = summary["events"]
    settings = summary["settings"]
    titles = {example.key: example.title for example in EXAMPLES}
    lines = [
        "# Risk assessment on ESA's Kelvins data",
        "",
        "<!-- Generated by tools/risk_report.py; do not edit. -->",
        "",
        f"keepout assessed every event of ESA's Collision Avoidance Challenge "
        f"data set: {total} conjunction events of ESA missions from 2015 to 2019, "
        f"{summary['cdms']} CDMs in all. Each CDM gives ESA's own Pc, which "
        "keepout does not recompute (the data lacks absolute states and ESA's "
        "method is not published); CDMs at ESA's floor of 1e-30 count as "
        "below it.",
        "",
        f"Settings: green below {settings['green_below']:.0e}, red from "
        f"{settings['red_from']:.0e} ({settings['basis']} basis); trend over the "
        f"last {settings['trend_window']} CDMs, steady within "
        f"±{settings['steady_within']:g} decades per day; decision point "
        f"{settings['decision_point_days']:g} days before TCA. See "
        "[Risk assessment](risk.md) for the rules.",
        "",
        "Data: ESA, *Collision Avoidance Challenge dataset*, Zenodo, 2021, "
        f"doi:[{summary['doi']}](https://doi.org/{summary['doi']}); © ESA, "
        "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).",
        "",
        "## Decisions",
        "",
        *_table(
            ("Decision", "Events", "Share"),
            [
                (action, count, _share(count, total))
                for action, count in summary["actions"].items()
            ],
        ),
        "",
        "## Trend at the last CDM before TCA",
        "",
        *_table(
            ("Direction", "Events", "Share"),
            [
                (direction, count, _share(count, total))
                for direction, count in summary["directions"].items()
            ],
        ),
        "",
        "## Risk at the decision point and at the last CDM",
        "",
        "Rows: the level of the latest CDM issued at least "
        f"{settings['decision_point_days']:g} days before TCA (*none*: no CDM "
        "that early). Columns: the level of the last CDM before TCA (*none*: "
        "every CDM was issued after TCA).",
        "",
        *_table(
            ("At the decision point", *LEVELS),
            [
                (level, *summary["decision_point_vs_final"][level].values())
                for level in LEVELS
            ],
        ),
        "",
        "## Against the challenge's high-risk label",
        "",
        "The challenge called an event high-risk when its last CDM has "
        f"log10 Pc >= {HIGH_RISK:g}. keepout decides on the CDMs issued "
        "before TCA, so an event whose only high Pc came after TCA is judged "
        "without it.",
        "",
        *_table(
            ("Decision", "High-risk", "Not high-risk"),
            [
                (action, counts["high"], counts["low"])
                for action, counts in summary["challenge_label"].items()
            ],
        ),
        "",
        "## Example events",
    ]
    for key, event_id in summary["examples"].items():
        lines += [
            "",
            f"### {titles[key]}",
            "",
            f"![Kelvins event {event_id}]({images}/{image(event_id)})",
        ]
    return "\n".join(lines) + "\n"


def plot(event: KelvinsEvent, assessment: Assessment, path: Path) -> None:
    """Draw an event's Pc against the time to TCA as a reproducible SVG."""
    with mpl.rc_context({"svg.hashsalt": "keepout", "svg.fonttype": "none"}):
        figure = Figure(figsize=(8.5, 3.6))
        axes = figure.subplots()
        days = [point.time_to_tca / DAY for point in assessment.trend.points]
        values = [max(point.value, PC_FLOOR) for point in assessment.trend.points]
        axes.plot(days, values, color="#757575", linewidth=1.0, zorder=1)
        for point, day, value in zip(
            assessment.trend.points, days, values, strict=True
        ):
            floored = point.observation.kind == "below_floor"
            colour = COLOURS[point.level]
            axes.scatter(
                [day],
                [value],
                s=36,
                zorder=2,
                facecolors="none" if floored else colour,
                edgecolors=colour,
            )
        axes.axhline(
            THRESHOLDS.red_from,
            color=COLOURS[RiskLevel.RED],
            linestyle="--",
            label=f"red from {THRESHOLDS.red_from:.0e}",
        )
        axes.axhline(
            THRESHOLDS.green_below,
            color=COLOURS[RiskLevel.YELLOW],
            linestyle="--",
            label=f"green below {THRESHOLDS.green_below:.0e}",
        )
        axes.axvline(
            DECISION.decision_point / DAY,
            color="#424242",
            linestyle=":",
            label="decision point",
        )
        axes.set_yscale("log")
        axes.set_ylim(min([LOWEST_AXIS, *values]) / 10, 1.0)
        axes.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize="small")
        axes.set_xlim(7.5, 0.0)
        axes.set_xlabel("days to TCA")
        axes.set_ylabel("Pc (ESA)")
        axes.set_title(
            f"Kelvins event {event.event_id} (mission {event.mission_id}): "
            f"{action_name(assessment)}, trend {assessment.trend.direction}"
        )
        axes.grid(True, which="major", alpha=0.3)
        figure.tight_layout()
        figure.savefig(path, format="svg", metadata={"Date": None})


def _relative(path: Path, start: Path) -> str:
    """``path`` as a link from a file in ``start``."""
    return Path(os.path.relpath(path, start)).as_posix()


def main(argv: Sequence[str] | None = None) -> int:
    """Write (or with ``--check`` verify) the Kelvins risk report.

    Returns
    -------
    int
        0 on success; 1 if, with ``--check``, a file is out of date or a plot
        is missing.
    """
    parser = argparse.ArgumentParser(prog="tools/risk_report.py")
    parser.add_argument("--check", action="store_true", help="fail if out of date")
    parser.add_argument(
        "--table", type=Path, default=None, help="a local raw table or excerpt"
    )
    parser.add_argument("--page", type=Path, default=PAGE)
    parser.add_argument("--images", type=Path, default=IMAGES)
    parser.add_argument("--readme", type=Path, default=REPO / "README.md")
    args = parser.parse_args(argv)

    events = kelvins.load(args.table)
    summary = summarise(events)
    files = {
        args.page.with_suffix(".json"): json.dumps(summary, indent=2) + "\n",
        args.page: render_page(summary, _relative(args.images, args.page.parent)),
        args.readme: replace_block(
            args.readme.read_text(encoding="utf-8"),
            START,
            END,
            render_summary(summary, _relative(args.images, args.readme.parent)),
        ),
    }
    status = write_or_check(files, check=args.check, tool="tools/risk_report.py")
    by_id = {event.event_id: event for event in events}
    for event_id in summary["examples"].values():
        path = args.images / image(event_id)
        if args.check:
            if not path.exists():
                print(f"{path} is missing; run tools/risk_report.py")
                status = 1
            continue
        args.images.mkdir(parents=True, exist_ok=True)
        event = by_id[event_id]
        plot(event, assess_with_settings(event.observations()), path)
        print(f"wrote {path}")
    return status


if __name__ == "__main__":
    sys.exit(main())
