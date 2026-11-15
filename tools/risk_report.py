"""The risk report: every event of ESA's Kelvins data, assessed.

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
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    TypeAlias,
    TypedDict,
    TypeVar,
    get_args,
)

import matplotlib as mpl
from generated import (
    README,
    REPO,
    generated_notice,
    json_text,
    readme_with,
    relative,
    table,
    threshold_settings,
    threshold_text,
    write_or_check,
)
from matplotlib.figure import Figure

from keepout._text import plural
from keepout.figures import FIT_COLOUR, RISK_COLOURS, THRESHOLD_LABELS, trend_figure
from keepout.risk import (
    Action,
    Assessment,
    Basis,
    DecisionConfig,
    Direction,
    PcObservation,
    RiskLevel,
    Thresholds,
    TrendConfig,
    assess,
)
from keepout.sources import kelvins
from keepout.units import DAY, HOUR

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from keepout.figures import TrendFigure
    from keepout.sources.kelvins import KelvinsEvent

PAGE: Final = REPO / "docs/risk-kelvins.md"
IMAGES: Final = REPO / "docs/img/risk-kelvins"
START: Final = "<!-- risk-kelvins-summary:start -->"
END: Final = "<!-- risk-kelvins-summary:end -->"
TOOL: Final = "tools/risk_report.py"

THRESHOLDS: Final = Thresholds()
TREND: Final = TrendConfig()
DECISION: Final = DecisionConfig()

HIGH_RISK: Final = -6.0
"""The challenge's high-risk label: log10 Pc of an event's last CDM >= -6."""

Label: TypeAlias = Literal["high", "low"]
"""The challenge's risk label of an event."""

NO_LEVEL: Final = "none"
"""The level of an event without a Pc, as the report names it."""

ACTIONS: Final = tuple(action.label for action in Action)
"""Every :class:`~keepout.risk.Action`, as the report names it
(:attr:`~keepout.risk.Action.label`)."""
DIRECTIONS: Final = (
    Direction.RISING,
    Direction.STEADY,
    Direction.FALLING,
    Direction.UNDETERMINED,
)
LEVELS: Final = (*(level.key for level in RiskLevel), NO_LEVEL)
"""Every :class:`~keepout.risk.RiskLevel`, as the report names it
(:attr:`~keepout.risk.RiskLevel.key`), then :data:`NO_LEVEL`."""
LABELS: Final[tuple[Label, ...]] = get_args(Label)

PLOT_DAYS: Final = 7.5
"""Days before TCA a plot starts at (Kelvins events span about a week)."""

LOWEST_AXIS: Final = 1e-12
"""The Pc axis reaches at least this low (lower if an event needs it)."""


def level_label(level: RiskLevel | None) -> str:
    """A level as one of :data:`LEVELS` (:data:`NO_LEVEL` for no Pc)."""
    return NO_LEVEL if level is None else level.key


class Settings(TypedDict):
    """The report's settings, as written to the JSON file."""

    green_below: float
    red_from: float
    basis: Basis
    trend_window: int
    steady_within: float
    decision_point_days: float


class Summary(TypedDict):
    """The report's numbers, as written to the JSON file.

    ``high_risk_after_tca`` counts, by decision, the high-risk events whose
    last CDM was issued after TCA (keepout decides without it);
    ``high_risk_levels`` the high-risk events by decision and level of the
    last CDM before TCA.
    """

    doi: str
    settings: Settings
    events: int
    cdms: int
    actions: dict[str, int]
    directions: dict[Direction, int]
    decision_point_vs_final: dict[str, dict[str, int]]
    challenge_label: dict[str, dict[Label, int]]
    high_risk_after_tca: dict[str, int]
    high_risk_levels: dict[str, dict[str, int]]
    examples: dict[str, int]


def assess_with_settings(observations: Iterable[PcObservation]) -> Assessment:
    """The assessment of observations with the report's :data:`THRESHOLDS`,
    :data:`TREND` and :data:`DECISION`, so the page, the numbers and the plots
    agree."""
    return assess(
        observations,
        thresholds=THRESHOLDS,
        trend_config=TREND,
        decision_config=DECISION,
    )


def at_decision_point(observations: Iterable[PcObservation]) -> RiskLevel | None:
    """Risk level of the latest observation issued at or before the decision
    point."""
    known = [o for o in observations if o.time_to_tca >= DECISION.decision_point]
    return assess_with_settings(known).level if known else None


@dataclass(frozen=True, slots=True)
class Example:
    """An example event chosen for a plot."""

    key: str
    title: str
    pick: Callable[[Assessment], bool]
    caption: str
    """One line under the plot; ``{days}`` is the decision point [days] and
    ``{steady}`` the steady band [decades per day]."""


EXAMPLES: Final = (
    Example(
        "maneuver",
        "Red within the decision point: maneuver",
        lambda a: a.decision.action is Action.MANEUVER,
        "The last CDM before TCA is red and at most {days:g} days before "
        "TCA, so keepout decides to manoeuvre.",
    ),
    Example(
        "prepare",
        "Red before the decision point: monitor and prepare a manoeuvre",
        lambda a: a.decision.action is Action.PREPARE_MANEUVER,
        "The last CDM before TCA is red but more than {days:g} days before TCA: "
        "there is still time for more CDMs, so keepout monitors and prepares a "
        "manoeuvre.",
    ),
    Example(
        "rising",
        "Green but rising towards yellow: monitor",
        lambda a: (
            a.level is RiskLevel.GREEN
            and a.trend.direction is Direction.RISING
            and a.decision.action is Action.MONITOR
        ),
        "The last CDM is green, but the Pc rises by more than {steady:g} decades "
        "per day, so keepout monitors the event.",
    ),
    Example(
        "falling",
        "Falling to green: ignore",
        lambda a: (
            a.decision.action is Action.IGNORE
            and a.trend.direction is Direction.FALLING
        ),
        "The Pc falls and the last CDM before TCA is green, so keepout ignores "
        "the event.",
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


def summarise(events: Sequence[KelvinsEvent]) -> Summary:
    """The report's numbers."""
    assessed: dict[int, Assessment] = {}
    known: dict[str, Counter[str]] = {level: Counter() for level in LEVELS}
    labels: dict[str, Counter[Label]] = {action: Counter() for action in ACTIONS}
    after_tca: Counter[str] = Counter()
    high_levels: dict[str, Counter[str]] = {action: Counter() for action in ACTIONS}
    for event in events:
        observations = event.observations()
        assessment = assessed[event.event_id] = assess_with_settings(observations)
        final = level_label(assessment.level)
        known[level_label(at_decision_point(observations))][final] += 1
        high = event.column("risk")[-1] >= HIGH_RISK
        action = assessment.decision.action.label
        labels[action]["high" if high else "low"] += 1
        if high:
            high_levels[action][final] += 1
            after_tca[action] += int(event.time_to_tca[-1] < 0.0)
    actions = Counter(a.decision.action.label for a in assessed.values())
    directions = Counter(a.trend.direction for a in assessed.values())
    return {
        "doi": kelvins.DOI,
        "settings": {
            **threshold_settings(THRESHOLDS),
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
            action: {label: labels[action][label] for label in LABELS}
            for action in ACTIONS
        },
        "high_risk_after_tca": {action: after_tca[action] for action in ACTIONS},
        "high_risk_levels": {
            action: {level: high_levels[action][level] for level in LEVELS}
            for action in ACTIONS
        },
        "examples": choose(EXAMPLES, assessed),
    }


def _share(count: int, total: int) -> str:
    if not total:
        return "-"
    percent = 100 * count / total
    return "<0.1%" if 0 < percent < 0.1 else f"{percent:.1f}%"


_Key = TypeVar("_Key", bound=str)


def _counts_table(title: str, counts: Mapping[_Key, int], total: int) -> list[str]:
    """Events per ``title`` value, with their share of ``total``."""
    return table(
        (title, "Events", "Share"),
        [(key, count, _share(count, total)) for key, count in counts.items()],
    )


def image(event_id: int) -> str:
    """File name of an example plot."""
    return f"event-{event_id}.svg"


def render_summary(summary: Summary, images: str) -> str:
    """The README block: the decisions and the first example plot."""
    total = summary["events"]
    lines = [
        (
            f"keepout's decision for each of the {total} events of ESA's Kelvins "
            "data, from ESA's Pc of every CDM ([report](docs/risk-kelvins.md)):"
        ),
        "",
        *_counts_table("Decision", summary["actions"], total),
    ]
    examples = summary["examples"]
    if examples:
        key, event_id = next(iter(examples.items()))
        lines += ["", f"![Kelvins event {event_id}: {key}]({images}/{image(event_id)})"]
    return "\n".join(lines)


def render_page(summary: Summary, images: str) -> str:
    """The report page, with the example plots under ``images``."""
    total = summary["events"]
    settings = summary["settings"]
    titles = {example.key: example.title for example in EXAMPLES}
    lines = [
        "# Risk assessment on ESA's Kelvins data",
        "",
        generated_notice(TOOL),
        "",
        (
            "This page runs keepout's decision rules (risk level, trend and "
            "decision point) on real sequences of CDMs. It uses ESA's Pc from "
            "each CDM, not keepout's own Pc: the data has no absolute states, and "
            "ESA's method is not published. There is no ground truth. Nobody "
            "knows which of these events would have ended in a collision, so the "
            "page shows what the rules decide, not whether they decide right."
        ),
        "",
        (
            "The data is ESA's Collision Avoidance Challenge set: "
            f"{total} conjunction events of ESA missions from 2015 to 2019, "
            f"{summary['cdms']} CDMs in all. A CDM at ESA's floor of "
            f"1e{kelvins.RISK_FLOOR:.0f} counts as a Pc below the floor."
        ),
        "",
        (
            "Settings: "
            f"{threshold_text(settings['green_below'], settings['red_from'])} "
            f"([{settings['basis']} basis](risk.md#risk-levels)); trend over the "
            f"last {settings['trend_window']} CDMs, steady within "
            f"±{settings['steady_within']:g} decades per day; decision point "
            f"{settings['decision_point_days']:g} days before TCA. See "
            "[Risk assessment](risk.md) for the rules."
        ),
        "",
        (
            "Data: ESA, *Collision Avoidance Challenge dataset*, Zenodo, 2021, "
            f"doi:[{summary['doi']}](https://doi.org/{summary['doi']}); © ESA, "
            "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)."
        ),
        "",
        "## Decisions",
        "",
        *_counts_table("Decision", summary["actions"], total),
        "",
        "## Trend at the last CDM before TCA",
        "",
        *_counts_table("Direction", summary["directions"], total),
        "",
        "## Risk at the decision point and at the last CDM",
        "",
        (
            "Rows: the level of the latest CDM issued at least "
            f"{settings['decision_point_days']:g} days before TCA (*none*: no CDM "
            "that early). Columns: the level of the last CDM before TCA (*none*: "
            "every CDM was issued after TCA)."
        ),
        "",
        *table(
            ("At the decision point", *LEVELS),
            [
                (level, *summary["decision_point_vs_final"][level].values())
                for level in LEVELS
            ],
        ),
        "",
        "## Against the challenge's high-risk label",
        "",
        (
            "The challenge called an event high-risk when its last CDM has "
            f"log10 Pc >= {HIGH_RISK:g}. keepout decides on the CDMs issued "
            "before TCA, so an event whose only high Pc came after TCA is judged "
            "without it."
        ),
        "",
        *table(
            ("Decision", "High-risk", "Not high-risk"),
            [
                (action, counts["high"], counts["low"])
                for action, counts in summary["challenge_label"].items()
            ],
        ),
        *_label_notes(summary),
        "",
        "## Example events",
        "",
        (
            "Each plot shows ESA's Pc of every CDM of one event against the days "
            "to TCA, coloured by level (hollow: at ESA's floor). Dashed lines: "
            "the thresholds; dotted line: the decision point."
        ),
    ]
    captions = {example.key: example.caption for example in EXAMPLES}
    for key, event_id in summary["examples"].items():
        caption = captions[key].format(
            days=settings["decision_point_days"], steady=settings["steady_within"]
        )
        lines += [
            "",
            f"### {titles[key]}",
            "",
            f"![Kelvins event {event_id}]({images}/{image(event_id)})",
            "",
            caption,
        ]
    return "\n".join(lines) + "\n"


def _label_notes(summary: Summary) -> list[str]:
    """What the high-risk table says: the high-risk events keepout ignores
    and those it only monitors."""
    settings = summary["settings"]
    label = summary["challenge_label"]
    notes: list[str] = []
    ignored = label[Action.IGNORE.label]["high"]
    if ignored:
        late = summary["high_risk_after_tca"][Action.IGNORE.label]
        events = plural(ignored, "high-risk event")
        if late == ignored:
            whose = "its" if ignored == 1 else "their"
            notes.append(
                f"keepout ignores {events}: {whose} high-risk label comes from a "
                "CDM issued after TCA, which keepout does not use."
            )
        else:
            notes.append(
                f"keepout ignores {events}; for {late} of them the high-risk "
                "label comes from a CDM issued after TCA, which keepout does not "
                "use."
            )
    monitored = label[Action.MONITOR.label]["high"]
    if monitored:
        yellow = summary["high_risk_levels"][Action.MONITOR.label][RiskLevel.YELLOW.key]
        share = "All" if yellow == monitored else f"{yellow} of the"
        notes.append(
            f"{share} {monitored} high-risk events keepout monitors are yellow at "
            "the last CDM before TCA. The challenge's line, a Pc of "
            f"1e{HIGH_RISK:.0f}, lies inside keepout's yellow band "
            f"({threshold_text(settings['green_below'], settings['red_from'])}): "
            "such an event is watched for more CDMs, and a manoeuvre is "
            "considered only when its Pc reaches red."
        )
    return [line for note in notes for line in ("", note)]


def plot_title(event: KelvinsEvent, assessment: Assessment) -> str:
    """The title of an event's plot: its action and trend."""
    return (
        f"Kelvins event {event.event_id} (mission {event.mission_id}): "
        f"{assessment.decision.action.label}, trend {assessment.trend.direction}"
    )


def trend_plot(figure: TrendFigure, title: str) -> Figure:
    """An event's Pc against the time to TCA; a point below the source's
    floor is drawn hollow."""
    days = figure.hours_to_tca * HOUR / DAY
    with mpl.rc_context({"svg.hashsalt": "keepout", "svg.fonttype": "none"}):
        drawing = Figure(figsize=(8.5, 3.6))
        axes = drawing.subplots()
        if figure.fit is not None:
            hours, values = figure.fit
            axes.plot(
                hours * HOUR / DAY, values, color=FIT_COLOUR, linewidth=1.0, zorder=1
            )
        for day, value, level, floored in zip(
            days, figure.pc, figure.levels, figure.below_floor, strict=True
        ):
            colour = RISK_COLOURS[level]
            axes.scatter(
                [day],
                [value],
                s=36,
                zorder=2,
                facecolors="none" if floored else colour,
                edgecolors=colour,
            )
        for value, level in reversed(figure.thresholds):
            axes.axhline(
                value,
                color=RISK_COLOURS[level],
                linestyle="--",
                label=f"{THRESHOLD_LABELS[level]} {value:.0e}",
            )
        axes.axvline(
            DECISION.decision_point / DAY,
            color="#424242",
            linestyle=":",
            label="decision point",
        )
        axes.set_yscale("log")
        axes.set_ylim(min([LOWEST_AXIS, *figure.pc]) / 10, 1.0)
        axes.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize="small")
        axes.set_xlim(PLOT_DAYS, 0.0)
        axes.set_xlabel("days to TCA")
        axes.set_ylabel("Pc (ESA)")
        axes.set_title(title)
        axes.grid(visible=True, which="major", alpha=0.3)
        drawing.tight_layout()
    return drawing


def plot(figure: TrendFigure, title: str, path: Path) -> None:
    """An event's plot (:func:`trend_plot`), saved as a reproducible SVG."""
    drawing = trend_plot(figure, title)
    with mpl.rc_context({"svg.hashsalt": "keepout", "svg.fonttype": "none"}):
        drawing.savefig(path, format="svg", metadata={"Date": None})


def main(argv: Sequence[str] | None = None) -> int:
    """The entry point writing (or with ``--check`` verifying) the Kelvins risk report.

    Returns
    -------
    int
        0 on success; 1 if, with ``--check``, a file is out of date or a plot
        is missing.
    """
    parser = argparse.ArgumentParser(prog=TOOL)
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "fail if out of date (the plots only need to exist: their SVG bytes "
            "vary with the matplotlib version)"
        ),
    )
    parser.add_argument(
        "--table", type=Path, default=None, help="a local raw table or excerpt"
    )
    parser.add_argument("--page", type=Path, default=PAGE)
    parser.add_argument("--images", type=Path, default=IMAGES)
    parser.add_argument("--readme", type=Path, default=README)
    args = parser.parse_args(argv)

    events = kelvins.load(args.table)
    summary = summarise(events)
    block = render_summary(summary, relative(args.images, args.readme.parent))
    files = {
        args.page.with_suffix(".json"): json_text(summary),
        args.page: render_page(summary, relative(args.images, args.page.parent)),
        args.readme: readme_with(args.readme, START, END, block),
    }
    status = write_or_check(files, check=args.check, tool=TOOL)
    by_id = {event.event_id: event for event in events}
    for event_id in summary["examples"].values():
        path = args.images / image(event_id)
        if args.check:
            # Only its existence: matplotlib versions write different SVG
            # bytes for the same figure.
            if not path.exists():
                print(f"{path} is missing; run {TOOL}")
                status = 1
            continue
        args.images.mkdir(parents=True, exist_ok=True)
        event = by_id[event_id]
        assessment = assess_with_settings(event.observations())
        plot(
            trend_figure(assessment, THRESHOLDS),
            plot_title(event, assessment),
            path,
        )
        print(f"wrote {path}")
    return status


if __name__ == "__main__":
    sys.exit(main())
