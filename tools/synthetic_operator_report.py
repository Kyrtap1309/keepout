"""Run keepout's example synthetic operator on the frozen catalog and write the report.

Usage: ``uv run python tools/synthetic_operator_report.py [--check] [--workers N]``.
Screens the example fleet (``examples/synthetic-operator/fleet.toml``)
against the committed public catalog (``tests/screening/data/``, see
``tools/screening_snapshot.py``) with :func:`keepout.synthetic.operator.run`
over the week of the snapshot's SOCRATES report, with the defaults: the 5 km
sphere, the published covariance model, keepout's risk thresholds and
decision point. Writes ``docs/synthetic-operator.json`` (the numbers) and
``docs/synthetic-operator.md`` (the page: the fleet, the model, every
approach with its Pc and decision, and the CDM of the closest approach).
Needs no network. ``--check`` fails instead of writing when a file is out of
date.

Every covariance is the model's assumption, so for each Pc the report also
gives the largest Pc over a scale factor of the combined covariance
(:func:`keepout.pc.dilution`): how far the Pc rests on the assumed size.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, TypedDict

import screening_snapshot as snapshot
from astropy.time import Time
from generated import table, write_or_check

from keepout._units import DAY, KM
from keepout.cdm import kvn
from keepout.encounter import encounter_plane
from keepout.omm import Omm, deduplicate
from keepout.pc import dilution
from keepout.risk import Action, DecisionConfig, Thresholds
from keepout.screening import DEFAULT_VOLUME
from keepout.screening.pipeline import DEFAULT_DURATION
from keepout.synthetic import (
    FleetSatellite,
    OperatorEvent,
    OperatorResult,
    OrbitRegime,
    SyntheticCovarianceModel,
    load_fleet,
    orbit_regime,
)
from keepout.synthetic.operator import UNKNOWN, run

REPO: Final = Path(__file__).resolve().parents[1]
PAGE: Final = REPO / "docs/synthetic-operator.md"
FLEET: Final = REPO / "examples/synthetic-operator/fleet.toml"
TOOL: Final = "tools/synthetic_operator_report.py"

THRESHOLDS: Final = Thresholds()
DECISION: Final = DecisionConfig()

SIGNIFICANT_DIGITS: Final = 10
"""Significant digits of the numbers of the example CDM on the page: enough
to read, and the last digits of a full-precision value could differ with the
machine's floating-point library."""

NOISE_FLOOR: Final = 1e-6
"""Numbers of the example CDM smaller than this in magnitude (in the CDM's
units: m, m/s, km, km/s, m**2, ...) are shown as 0.0. They are rounding
noise, such as the off-diagonal terms (about 1e-9 m**2) of the model's
diagonal RTN covariance after its rotations, whose digits depend on the
machine."""

ACTION_NAMES: Final[Mapping[Action, str]] = {
    Action.IGNORE: "ignore",
    Action.MONITOR: "monitor",
    Action.PREPARE_MANOEUVRE: "monitor, prepare manoeuvre",
    Action.MANOEUVRE: "manoeuvre",
}
"""Each action as the report names it (as ``tools/risk_report.py`` does)."""


class Settings(TypedDict):
    """How the operator ran."""

    fleet: str
    start: str
    days: float
    volume: str
    catalog_hard_body_radius: float
    safety_factor: float
    green_below: float
    red_from: float
    basis: str
    decision_point_days: float


class Catalog(TypedDict):
    """The catalog screened against."""

    retrieved: str
    element_sets: int
    screened: int
    skipped: int


class Regime(TypedDict):
    """The model's parameters of an orbit regime, before the safety factor."""

    regime: str
    radial: float
    transverse: float
    normal: float
    transverse_rate_per_day: float


class Satellite(TypedDict):
    """A fleet satellite."""

    catalog_number: int
    name: str
    regime: str
    hard_body_radius: float
    maneuverable: bool
    epoch: str
    approaches: int


class Event(TypedDict):
    """One approach: its CDM's Pc and the decision."""

    message_id: str
    primary: int
    secondary: int
    secondary_name: str
    secondary_regime: str
    tca: str
    miss_distance: float
    relative_speed: float
    transverse_sigmas: list[float]
    pc: float | None
    method: str | None
    max_pc: float | None
    scale_factor: float | None
    diluted: bool | None
    level: str | None
    action: str
    notes: list[str]


class Summary(TypedDict):
    """The report's numbers, as written to the JSON file."""

    settings: Settings
    catalog: Catalog
    regimes: list[Regime]
    fleet: list[Satellite]
    actions: dict[str, int]
    events: list[Event]
    example: str | None


def shown(path: Path) -> str:
    """``path`` as the page names it: from the repository's root if inside
    it, else as given."""
    resolved = path.resolve()
    if resolved.is_relative_to(REPO):
        return resolved.relative_to(REPO).as_posix()
    return path.as_posix()


def window_start(manifest: snapshot.Manifest) -> Time:
    """0 h UTC of the day of the first TCA of the snapshot's SOCRATES report:
    the week ``tools/screening_report.py`` screens too."""
    return Time(manifest["socrates"]["first_tca"][:10], scale="utc")


def _significant(value: float, digits: int = 3) -> float:
    return float(f"{value:.{digits}g}")


def _level(event: OperatorEvent) -> str | None:
    level = event.assessment.level
    return None if level is None else level.name.lower()


def _satellite(satellite: FleetSatellite, omm: Omm, approaches: int) -> Satellite:
    return {
        "catalog_number": satellite.catalog_number,
        "name": satellite.name or omm.object_name or UNKNOWN,
        "regime": str(orbit_regime(omm)),
        "hard_body_radius": satellite.hard_body_radius,
        "maneuverable": satellite.maneuverable,
        "epoch": omm.epoch.utc.isot,
        "approaches": approaches,
    }


def _event(
    event: OperatorEvent, elements: Mapping[int, Omm], model: SyntheticCovarianceModel
) -> Event:
    approach, observation = event.approach, event.observation
    primary = elements[approach.primary_number]
    secondary = elements[approach.secondary_number]
    sigmas = [
        round(model.sigmas(omm, approach.tca).transverse, 1)
        for omm in (primary, secondary)
    ]
    pc: float | None = None
    method: str | None = None
    max_pc: float | None = None
    scale: float | None = None
    diluted: bool | None = None
    if observation.pc is not None and observation.result is not None:
        conjunction = event.cdm.conjunction()
        bound = dilution(encounter_plane(conjunction), conjunction.hard_body_radius)
        pc = _significant(observation.pc)
        method = str(observation.result.method)
        max_pc = _significant(bound.value)
        scale = _significant(bound.details["scale_factor"])
        diluted = bool(bound.details["diluted"])
    notes = list(observation.diagnostics)
    if observation.error is not None:
        notes.append(observation.error)
    return {
        "message_id": event.cdm.message_id,
        "primary": approach.primary_number,
        "secondary": approach.secondary_number,
        "secondary_name": secondary.object_name or UNKNOWN,
        "secondary_regime": str(orbit_regime(secondary)),
        "tca": approach.tca.utc.iso,
        "miss_distance": round(approach.miss_distance, 3),
        "relative_speed": round(approach.relative_speed, 3),
        "transverse_sigmas": sigmas,
        "pc": pc,
        "method": method,
        "max_pc": max_pc,
        "scale_factor": scale,
        "diluted": diluted,
        "level": _level(event),
        "action": ACTION_NAMES[event.action],
        "notes": notes,
    }


def closest(result: OperatorResult) -> OperatorEvent | None:
    """The event of the closest approach (the earliest of equals)."""
    return min(result.events, key=lambda e: e.approach.miss_distance, default=None)


def summarise(
    result: OperatorResult,
    catalog: Sequence[Omm],
    retrieved: str,
    fleet: str,
) -> Summary:
    """The report's numbers.

    Parameters
    ----------
    result
        The operator's run.
    catalog
        The element sets it ran on.
    retrieved
        The day the catalog was retrieved (UTC, ISO 8601).
    fleet
        The fleet file, as the page names it.
    """
    screening = result.screening
    elements = {omm.catalog_number: omm for omm in deduplicate(catalog).kept}
    per_satellite = Counter(e.approach.primary_number for e in result.events)
    actions = Counter(e.action for e in result.events)
    example = closest(result)
    model = result.model
    return {
        "settings": {
            "fleet": fleet,
            "start": screening.start.utc.isot,
            "days": screening.duration / DAY,
            "volume": DEFAULT_VOLUME.label,
            "catalog_hard_body_radius": result.fleet.catalog_hard_body_radius,
            "safety_factor": model.safety_factor,
            "green_below": THRESHOLDS.green_below,
            "red_from": THRESHOLDS.red_from,
            "basis": THRESHOLDS.basis,
            "decision_point_days": DECISION.decision_point / DAY,
        },
        "catalog": {
            "retrieved": retrieved,
            "element_sets": len(catalog),
            "screened": screening.catalog_count,
            "skipped": len(screening.skipped),
        },
        "regimes": [
            {
                "regime": str(regime),
                "radial": parameters.radial,
                "transverse": parameters.transverse,
                "normal": parameters.normal,
                "transverse_rate_per_day": round(parameters.transverse_rate * DAY, 6),
            }
            for regime in OrbitRegime
            for parameters in (model.parameters(regime),)
        ],
        "fleet": [
            _satellite(
                satellite,
                elements[satellite.catalog_number],
                per_satellite[satellite.catalog_number],
            )
            for satellite in result.fleet.satellites
        ],
        "actions": {name: actions[action] for action, name in ACTION_NAMES.items()},
        "events": [_event(event, elements, model) for event in result.events],
        "example": None if example is None else example.cdm.message_id,
    }


_NUMBER: Final = re.compile(
    r"^(?P<key>[A-Z0-9_]+\s*=\s*)"
    r"(?P<value>[-+]?(?:\d+\.\d*(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+))"
    r"(?P<unit>\s*\[[^\]]*\])?$"
)


def rounded(
    text: str, digits: int = SIGNIFICANT_DIGITS, floor: float = NOISE_FLOOR
) -> str:
    """KVN ``text`` with every real value rounded to ``digits`` significant
    digits, and those smaller than ``floor`` in magnitude to 0.0 (integers,
    times, names and comments unchanged)."""

    def shorten(line: str) -> str:
        match = _NUMBER.match(line)
        if match is None:
            return line
        number = float(match["value"])
        number = 0.0 if abs(number) < floor else float(f"{number:.{digits}g}")
        value = repr(number)
        return f"{match['key']}{value}{match['unit'] or ''}"

    return "\n".join(shorten(line) for line in text.split("\n"))


def _km(metres: float) -> str:
    return f"{metres / KM:.3f}"


def _pc(value: float | None) -> str:
    return "-" if value is None else f"{value:.2e}"


def _object(number: int, name: str) -> str:
    return f"{name} ({number})"


def _introduction(summary: Summary) -> list[str]:
    settings, catalog = summary["settings"], summary["catalog"]
    fleet = summary["fleet"]
    names = ", ".join(_object(s["catalog_number"], s["name"]) for s in fleet)
    return [
        "keepout ran as a conjunction assessment service for a small fleet: "
        f"{len(fleet)} real satellites ({names}), named in the fleet file "
        f"`{settings['fleet']}`. It screened them against the "
        f"{catalog['element_sets']} element sets of the public GP catalog "
        f"({catalog['screened']} screened, {catalog['skipped']} left out by "
        "screening's rules), "
        f"{settings['days']:g} days from "
        f"{settings['start'][:16].replace('T', ' ')} UTC with the "
        f"{settings['volume']}, issued a CCSDS CDM for every close approach, "
        "computed the Pc from each CDM and assessed each event with keepout's "
        f"default thresholds (green below {settings['green_below']:.0e}, red "
        f"from {settings['red_from']:.0e}, {settings['basis']} basis) and "
        f"decision point ({settings['decision_point_days']:g} days before "
        "TCA). See [Screening](screening.md) and [Risk assessment](risk.md). "
        "keepout is not these satellites' operator, and nothing here is advice "
        "about them.",
        "",
        '!!! warning "Assumed covariances"',
        "    GP element sets carry no covariance. Every covariance here is "
        "keepout's synthetic model (`keepout.synthetic.SyntheticCovarianceModel`): "
        "sigmas per orbit regime from the literature, the along-track sigma "
        "growing with the element set's age, times a safety factor of "
        f"{settings['safety_factor']:g}. It is an assumption, not orbit "
        "determination: the Pc values show what the model implies, not the "
        "real risk of these satellites. The CDMs say so in their comments "
        "and give `COVARIANCE_METHOD = DEFAULT`.",
        "",
        "Data: USSPACECOM/18 SDS via Space-Track.org, retrieved from CelesTrak "
        f"(celestrak.org) on {catalog['retrieved']} (see "
        "`tests/screening/data/NOTICE`).",
    ]


def _fleet(summary: Summary) -> list[str]:
    radius = summary["settings"]["catalog_hard_body_radius"]
    return [
        *table(
            (
                "Satellite",
                "Regime",
                "Hard-body radius",
                "Manoeuvrable",
                "Element set epoch (UTC)",
                "Approaches",
            ),
            [
                (
                    _object(s["catalog_number"], s["name"]),
                    s["regime"],
                    f"{s['hard_body_radius']:g} m",
                    "yes" if s["maneuverable"] else "no",
                    s["epoch"][:19].replace("T", " "),
                    s["approaches"],
                )
                for s in summary["fleet"]
            ],
        ),
        "",
        f"Every other object is given a hard-body radius of {radius:g} m. The "
        "radii are illustrative round numbers, not the operators' own values.",
    ]


def _model(summary: Summary) -> list[str]:
    events = summary["events"]
    factor = summary["settings"]["safety_factor"]
    lines = [
        "Each object's position covariance in its RTN frame is diagonal: "
        "sigma_R = f sigma_R0, sigma_T = f (sigma_T0 + k |t - epoch|), "
        f"sigma_N = f sigma_N0, with f = {factor:g} and the velocity block "
        "zero (so `auto` keeps the 2-D Pc). The parameters per regime, before "
        "f (epoch sigmas: Flohrer et al. 2009; k: Levit and Marshall 2011):",
        "",
        *table(
            ("Regime", "sigma_R0", "sigma_T0", "sigma_N0", "k"),
            [
                (
                    r["regime"],
                    f"{_km(r['radial'])} km",
                    f"{_km(r['transverse'])} km",
                    f"{_km(r['normal'])} km",
                    f"{r['transverse_rate_per_day'] / KM:.3f} km/day",
                )
                for r in summary["regimes"]
            ],
        ),
    ]
    sigmas = [sigma for e in events for sigma in e["transverse_sigmas"]]
    if sigmas:
        lines += [
            "",
            "At the approaches' TCAs, sigma_T is "
            f"{_km(min(sigmas))} to {_km(max(sigmas))} km.",
        ]
    return lines


def _decisions(summary: Summary) -> list[str]:
    return [
        *table(
            ("Decision", "Events"),
            [(name, count) for name, count in summary["actions"].items()],
        ),
        "",
        "Each approach is an event with a single CDM, issued at the start of "
        "the window, so the trend is undetermined and the decision rests on the "
        "Pc's level and the time from the CDM to TCA.",
    ]


def _approaches(summary: Summary) -> list[str]:
    events = summary["events"]
    if not events:
        return ["None."]
    names = {s["catalog_number"]: s["name"] for s in summary["fleet"]}
    rows = [
        (
            e["tca"],
            names[e["primary"]],
            _object(e["secondary"], e["secondary_name"]),
            f"{e['miss_distance']:.0f} m",
            f"{e['relative_speed']:.1f} m/s",
            " / ".join(_km(sigma) for sigma in e["transverse_sigmas"]),
            _pc(e["pc"]),
            "-"
            if e["max_pc"] is None
            else f"{_pc(e['max_pc'])} ({e['scale_factor']:.3g})",
            e["level"] or "-",
            e["action"],
        )
        for e in events
    ]
    lines = table(
        (
            "TCA (UTC)",
            "Satellite",
            "Object",
            "Range",
            "Relative speed",
            "sigma_T [km]",
            "Pc",
            "Max Pc (scale)",
            "Level",
            "Decision",
        ),
        rows,
    )
    diluted = [e for e in events if e["diluted"]]
    bounded = [e for e in events if e["diluted"] is False]
    lines += [
        "",
        "sigma_T: the satellite's and the object's along-track sigmas at TCA. "
        "Pc: keepout's Pc from the CDM, as its recipient would compute it "
        "(method `auto`, which keeps the 2-D Pc for position-only covariances). "
        "Max Pc: the largest Pc over a scale factor of the combined covariance "
        "(`keepout.pc.dilution`; Alfano 2005), and that factor: an upper "
        "bound that shows how far each Pc rests on the assumed size of the "
        "covariance.",
        "",
        f"{len(diluted)} of {len(events)} Pc "
        f"{'value is' if len(diluted) == 1 else 'values are'} in the dilution "
        "region (the maximum needs a smaller covariance, so a better-known "
        "orbit could raise the Pc). For "
        f"{'the other one' if len(bounded) == 1 else f'the other {len(bounded)}'} "
        "the maximum needs "
        "a larger covariance: the miss is far from the hard body against the "
        "covariance's narrow directions, so the Pc rises steeply as the "
        "covariance grows. Either way, the decisions hold only as far as the "
        "model does.",
    ]
    noted = [e for e in events if e["notes"]]
    if noted:
        lines += ["", "Notes of the Pc computation:", ""]
        lines += [
            f"- **{e['primary']} - {e['secondary']}** ({e['tca']}): {note}."
            for e in noted
            for note in e["notes"]
        ]
    return lines


def _example(summary: Summary, cdm: str | None) -> list[str]:
    if summary["example"] is None or cdm is None:
        return ["None."]
    event = next(e for e in summary["events"] if e["message_id"] == summary["example"])
    names = {s["catalog_number"]: s["name"] for s in summary["fleet"]}
    return [
        "The CDM of the closest approach "
        f"({_object(event['primary'], names[event['primary']])} and "
        f"{_object(event['secondary'], event['secondary_name'])}, "
        f"{event['miss_distance']:.0f} m) in KVN, its numbers rounded to "
        f"{SIGNIFICANT_DIGITS} significant digits and those below "
        f"{NOISE_FLOOR:g} (rounding noise) shown as 0.0:",
        "",
        "```text",
        *rounded(cdm).rstrip("\n").split("\n"),
        "```",
    ]


def render_page(summary: Summary, cdm: str | None) -> str:
    """The report page, with ``cdm`` the example CDM's KVN (``None`` if the
    run found no approach)."""
    lines = [
        "# Synthetic operator on public data",
        "",
        "<!-- Generated by tools/synthetic_operator_report.py; do not edit. -->",
        "",
        *_introduction(summary),
        "",
        "## Fleet",
        "",
        *_fleet(summary),
        "",
        "## Covariance model",
        "",
        *_model(summary),
        "",
        "## Decisions",
        "",
        *_decisions(summary),
        "",
        "## Approaches",
        "",
        *_approaches(summary),
        "",
        "## Example CDM",
        "",
        *_example(summary, cdm),
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Write (or with ``--check`` verify) the synthetic operator report.

    Returns
    -------
    int
        0 on success; 1 if, with ``--check``, a file is out of date.
    """
    parser = argparse.ArgumentParser(prog=TOOL)
    parser.add_argument("--check", action="store_true", help="fail if out of date")
    parser.add_argument("--data", type=Path, default=snapshot.DATA)
    parser.add_argument("--fleet", type=Path, default=FLEET)
    parser.add_argument("--page", type=Path, default=PAGE)
    parser.add_argument(
        "--workers", type=int, default=4, help="processes to screen with"
    )
    args = parser.parse_args(argv)

    manifest = snapshot.read_manifest(args.data)
    catalog = snapshot.read_catalog(args.data)
    fleet = load_fleet(args.fleet)
    result = run(
        fleet,
        catalog,
        window_start(manifest),
        DEFAULT_DURATION,
        DEFAULT_VOLUME,
        workers=args.workers,
        thresholds=THRESHOLDS,
        decision_config=DECISION,
    )
    summary = summarise(
        result, catalog, snapshot.retrieval_day(manifest), shown(args.fleet)
    )
    example = closest(result)
    cdm = None if example is None else kvn.dumps(example.cdm)
    files = {
        args.page.with_suffix(".json"): json.dumps(summary, indent=2) + "\n",
        args.page: render_page(summary, cdm),
    }
    return write_or_check(files, check=args.check, tool=TOOL)


if __name__ == "__main__":
    sys.exit(main())
