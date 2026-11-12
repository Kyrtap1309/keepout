"""The JSON of a manoeuvre outcome, and the key it is cached under."""

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Final

import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time
from screening_data import FIXTURE_START, load_fixture

import keepout
from keepout._types import JsonValue
from keepout.cdm import CovarianceMethod, Maneuverable, ObjectMetadata, make_cdm
from keepout.conjunction import Conjunction
from keepout.covariance import Covariance
from keepout.errors import StoreError
from keepout.events import ConjunctionEvent
from keepout.maneuver import (
    BurnGrid,
    ManeuverOutcome,
    NewApproach,
    PlannedBurn,
    RescreenResult,
    plan_event,
)
from keepout.objects import ObjectState
from keepout.orbits import EARTH_MU
from keepout.pc import computation
from keepout.propagation.forces import EARTH_RADIUS
from keepout.risk import RiskLevel, Thresholds
from keepout.screening import CloseApproach
from keepout.settings import Settings
from keepout.sources.catalog import CatalogFileRef, CatalogReference
from keepout.state import StateVector
from keepout.store.codec import result_key
from keepout.store.maneuver_codec import (
    MANEUVER_KIND,
    maneuver_key,
    outcome_from_json,
    outcome_to_json,
)
from keepout.synthetic import Fleet, FleetSatellite
from keepout.synthetic.operator import run
from keepout.units import DAY, HOUR, KM

TCA: Final = Time("2026-10-06T00:00:00", scale="utc")
RADIUS: Final = EARTH_RADIUS + 630 * KM
SMALL: Final = BurnGrid(steps_per_orbit=1, delta_v_per_direction=6)
"""About 70 cells at 25 h, which take a fraction of a second."""


def _item(object_id: str, r: list[float], angle: float) -> ObjectState:
    speed = math.sqrt(EARTH_MU / RADIUS)
    v = speed * np.array([0.0, math.cos(angle), math.sin(angle)])
    covariance = np.diag([100.0**2] * 3 + [0.01**2] * 3)
    return ObjectState(
        object_id,
        StateVector(TCA, r, v, "GCRF"),
        Covariance(covariance, "GCRF", TCA),
        10.0,
    )


def leo_event() -> ConjunctionEvent:
    """A fast crossing in LEO, 150 m apart radially, 25 h before TCA: Pc
    about 5e-3, as in the outcome's tests."""
    conjunction = Conjunction(
        _item("1001", [RADIUS, 0.0, 0.0], 0.9),
        _item("2002", [RADIUS + 150.0, 0.0, 0.0], -0.9),
        20.0,
    )
    method = CovarianceMethod.CALCULATED
    cdm = make_cdm(
        conjunction,
        ObjectMetadata(
            "PRI", "2020-001A", maneuverable=Maneuverable.YES, covariance_method=method
        ),
        ObjectMetadata("SEC", "2019-001A", covariance_method=method),
        created=TCA - 25 * HOUR * u.s,
    )
    return ConjunctionEvent.from_cdms([cdm], event_id="EVENT-7")


def approach(secondary: int, at: Time, pc: float, level: RiskLevel) -> NewApproach:
    """A new approach of 1001 at ``at``, its epochs in TDB to check the scale
    survives."""
    epoch = at.tdb
    return NewApproach(
        approach=CloseApproach(
            1001,
            secondary,
            StateVector(epoch, [RADIUS, 1.0 / 3.0, 0.0], [0.0, 7500.1, 0.2], "GCRF"),
            StateVector(epoch, [RADIUS + 512.25, 0.0, 0.1], [0.0, -10.0, 7500], "GCRF"),
        ),
        pc=pc,
        level=level,
    )


def with_rescreens(outcome: ManeuverOutcome) -> ManeuverOutcome:
    """``outcome`` with a re-screen of each burn: new, closer and returning
    approaches in the first, nothing found in the others."""
    start = TCA - DAY * u.s
    end = TCA + 6 * DAY * u.s
    burns = []
    for index, burn in enumerate(outcome.burns):
        found = index == 0
        burns.append(
            PlannedBurn(
                burn.option,
                RescreenResult(
                    option=burn.option,
                    new_approaches=(
                        approach(3003, TCA + 2 * DAY * u.s, 2e-4, RiskLevel.RED),
                        approach(4004, TCA + 3 * DAY * u.s, 3e-6, RiskLevel.YELLOW),
                    )
                    if found
                    else (),
                    closer_approaches=(
                        (approach(5005, TCA + 4 * DAY * u.s, 1e-9, RiskLevel.GREEN),)
                        if found
                        else ()
                    ),
                    event_returns=(
                        (approach(2002, TCA + 5 * DAY * u.s, 1e-3, RiskLevel.RED),)
                        if found
                        else ()
                    ),
                    event_range=1234.5 if found else None,
                    window=(start, end),
                    catalog_objects=217,
                    secondary_screened=found,
                    skipped_objects=3,
                ),
            )
        )
    return dataclasses.replace(outcome, burns=tuple(burns), catalog_label="gp.json")


@pytest.fixture(scope="module", name="outcome")
def an_outcome() -> ManeuverOutcome:
    planned = plan_event(leo_event(), Settings(), grid=SMALL)
    assert planned.burns
    return with_rescreens(planned)


def through_text(payload: dict[str, JsonValue]) -> Any:
    """``payload`` through strict JSON text, as a store keeps it."""
    return json.loads(json.dumps(payload, allow_nan=False))


def test_outcome_round_trip(outcome: ManeuverOutcome) -> None:
    payload = outcome_to_json(outcome)
    back = outcome_from_json(through_text(payload), outcome.event)
    assert outcome_to_json(back) == payload
    assert back.event is outcome.event
    assert back.target_pc == outcome.target_pc
    assert back.baseline == outcome.baseline
    assert back.notes == outcome.notes
    assert back.catalog_label == "gp.json"
    assert [burn.option for burn in back.burns] == [b.option for b in outcome.burns]
    assert back.verdict == outcome.verdict
    first = back.burns[0].rescreen
    original = outcome.burns[0].rescreen
    assert first is not None
    assert original is not None
    assert first.option == back.burns[0].option
    assert [item.level for item in first.new_approaches] == [
        RiskLevel.RED,
        RiskLevel.YELLOW,
    ]
    new = first.new_approaches[0].approach
    old = original.new_approaches[0].approach
    assert new.primary_state.epoch.scale == "tdb"
    assert new.primary_state.epoch.jd1 == old.primary_state.epoch.jd1
    assert new.primary_state.epoch.jd2 == old.primary_state.epoch.jd2
    assert new.primary_state.frame == "GCRF"
    assert np.array_equal(new.primary_state.r, old.primary_state.r)
    assert np.array_equal(new.secondary_state.v, old.secondary_state.v)
    assert new.range_at_tca == old.range_at_tca
    assert first.window[0].scale == "utc"
    assert first.window[0].jd1 == original.window[0].jd1
    assert first.window[0].jd2 == original.window[0].jd2
    assert first.event_range == 1234.5
    assert back.burns[1].rescreen is not None
    assert back.burns[1].rescreen.event_range is None
    assert back.grid is not None
    assert outcome.grid is not None
    assert np.array_equal(back.grid.pc, outcome.grid.pc, equal_nan=True)


@pytest.mark.slow
def test_outcome_of_a_real_rescreen_round_trips() -> None:
    """A plan of a synthetic event on the committed GP data, both burns
    re-screened against it (8 s)."""
    fixture = load_fixture()
    fleet = Fleet(
        (FleetSatellite(52939, 5.0, maneuverable=True),), catalog_hard_body_radius=1.0
    )
    (found,) = run(fleet, fixture, FIXTURE_START, 3 * DAY).events
    outcome = plan_event(
        found.event, Settings(), catalog=fixture, catalog_label="fixture", grid=SMALL
    )
    assert all(burn.rescreen is not None for burn in outcome.burns)
    payload = outcome_to_json(outcome)
    back = outcome_from_json(through_text(payload), outcome.event)
    assert outcome_to_json(back) == payload
    assert back.verdict == outcome.verdict


def test_outcome_without_grid_or_rescreens_round_trips(
    outcome: ManeuverOutcome,
) -> None:
    bare = dataclasses.replace(
        outcome,
        grid=None,
        catalog_label=None,
        burns=tuple(PlannedBurn(burn.option, None) for burn in outcome.burns),
    )
    payload = outcome_to_json(bare)
    back = outcome_from_json(through_text(payload), bare.event)
    assert back.grid is None
    assert back.catalog_label is None
    assert all(burn.rescreen is None for burn in back.burns)
    assert outcome_to_json(back) == payload


def test_nan_cells_are_null(outcome: ManeuverOutcome) -> None:
    assert outcome.grid is not None
    pc = np.array(outcome.grid.pc)
    pc[0, 0] = math.nan
    grid = dataclasses.replace(outcome.grid, pc=pc)
    payload = outcome_to_json(dataclasses.replace(outcome, grid=grid))
    cells = payload["grid"]
    assert isinstance(cells, dict)
    rows = cells["pc"]
    assert isinstance(rows, list)
    assert isinstance(rows[0], list)
    assert rows[0][0] is None
    back = outcome_from_json(through_text(payload), outcome.event)
    assert back.grid is not None
    assert math.isnan(back.grid.pc[0, 0])
    assert np.array_equal(back.grid.pc, pc, equal_nan=True)


CATALOG: Final = CatalogReference(
    "gp.json", (CatalogFileRef("gp.json", Path("gp.json"), "a" * 64),)
)


@pytest.mark.parametrize(
    "changed",
    [
        pytest.param({"settings": Settings(target_pc=1e-6)}, id="target_pc"),
        pytest.param(
            {"settings": Settings(thresholds=Thresholds(1e-6, 1e-4))}, id="green"
        ),
        pytest.param(
            {"settings": Settings(thresholds=Thresholds(1e-7, 1e-3))}, id="red"
        ),
        pytest.param(
            {"settings": Settings(compute=computation("foster"))}, id="method"
        ),
        pytest.param({"settings": Settings(policy="raise")}, id="policy"),
        pytest.param(
            {"settings": Settings(combined_hard_body_radius=5.0)}, id="radius"
        ),
        pytest.param(
            {
                "catalog": CatalogReference(
                    "gp.json", (CatalogFileRef("gp.json", Path("gp.json"), "b" * 64),)
                )
            },
            id="catalog_sha256",
        ),
        pytest.param({"catalog": None}, id="no_catalog"),
        pytest.param({"version": "0.0.0+other"}, id="version"),
    ],
)
def test_key_changes_with(
    changed: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    before = maneuver_key(Settings(), CATALOG)
    if "version" in changed:
        monkeypatch.setattr(keepout, "__version__", changed["version"])
    after = maneuver_key(
        changed.get("settings", Settings()), changed.get("catalog", CATALOG)
    )
    assert after != before


def test_key_ignores_the_catalogs_label_paths_and_file_order() -> None:
    files = (
        CatalogFileRef("a", Path("a.json"), "a" * 64),
        CatalogFileRef("b", Path("b.json"), "b" * 64),
    )
    moved = (
        CatalogFileRef("b2", Path("elsewhere/b.json"), "b" * 64),
        CatalogFileRef("a2", Path("elsewhere/a.json"), "a" * 64),
    )
    assert maneuver_key(Settings(), CatalogReference("one", files)) == maneuver_key(
        Settings(), CatalogReference("two", moved)
    )
    assert maneuver_key(Settings(), None) == maneuver_key(Settings(), None)
    assert len(maneuver_key(Settings(), None)) == 64


def test_key_differs_from_result_key() -> None:
    assert MANEUVER_KIND == "maneuver"
    assert maneuver_key(Settings(), None) != result_key(Settings())


def test_key_of_settings_a_file_cannot_hold_is_a_value_error() -> None:
    settings = Settings(compute=computation("montecarlo", samples=1000))
    with pytest.raises(ValueError, match=r"compute\.options"):
        maneuver_key(settings, None)


def test_broken_payload(outcome: ManeuverOutcome) -> None:
    payload = through_text(outcome_to_json(outcome))
    del payload["burns"]
    with pytest.raises(StoreError, match=r"^cached manoeuvre: missing key\(s\) burns$"):
        outcome_from_json(payload, outcome.event)


def _broken(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Copies of ``payload``, each broken in one place, with what the error
    names."""
    cases: list[tuple[str, dict[str, Any]]] = []

    def broken(match: str, change: Any) -> None:
        copy = json.loads(json.dumps(payload))
        change(copy)
        cases.append((match, copy))

    broken("unknown key", lambda p: p.update(extra=1))
    broken("target_pc", lambda p: p.update(target_pc="x"))
    broken("target_pc", lambda p: p.update(target_pc=True))
    broken("notes", lambda p: p.update(notes=[1]))
    broken("catalog_label", lambda p: p.update(catalog_label=1))
    broken("same length", lambda p: p["grid"].update(pc=[[0.1], [0.1, 0.2]]))
    broken(
        "must not be null",
        lambda p: p["burns"][0]["rescreen"]["window"][0].update(jd1=None),
    )
    broken("burns", lambda p: p.update(burns={}))
    broken("method", lambda p: p["baseline"].update(method="nope"))
    broken("meets_target", lambda p: p["baseline"].update(meets_target=1))
    broken("baseline", lambda p: p.update(baseline=[]))
    broken(
        "catalog_objects",
        lambda p: p["burns"][0]["rescreen"].update(catalog_objects=1.5),
    )
    broken(
        "level",
        lambda p: p["burns"][0]["rescreen"]["new_approaches"][0].update(level="RED"),
    )
    broken(
        "scale",
        lambda p: p["burns"][0]["rescreen"]["window"][0].update(scale="martian"),
    )
    broken("window", lambda p: p["burns"][0]["rescreen"].update(window=[]))
    broken(
        "r",
        lambda p: p["burns"][0]["rescreen"]["new_approaches"][0]["approach"][
            "primary_state"
        ].update(r=[1.0, 2.0]),
    )
    broken(
        "r",
        lambda p: p["burns"][0]["rescreen"]["new_approaches"][0]["approach"][
            "primary_state"
        ].update(r=[1.0, 2.0, None]),
    )
    broken(
        "epoch",
        lambda p: p["burns"][0]["rescreen"]["new_approaches"][0]["approach"][
            "secondary_state"
        ]["epoch"].update(jd1=0.0),
    )
    broken("pc", lambda p: p["grid"].update(pc=[[0.1]]))
    broken("delta_v", lambda p: p["grid"].update(delta_v=[["x"]]))
    return cases


def test_wrong_values_raise_store_errors(outcome: ManeuverOutcome) -> None:
    payload = through_text(outcome_to_json(outcome))
    for match, copy in _broken(payload):
        with pytest.raises(StoreError, match=f"^cached manoeuvre: .*{match}"):
            outcome_from_json(copy, outcome.event)
