"""The JSON of a manoeuvre outcome, and the key it is cached under."""

import contextlib
import dataclasses
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final

import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time
from builders import CROSSING_RADIUS as RADIUS
from builders import CROSSING_TCA as TCA
from builders import SMALL_GRID, leo_crossing
from hypothesis import given
from hypothesis import strategies as st
from inline_snapshot import snapshot
from screening_data import FIXTURE_START, load_fixture

import keepout
from keepout._types import JsonValue
from keepout.errors import StoreError
from keepout.maneuver import (
    ManeuverOutcome,
    NewApproach,
    PlannedBurn,
    RescreenResult,
    plan_event,
)
from keepout.pc import computation
from keepout.risk import RiskLevel, Thresholds
from keepout.screening import CloseApproach
from keepout.settings import Settings
from keepout.sources.catalog import CatalogFileRef, CatalogReference
from keepout.state import StateVector
from keepout.store.codec import MANEUVER_KIND, maneuver_key, result_key
from keepout.store.maneuver_codec import outcome_from_json, outcome_to_json
from keepout.synthetic import Fleet, FleetSatellite
from keepout.synthetic.operator import run
from keepout.units import DAY


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
    planned = plan_event(leo_crossing(), grid=SMALL_GRID)
    assert planned.burns
    return with_rescreens(planned)


def through_text(payload: dict[str, JsonValue]) -> Any:
    """``payload`` through strict JSON text, as a store keeps it."""
    return json.loads(json.dumps(payload, allow_nan=False))


def _same_float(actual: float | None, expected: float | None) -> bool:
    """Whether two floats are equal, NaN equal to NaN."""
    if actual is None or expected is None:
        return actual is expected
    return actual == expected or (math.isnan(actual) and math.isnan(expected))


def assert_same_time(actual: Time, expected: Time) -> None:
    assert (actual.scale, actual.jd1, actual.jd2) == (
        expected.scale,
        expected.jd1,
        expected.jd2,
    )


def assert_same_state(actual: StateVector, expected: StateVector) -> None:
    assert_same_time(actual.epoch, expected.epoch)
    assert actual.frame == expected.frame
    assert np.array_equal(actual.r, expected.r)
    assert np.array_equal(actual.v, expected.v)


def assert_same_approaches(
    actual: Sequence[NewApproach], expected: Sequence[NewApproach]
) -> None:
    """Approaches equal field by field: :class:`NewApproach` and
    :class:`CloseApproach` compare by identity."""
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        assert got.level is want.level
        assert _same_float(got.pc, want.pc)
        assert (got.approach.primary_number, got.approach.secondary_number) == (
            want.approach.primary_number,
            want.approach.secondary_number,
        )
        assert_same_state(got.approach.primary_state, want.approach.primary_state)
        assert_same_state(got.approach.secondary_state, want.approach.secondary_state)


def assert_same_rescreen(
    actual: RescreenResult | None, expected: RescreenResult | None
) -> None:
    """Re-screens equal field by field: :class:`RescreenResult` compares by
    identity."""
    if expected is None:
        assert actual is None
        return
    assert actual is not None
    assert actual.option == expected.option
    assert_same_approaches(actual.new_approaches, expected.new_approaches)
    assert_same_approaches(actual.closer_approaches, expected.closer_approaches)
    assert_same_approaches(actual.event_returns, expected.event_returns)
    assert _same_float(actual.event_range, expected.event_range)
    for got, want in zip(actual.window, expected.window, strict=True):
        assert_same_time(got, want)
    assert (
        actual.catalog_objects,
        actual.secondary_screened,
        actual.skipped_objects,
    ) == (
        expected.catalog_objects,
        expected.secondary_screened,
        expected.skipped_objects,
    )


def assert_same_outcome(actual: ManeuverOutcome, expected: ManeuverOutcome) -> None:
    """Outcomes equal field by field, against the original objects."""
    assert actual.event is expected.event
    assert actual.target_pc == expected.target_pc
    assert actual.baseline == expected.baseline
    assert actual.notes == expected.notes
    assert actual.catalog_label == expected.catalog_label
    assert len(actual.burns) == len(expected.burns)
    for got, want in zip(actual.burns, expected.burns, strict=True):
        assert got.option == want.option
        assert_same_rescreen(got.rescreen, want.rescreen)
    if expected.grid is None:
        assert actual.grid is None
        return
    assert actual.grid is not None
    for name in ("burn_leads", "delta_v", "pc", "range_at_tca"):
        assert np.array_equal(
            getattr(actual.grid, name), getattr(expected.grid, name), equal_nan=True
        )


def test_outcome_round_trip(outcome: ManeuverOutcome) -> None:
    payload = outcome_to_json(outcome)
    back = outcome_from_json(through_text(payload), outcome.event)
    assert_same_outcome(back, outcome)
    assert outcome_to_json(back) == payload
    assert back.verdict == outcome.verdict
    first = back.burns[0].rescreen
    assert first is not None
    assert first.new_approaches[0].approach.primary_state.epoch.scale == "tdb"
    assert first.window[0].scale == "utc"


@pytest.mark.slow
def test_outcome_of_a_real_rescreen_round_trips() -> None:
    """A plan of a synthetic event on the committed GP data, both burns
    re-screened against it (10 s)."""
    fixture = load_fixture()
    fleet = Fleet(
        (FleetSatellite(52939, 5.0, maneuverable=True),), catalog_hard_body_radius=1.0
    )
    (found,) = run(fleet, fixture, FIXTURE_START, 3 * DAY).events
    outcome = plan_event(
        found.event, catalog=fixture, catalog_label="fixture", grid=SMALL_GRID
    )
    assert outcome.burns
    assert all(burn.rescreen is not None for burn in outcome.burns)
    payload = outcome_to_json(outcome)
    back = outcome_from_json(through_text(payload), outcome.event)
    assert_same_outcome(back, outcome)
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
    assert_same_outcome(back, bare)
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


def test_key_is_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    # A change of the key's text makes every cached plan miss: on purpose only.
    monkeypatch.setattr(keepout, "__version__", "0.0.0+pinned")
    assert maneuver_key(Settings(), None) == snapshot(
        "d5a675fe894d83ea08b903c85d4d1ff69d8135964df9cf086d28e50a11e3271a"
    )
    assert maneuver_key(Settings(), CATALOG) == snapshot(
        "65f965b7ad89cc7e7f301a9a463ac7f738d359159ecbe4cb9564bdf0578940a5"
    )


def test_key_differs_from_result_key() -> None:
    assert MANEUVER_KIND == "maneuver"
    assert maneuver_key(Settings(), None) != result_key(Settings())


def test_key_of_settings_a_file_cannot_hold_is_a_value_error() -> None:
    settings = Settings(compute=computation("montecarlo", samples=1000))
    with pytest.raises(ValueError, match=r"compute\.options"):
        maneuver_key(settings, None)


def _rescreen_of(payload: Any) -> Any:
    return payload["burns"][0]["rescreen"]


def _state_of(payload: Any) -> Any:
    return _rescreen_of(payload)["new_approaches"][0]["approach"]["primary_state"]


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda p: p.pop("burns"), id="missing_key"),
        pytest.param(lambda p: p.update(target_pc="x"), id="text_for_number"),
        pytest.param(lambda p: p.update(target_pc=True), id="bool_for_number"),
        pytest.param(lambda p: p["baseline"].update(method="nope"), id="method"),
        pytest.param(lambda p: p.update(baseline=[]), id="list_for_object"),
        pytest.param(
            lambda p: p["grid"].update(pc=[[0.1], [0.1, 0.2]]), id="ragged_grid"
        ),
        pytest.param(lambda p: p["grid"].update(pc=[[0.1]]), id="grid_shape"),
        pytest.param(
            lambda p: _rescreen_of(p)["window"][0].update(scale="martian"),
            id="time_scale",
        ),
        pytest.param(lambda p: _rescreen_of(p).update(window=[]), id="window"),
        pytest.param(lambda p: _state_of(p).update(r=[1.0, 2.0]), id="state"),
    ],
)
def test_wrong_values_raise_store_errors(
    outcome: ManeuverOutcome, change: Callable[[Any], object]
) -> None:
    payload = through_text(outcome_to_json(outcome))
    change(payload)
    with pytest.raises(StoreError, match=r"^cached manoeuvre unreadable: "):
        outcome_from_json(payload, outcome.event)


JSON_VALUES: Final = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats() | st.text(),
    lambda children: st.lists(children) | st.dictionaries(st.text(), children),
    max_leaves=10,
)
"""JSON values, as :func:`json.loads` gives them (NaN and infinities too)."""


@given(JSON_VALUES)
def test_any_json_raises_only_store_errors(outcome: ManeuverOutcome, data: Any) -> None:
    with pytest.raises(StoreError):
        outcome_from_json(data, outcome.event)


@given(st.data())
def test_a_payload_changed_anywhere_raises_only_store_errors(
    outcome: ManeuverOutcome, data: st.DataObject
) -> None:
    payload = through_text(outcome_to_json(outcome))
    node = payload
    while True:
        keys = list(node) if isinstance(node, dict) else list(range(len(node)))
        key = data.draw(st.sampled_from(keys))
        child = node[key]
        if isinstance(child, dict | list) and child and data.draw(st.booleans()):
            node = child
            continue
        node[key] = data.draw(JSON_VALUES)
        break
    with contextlib.suppress(StoreError):
        outcome_from_json(payload, outcome.event)
