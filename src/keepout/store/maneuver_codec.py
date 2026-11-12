"""The JSON of a cached manoeuvre outcome.

A store keeps an event's :class:`~keepout.maneuver.ManeuverOutcome` as JSON
(:func:`outcome_to_json`, :func:`outcome_from_json`). The JSON holds
everything but the event, which the store holds already. Floats are JSON
numbers, and NaN is ``null``. A time is its scale and two-part Julian date,
so it reads back exactly. A state is its epoch, position and velocity, in
GCRF as every state of a close approach is. A risk level is its
:attr:`~keepout._types.OrderedEnum.key`.

Only keepout writes this JSON, under a key that holds keepout's version
(:func:`~keepout.store.codec.maneuver_key`). So the reader checks little:
JSON it cannot read is a :class:`~keepout.errors.StoreError` that does not
say where.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final, TypeVar, cast, get_args

import numpy as np
from astropy.time import Time

from keepout._methods import Method
from keepout._validation import require_choice
from keepout.errors import KeepoutError, StoreError
from keepout.maneuver.outcome import ManeuverGrid, ManeuverOutcome, PlannedBurn
from keepout.maneuver.planner import ManeuverOption
from keepout.maneuver.rescreen import NewApproach, RescreenResult
from keepout.risk import RiskLevel
from keepout.screening import CloseApproach
from keepout.state import StateVector

if TYPE_CHECKING:
    from collections.abc import Mapping

    from keepout._types import FloatArray, JsonValue
    from keepout.events import ConjunctionEvent

__all__ = ["outcome_from_json", "outcome_to_json"]

_METHODS: Final[tuple[Method, ...]] = get_args(Method)
_SCALES: Final = ("tai", "tcb", "tcg", "tdb", "tt", "ut1", "utc")
"""The time scales of astropy but ``local``, which has no Julian date."""
_LEVELS: Final = {level.key: level for level in RiskLevel}

_UNREADABLE: Final = (
    KeyError,
    TypeError,
    ValueError,
    IndexError,
    OverflowError,
    KeepoutError,
)
"""What reading JSON that keepout did not write may raise."""

_T = TypeVar("_T")


# --- To JSON ---------------------------------------------------------------


def _number(value: float) -> float | None:
    """``value`` as a float, ``None`` for NaN."""
    number = float(value)
    return None if math.isnan(number) else number


def _array_json(array: FloatArray) -> JsonValue:
    """The nested lists of an array, NaN as ``None``."""
    cells = array.astype(object)
    cells[np.isnan(array)] = None
    return cast("JsonValue", cells.tolist())  # numpy types tolist() as Any


def _time_json(time: Time) -> dict[str, JsonValue]:
    return {"scale": time.scale, "jd1": float(time.jd1), "jd2": float(time.jd2)}


def _state_json(state: StateVector) -> dict[str, JsonValue]:
    return {
        "epoch": _time_json(state.epoch),
        "r": _array_json(state.r),
        "v": _array_json(state.v),
    }


def _option_json(option: ManeuverOption) -> dict[str, JsonValue]:
    return {
        "burn_lead": _number(option.burn_lead),
        "delta_v": _number(option.delta_v),
        "pc": _number(option.pc),
        "range_at_tca": _number(option.range_at_tca),
        "method": option.method,
        "meets_target": option.meets_target,
        "notes": list(option.notes),
    }


def _new_json(item: NewApproach) -> dict[str, JsonValue]:
    approach = item.approach
    return {
        "approach": {
            "primary_number": approach.primary_number,
            "secondary_number": approach.secondary_number,
            "primary_state": _state_json(approach.primary_state),
            "secondary_state": _state_json(approach.secondary_state),
        },
        "pc": _number(item.pc),
        "level": item.level.key,
    }


def _rescreen_json(result: RescreenResult) -> dict[str, JsonValue]:
    return {
        "new_approaches": [_new_json(item) for item in result.new_approaches],
        "closer_approaches": [_new_json(item) for item in result.closer_approaches],
        "event_returns": [_new_json(item) for item in result.event_returns],
        "event_range": None
        if result.event_range is None
        else _number(result.event_range),
        "window": [_time_json(time) for time in result.window],
        "catalog_objects": result.catalog_objects,
        "secondary_screened": result.secondary_screened,
        "skipped_objects": result.skipped_objects,
    }


def _grid_json(grid: ManeuverGrid) -> dict[str, JsonValue]:
    return {
        "burn_leads": _array_json(grid.burn_leads),
        "delta_v": _array_json(grid.delta_v),
        "pc": _array_json(grid.pc),
        "range_at_tca": _array_json(grid.range_at_tca),
    }


def outcome_to_json(outcome: ManeuverOutcome) -> dict[str, JsonValue]:
    """The JSON of an outcome, without its event.

    The JSON holds no NaN, so ``json.dumps(..., allow_nan=False)`` takes it.
    A re-screen's burn is the burn it belongs to, and is not repeated.

    Parameters
    ----------
    outcome
        The outcome.
    """
    return {
        "target_pc": _number(outcome.target_pc),
        "baseline": _option_json(outcome.baseline),
        "notes": list(outcome.notes),
        "burns": [
            {
                "option": _option_json(burn.option),
                "rescreen": None
                if burn.rescreen is None
                else _rescreen_json(burn.rescreen),
            }
            for burn in outcome.burns
        ],
        "grid": None if outcome.grid is None else _grid_json(outcome.grid),
        "catalog_label": outcome.catalog_label,
    }


# --- From JSON -------------------------------------------------------------
# Each reader raises one of _UNREADABLE on JSON it cannot read.


def _of(value: object, kind: type[_T]) -> _T:
    """``value``, of type ``kind``; a bool is of no type but bool.

    Raises
    ------
    TypeError
        If it is not.
    """
    if not isinstance(value, kind) or (isinstance(value, bool) and kind is not bool):
        raise TypeError(f"expected a {kind.__name__}, got {value!r}")
    return value


def _object(value: object) -> Mapping[str, object]:
    """``value``, a JSON object."""
    # A JSON object's keys are strings.
    return cast("Mapping[str, object]", _of(value, dict))


def _list(value: object) -> list[object]:
    """``value``, a JSON array."""
    return cast("list[object]", _of(value, list))


def _float(value: object) -> float:
    """``value``, a number, or NaN for ``null``."""
    if value is None:
        return math.nan
    return float(_of(value, float if isinstance(value, float) else int))


def _texts(value: object) -> tuple[str, ...]:
    """``value``, a list of strings."""
    return tuple(_of(item, str) for item in _list(value))


def _array(value: object, ndim: int) -> FloatArray:
    """``value``, nested lists of numbers (``null`` for NaN) ``ndim`` deep."""
    rows = _list(value)
    if not rows:
        return np.empty((0,) * ndim, dtype=np.float64)
    if ndim == 1:
        return np.array([_float(item) for item in rows], dtype=np.float64)
    return np.array([_array(row, ndim - 1) for row in rows], dtype=np.float64)


def _time(value: object) -> Time:
    data = _object(value)
    scale = require_choice(data["scale"], _SCALES, "time scale")
    jd1, jd2 = _float(data["jd1"]), _float(data["jd2"])
    if not (math.isfinite(jd1) and math.isfinite(jd2)):
        raise ValueError("a time's jd1 and jd2 must be numbers")
    return Time(jd1, jd2, format="jd", scale=scale)


def _state(value: object) -> StateVector:
    data = _object(value)
    epoch = _time(data["epoch"])
    return StateVector(epoch, _array(data["r"], 1), _array(data["v"], 1), "GCRF")


def _new(value: object) -> NewApproach:
    data = _object(value)
    approach = _object(data["approach"])
    level = require_choice(data["level"], tuple(_LEVELS), "risk level")
    return NewApproach(
        approach=CloseApproach(
            _of(approach["primary_number"], int),
            _of(approach["secondary_number"], int),
            _state(approach["primary_state"]),
            _state(approach["secondary_state"]),
        ),
        pc=_float(data["pc"]),
        level=_LEVELS[level],
    )


def _option(value: object) -> ManeuverOption:
    data = _object(value)
    return ManeuverOption(
        burn_lead=_float(data["burn_lead"]),
        delta_v=_float(data["delta_v"]),
        pc=_float(data["pc"]),
        range_at_tca=_float(data["range_at_tca"]),
        method=require_choice(data["method"], _METHODS, "method"),
        meets_target=_of(data["meets_target"], bool),
        notes=_texts(data["notes"]),
    )


def _rescreen(value: object, option: ManeuverOption) -> RescreenResult:
    data = _object(value)
    start, end = (_time(item) for item in _list(data["window"]))
    event_range = data["event_range"]
    return RescreenResult(
        option=option,
        new_approaches=tuple(_new(item) for item in _list(data["new_approaches"])),
        closer_approaches=tuple(
            _new(item) for item in _list(data["closer_approaches"])
        ),
        event_returns=tuple(_new(item) for item in _list(data["event_returns"])),
        event_range=None if event_range is None else _float(event_range),
        window=(start, end),
        catalog_objects=_of(data["catalog_objects"], int),
        secondary_screened=_of(data["secondary_screened"], bool),
        skipped_objects=_of(data["skipped_objects"], int),
    )


def _burn(value: object) -> PlannedBurn:
    data = _object(value)
    option = _option(data["option"])
    found = data["rescreen"]
    return PlannedBurn(option, None if found is None else _rescreen(found, option))


def _grid(value: object) -> ManeuverGrid:
    data = _object(value)
    return ManeuverGrid(
        burn_leads=_array(data["burn_leads"], 1),
        delta_v=_array(data["delta_v"], 1),
        pc=_array(data["pc"], 2),
        range_at_tca=_array(data["range_at_tca"], 2),
    )


def _outcome(value: object, event: ConjunctionEvent) -> ManeuverOutcome:
    data = _object(value)
    grid, label = data["grid"], data["catalog_label"]
    return ManeuverOutcome(
        event=event,
        target_pc=_float(data["target_pc"]),
        baseline=_option(data["baseline"]),
        notes=_texts(data["notes"]),
        burns=tuple(_burn(item) for item in _list(data["burns"])),
        grid=None if grid is None else _grid(grid),
        catalog_label=None if label is None else _of(label, str),
    )


def outcome_from_json(data: object, event: ConjunctionEvent) -> ManeuverOutcome:
    """The outcome of its JSON (:func:`outcome_to_json`) and its event.

    Parameters
    ----------
    data
        The JSON, as :func:`json.loads` returns it.
    event
        The event planned, which the JSON does not hold.

    Raises
    ------
    StoreError
        If the JSON makes no outcome: a key is missing, or a value is of the
        wrong type or out of range.
    """
    try:
        return _outcome(data, event)
    except _UNREADABLE as exc:
        raise StoreError(f"cached manoeuvre unreadable: {exc}") from exc
