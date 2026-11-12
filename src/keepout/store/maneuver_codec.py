"""The JSON of a cached manoeuvre outcome, and the key it is cached under.

A store keeps an event's :class:`~keepout.maneuver.ManeuverOutcome` as JSON
(:func:`outcome_to_json`, :func:`outcome_from_json`): everything but its
event, which the store holds already. Floats are JSON numbers, NaN is
``null``; a time is its scale and two-part Julian date, so it reads back
exactly; a state is its epoch, position and velocity, in GCRF as every
state of a close approach is; a risk level is its
:attr:`~keepout._types.OrderedEnum.key`.

The key it is cached under (:func:`maneuver_key`) is a hash of all that
changes a plan: keepout's version, the operator's settings (thresholds and
target among them, unlike :func:`~keepout.store.codec.result_key`) and the
digests of the catalog's files the burns were re-screened against. Its
``kind`` keeps it apart from every key of a result.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Final, TypeVar, get_args

import numpy as np
from astropy.time import Time

import keepout
from keepout._methods import Method
from keepout.errors import KeepoutError, StoreError
from keepout.maneuver.outcome import ManeuverGrid, ManeuverOutcome, PlannedBurn
from keepout.maneuver.planner import ManeuverOption
from keepout.maneuver.rescreen import NewApproach, RescreenResult
from keepout.risk import RiskLevel
from keepout.screening import CloseApproach
from keepout.settings import settings_to_data
from keepout.state import StateVector

if TYPE_CHECKING:
    from keepout._types import FloatArray, JsonValue
    from keepout.events import ConjunctionEvent
    from keepout.settings import Settings
    from keepout.sources.catalog import CatalogReference

__all__ = ["MANEUVER_KIND", "maneuver_key", "outcome_from_json", "outcome_to_json"]

MANEUVER_KIND: Final = "maneuver"
"""The ``kind`` in the text a manoeuvre key hashes."""

_WHERE: Final = "cached manoeuvre: "

_METHODS: Final[tuple[Method, ...]] = get_args(Method)
_SCALES: Final = ("tai", "tcb", "tcg", "tdb", "tt", "ut1", "utc")
"""The time scales of astropy but ``local``, which has no Julian date."""
_LEVELS: Final = {level.key: level for level in RiskLevel}

_OUTCOME_KEYS: Final = frozenset(
    {"target_pc", "baseline", "notes", "burns", "grid", "catalog_label"}
)
_OPTION_KEYS: Final = frozenset(
    {
        "burn_lead",
        "delta_v",
        "pc",
        "range_at_tca",
        "method",
        "meets_target",
        "notes",
    }
)
_BURN_KEYS: Final = frozenset({"option", "rescreen"})
_GRID_KEYS: Final = frozenset({"burn_leads", "delta_v", "pc", "range_at_tca"})
_RESCREEN_KEYS: Final = frozenset(
    {
        "new_approaches",
        "closer_approaches",
        "event_returns",
        "event_range",
        "window",
        "catalog_objects",
        "secondary_screened",
        "skipped_objects",
    }
)
_NEW_KEYS: Final = frozenset({"approach", "pc", "level"})
_APPROACH_KEYS: Final = frozenset(
    {"primary_number", "secondary_number", "primary_state", "secondary_state"}
)
_STATE_KEYS: Final = frozenset({"epoch", "r", "v"})
_TIME_KEYS: Final = frozenset({"scale", "jd1", "jd2"})

_T = TypeVar("_T")


def maneuver_key(settings: Settings, catalog: CatalogReference | None) -> str:
    """The key of the outcome ``settings`` plan with ``catalog``: the SHA-256
    (hex) of the JSON of keepout's version, ``kind`` :data:`MANEUVER_KIND`,
    the settings as a settings file holds them and the sorted digests of the
    catalog's files.

    The catalog's label and the files' names and paths do not change it: the
    same bytes give the same re-screens.

    Parameters
    ----------
    settings
        The settings of the plan.
    catalog
        The catalog the burns are re-screened against; ``None``: none.

    Raises
    ------
    ValueError
        If a settings file cannot hold ``settings``, as
        :func:`~keepout.settings.settings_to_data`.
    """
    text = json.dumps(
        {
            "keepout": keepout.__version__,
            "kind": MANEUVER_KIND,
            "settings": settings_to_data(settings),
            "catalog": None
            if catalog is None
            else sorted(file.sha256 for file in catalog.files),
        },
        sort_keys=True,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- To JSON ---------------------------------------------------------------


def _number(value: float) -> float | None:
    """``value`` as a float, ``None`` for NaN."""
    number = float(value)
    return None if math.isnan(number) else number


def _array_json(array: FloatArray) -> JsonValue:
    """The nested lists of an array, NaN as ``None``."""
    if array.ndim == 0:
        return _number(float(array))
    return [_array_json(item) for item in array]


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


def _message(where: str, message: str) -> str:
    """The message of a :class:`StoreError` at ``where`` in the JSON."""
    return f"{_WHERE}{where}{': ' if where else ''}{message}"


def _object(value: object, where: str, keys: frozenset[str]) -> Mapping[str, object]:
    """``value``, a JSON object with exactly ``keys``.

    Raises
    ------
    StoreError
        If it is no object, or a key is unknown or missing.
    """
    if not isinstance(value, Mapping):
        raise StoreError(_message(where, f"must be an object, got {value!r}"))
    data: Mapping[str, object] = value
    unknown = sorted(set(data) - keys)
    if unknown:
        raise StoreError(_message(where, f"unknown key(s) {', '.join(unknown)}"))
    missing = sorted(keys - set(data))
    if missing:
        raise StoreError(_message(where, f"missing key(s) {', '.join(missing)}"))
    return data


def _path(where: str, key: str | int) -> str:
    if isinstance(key, int):
        return f"{where}[{key}]"
    return f"{where}.{key}" if where else key


def _float(value: object, where: str) -> float:
    """``value``, a number, or NaN for ``null``.

    Raises
    ------
    StoreError
        If it is neither (a bool is not a number).
    """
    if value is None:
        return math.nan
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise StoreError(_message(where, f"must be a number, got {value!r}"))
    return float(value)


def _integer(value: object, where: str) -> int:
    """``value``, an integer.

    Raises
    ------
    StoreError
        If it is not one (a bool is not one).
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise StoreError(_message(where, f"must be an integer, got {value!r}"))
    return value


def _flag(value: object, where: str) -> bool:
    """``value``, true or false.

    Raises
    ------
    StoreError
        If it is neither.
    """
    if not isinstance(value, bool):
        raise StoreError(_message(where, f"must be true or false, got {value!r}"))
    return value


def _list(value: object, where: str) -> list[object]:
    """``value``, a JSON array.

    Raises
    ------
    StoreError
        If it is not one.
    """
    if not isinstance(value, list):
        raise StoreError(_message(where, f"must be a list, got {value!r}"))
    items: list[object] = value
    return items


def _strings(value: object, where: str) -> tuple[str, ...]:
    """``value``, an array of strings.

    Raises
    ------
    StoreError
        If it is not one.
    """
    items = _list(value, where)
    if not all(isinstance(item, str) for item in items):
        raise StoreError(_message(where, f"must be a list of strings, got {value!r}"))
    return tuple(str(item) for item in items)


def _optional_text(value: object, where: str) -> str | None:
    """``value``, a string or ``None``.

    Raises
    ------
    StoreError
        If it is neither.
    """
    if value is not None and not isinstance(value, str):
        raise StoreError(_message(where, f"must be a string, got {value!r}"))
    return value


def _array(value: object, where: str, ndim: int) -> FloatArray:
    """``value``, nested arrays of numbers (``null`` for NaN) ``ndim`` deep.

    Raises
    ------
    StoreError
        If it is not.
    """
    items = _list(value, where)
    if ndim == 1:
        return np.array(
            [_float(item, _path(where, i)) for i, item in enumerate(items)],
            dtype=np.float64,
        )
    rows = [_array(item, _path(where, i), ndim - 1) for i, item in enumerate(items)]
    if len({row.shape for row in rows}) > 1:
        raise StoreError(_message(where, "rows must have the same length"))
    if not rows:
        return np.empty((0,) * ndim, dtype=np.float64)
    return np.array(rows, dtype=np.float64)


def _items(
    value: object, where: str, read: Callable[[object, str], _T]
) -> tuple[_T, ...]:
    """The items of an array, each read by ``read``.

    Raises
    ------
    StoreError
        If it is no array, or as ``read``.
    """
    return tuple(
        read(item, _path(where, i)) for i, item in enumerate(_list(value, where))
    )


def _built(where: str, build: Callable[[], _T]) -> _T:
    """``build()``, its errors of a value as :class:`StoreError`.

    Raises
    ------
    StoreError
        If it raises a :class:`ValueError` or a :class:`KeepoutError`.
    """
    try:
        return build()
    except (ValueError, KeepoutError) as exc:
        raise StoreError(_message(where, str(exc))) from exc


def _time(value: object, where: str) -> Time:
    data = _object(value, where, _TIME_KEYS)
    scale = data["scale"]
    if not isinstance(scale, str) or scale not in _SCALES:
        raise StoreError(
            _message(_path(where, "scale"), f"unknown time scale {scale!r}")
        )
    jd1 = _float(data["jd1"], _path(where, "jd1"))
    jd2 = _float(data["jd2"], _path(where, "jd2"))
    if not (math.isfinite(jd1) and math.isfinite(jd2)):
        raise StoreError(_message(where, "jd1 and jd2 must not be null"))
    return Time(jd1, jd2, format="jd", scale=scale)


def _state(value: object, where: str) -> StateVector:
    data = _object(value, where, _STATE_KEYS)
    epoch = _time(data["epoch"], _path(where, "epoch"))
    r = _array(data["r"], _path(where, "r"), 1)
    v = _array(data["v"], _path(where, "v"), 1)
    return _built(where, lambda: StateVector(epoch, r, v, "GCRF"))


def _new(value: object, where: str) -> NewApproach:
    data = _object(value, where, _NEW_KEYS)
    at = _path(where, "approach")
    fields = _object(data["approach"], at, _APPROACH_KEYS)
    primary = _integer(fields["primary_number"], _path(at, "primary_number"))
    secondary = _integer(fields["secondary_number"], _path(at, "secondary_number"))
    primary_state = _state(fields["primary_state"], _path(at, "primary_state"))
    secondary_state = _state(fields["secondary_state"], _path(at, "secondary_state"))
    approach = _built(
        at, lambda: CloseApproach(primary, secondary, primary_state, secondary_state)
    )
    level = data["level"]
    if not isinstance(level, str) or level not in _LEVELS:
        raise StoreError(
            _message(
                _path(where, "level"),
                f"must be one of {', '.join(_LEVELS)}, got {level!r}",
            )
        )
    return NewApproach(
        approach=approach,
        pc=_float(data["pc"], _path(where, "pc")),
        level=_LEVELS[level],
    )


def _option(value: object, where: str) -> ManeuverOption:
    data = _object(value, where, _OPTION_KEYS)
    method = data["method"]
    if not isinstance(method, str) or method not in _METHODS:
        raise StoreError(
            _message(
                _path(where, "method"),
                f"must be one of {', '.join(_METHODS)}, got {method!r}",
            )
        )
    return ManeuverOption(
        burn_lead=_float(data["burn_lead"], _path(where, "burn_lead")),
        delta_v=_float(data["delta_v"], _path(where, "delta_v")),
        pc=_float(data["pc"], _path(where, "pc")),
        range_at_tca=_float(data["range_at_tca"], _path(where, "range_at_tca")),
        method=method,
        meets_target=_flag(data["meets_target"], _path(where, "meets_target")),
        notes=_strings(data["notes"], _path(where, "notes")),
    )


def _rescreen(value: object, where: str, option: ManeuverOption) -> RescreenResult:
    data = _object(value, where, _RESCREEN_KEYS)
    window = _items(data["window"], _path(where, "window"), _time)
    if len(window) != 2:
        raise StoreError(
            _message(_path(where, "window"), f"must hold 2 times, got {len(window)}")
        )
    event_range = data["event_range"]
    return RescreenResult(
        option=option,
        new_approaches=_items(
            data["new_approaches"], _path(where, "new_approaches"), _new
        ),
        closer_approaches=_items(
            data["closer_approaches"], _path(where, "closer_approaches"), _new
        ),
        event_returns=_items(
            data["event_returns"], _path(where, "event_returns"), _new
        ),
        event_range=None
        if event_range is None
        else _float(event_range, _path(where, "event_range")),
        window=(window[0], window[1]),
        catalog_objects=_integer(
            data["catalog_objects"], _path(where, "catalog_objects")
        ),
        secondary_screened=_flag(
            data["secondary_screened"], _path(where, "secondary_screened")
        ),
        skipped_objects=_integer(
            data["skipped_objects"], _path(where, "skipped_objects")
        ),
    )


def _burn(value: object, where: str) -> PlannedBurn:
    data = _object(value, where, _BURN_KEYS)
    option = _option(data["option"], _path(where, "option"))
    found = data["rescreen"]
    return PlannedBurn(
        option,
        None if found is None else _rescreen(found, _path(where, "rescreen"), option),
    )


def _grid(value: object, where: str) -> ManeuverGrid:
    data = _object(value, where, _GRID_KEYS)
    leads = _array(data["burn_leads"], _path(where, "burn_leads"), 1)
    changes = _array(data["delta_v"], _path(where, "delta_v"), 1)
    pc = _array(data["pc"], _path(where, "pc"), 2)
    ranges = _array(data["range_at_tca"], _path(where, "range_at_tca"), 2)
    return _built(
        where,
        lambda: ManeuverGrid(
            burn_leads=leads, delta_v=changes, pc=pc, range_at_tca=ranges
        ),
    )


def outcome_from_json(
    data: Mapping[str, object], event: ConjunctionEvent
) -> ManeuverOutcome:
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
        If a key is unknown or missing, a value has the wrong type, or the
        values make no valid outcome; the message, ``cached manoeuvre: ...``,
        names where.
    """
    fields = _object(data, "", _OUTCOME_KEYS)
    grid = fields["grid"]
    return ManeuverOutcome(
        event=event,
        target_pc=_float(fields["target_pc"], "target_pc"),
        baseline=_option(fields["baseline"], "baseline"),
        notes=_strings(fields["notes"], "notes"),
        burns=_items(fields["burns"], "burns", _burn),
        grid=None if grid is None else _grid(grid, "grid"),
        catalog_label=_optional_text(fields["catalog_label"], "catalog_label"),
    )
