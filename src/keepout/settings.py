"""An operator's settings: how Pc is computed, risk thresholds and the
manoeuvre target, read from a TOML file (the tables and keys of
:data:`SETTINGS_KEYS`).

A file with every key:

```toml
[pc]
method = "foster"     # a Pc method, or "auto" (the default)
policy = "repair"     # an invalid covariance: "repair" it, or "raise"
hbr_m = 20.0          # combined hard-body radius [m]; else the CDMs' HBR

[risk]
green_below = 1e-7    # Pc below which an event is green
red_from = 1e-4       # Pc from which an event is red

[maneuver]
target_pc = 1e-5      # Pc a burn must take the event below
```

Every key is optional; a missing one takes the library's default (the
values above but ``method = "auto"`` and no ``hbr_m``). No other key is
allowed, so a typo is an error, not a silent default.
:func:`load_settings` raises :class:`~keepout.errors.SettingsError` naming
the file and the key at fault. :func:`settings_to_data` and
:func:`settings_from_data` turn settings into a settings file's tables
(:class:`SettingsData`) and back, to store them, say, as JSON.
:func:`plan_arguments` gives the arguments of
:func:`~keepout.maneuver.plan_event` that settings hold.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final, NotRequired, TypedDict, TypeVar, cast

from keepout._defaults import TARGET_PC
from keepout._methods import METHOD_CHOICES, MethodChoice
from keepout._validation import (
    positive_float,
    require_choice,
    require_open_probability,
)
from keepout.covariance import POLICIES, Policy
from keepout.errors import SettingsError
from keepout.pc import AUTO, PcComputation
from keepout.risk import DEFAULT_THRESHOLDS, Thresholds

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

__all__ = [
    "SETTINGS_KEYS",
    "ManeuverTable",
    "PcTable",
    "PlanArguments",
    "RiskTable",
    "Settings",
    "SettingsData",
    "load_settings",
    "plan_arguments",
    "settings_from_data",
    "settings_to_data",
]

SETTINGS_KEYS: Final[Mapping[str, frozenset[str]]] = {
    "pc": frozenset({"method", "policy", "hbr_m"}),
    "risk": frozenset({"green_below", "red_from"}),
    "maneuver": frozenset({"target_pc"}),
}
"""The tables of a settings file and the keys of each; all optional."""

_C = TypeVar("_C")


@dataclass(frozen=True, slots=True, kw_only=True)
class Settings:
    """An operator's settings: how Pc is computed, risk is judged and a
    manoeuvre is planned.

    Parameters
    ----------
    compute
        The Pc method and its options.
    policy
        What to do with an invalid covariance.
    combined_hard_body_radius
        The combined hard-body radius [m]; ``None`` for the sum of the
        CDMs' radii.
    thresholds
        The Pc thresholds of the risk levels.
    target_pc
        Pc below which a manoeuvre is an option.

    Raises
    ------
    ValueError
        If ``policy`` is unknown, ``combined_hard_body_radius`` is not
        positive, or ``target_pc`` is not in ``(0, 1)``.
    """

    compute: PcComputation = AUTO
    policy: Policy = "repair"
    combined_hard_body_radius: float | None = None
    thresholds: Thresholds = DEFAULT_THRESHOLDS
    target_pc: float = TARGET_PC

    def __post_init__(self) -> None:
        require_choice(self.policy, POLICIES, "policy")
        if self.combined_hard_body_radius is not None:
            object.__setattr__(
                self,
                "combined_hard_body_radius",
                positive_float(
                    self.combined_hard_body_radius, "combined_hard_body_radius"
                ),
            )
        require_open_probability(self.target_pc, "target_pc")


class PcTable(TypedDict):
    """The ``[pc]`` table of :class:`SettingsData`."""

    method: MethodChoice
    """The Pc method: ``"auto"`` or a method name of
    :data:`~keepout.pc.Method` (:attr:`Settings.compute` without options)."""
    policy: Policy
    """What to do with an invalid covariance: ``"repair"`` it with a
    diagnostic, or ``"raise"`` an error (:attr:`Settings.policy`)."""
    hbr_m: NotRequired[float]
    """The combined hard-body radius [m], positive; only when it is set
    (:attr:`Settings.combined_hard_body_radius`)."""


class RiskTable(TypedDict):
    """The ``[risk]`` table of :class:`SettingsData`."""

    green_below: float
    """Pc below which an event is green
    (:attr:`~keepout.risk.Thresholds.green_below`)."""
    red_from: float
    """Pc from which an event is red, above ``green_below`` and at most 1
    (:attr:`~keepout.risk.Thresholds.red_from`)."""


class ManeuverTable(TypedDict):
    """The ``[maneuver]`` table of :class:`SettingsData`."""

    target_pc: float
    """Pc a burn must take the event below, in (0, 1)
    (:attr:`Settings.target_pc`)."""


class SettingsData(TypedDict):
    """The tables of a settings file, as :func:`settings_to_data` gives
    them."""

    pc: PcTable
    """How Pc is computed."""
    risk: RiskTable
    """The risk thresholds."""
    maneuver: ManeuverTable
    """The manoeuvre target."""


class PlanArguments(TypedDict):
    """The keyword arguments of :func:`~keepout.maneuver.plan_event` that
    settings hold; pass them as ``plan_event(event, **plan_arguments(settings))``."""

    target_pc: float
    """Pc a burn must take the event below (:attr:`Settings.target_pc`)."""
    compute: PcComputation
    """The Pc method and its options (:attr:`Settings.compute`)."""
    policy: Policy
    """What to do with an invalid covariance (:attr:`Settings.policy`)."""
    combined_hard_body_radius: float | None
    """The combined hard-body radius [m]; ``None`` for the sum of the CDM's
    radii (:attr:`Settings.combined_hard_body_radius`)."""
    thresholds: Thresholds
    """The Pc thresholds, whose basis turns a result into the Pc compared
    with ``target_pc`` (:attr:`Settings.thresholds`)."""


def plan_arguments(settings: Settings) -> PlanArguments:
    """The keyword arguments of :func:`~keepout.maneuver.plan_event` that
    ``settings`` give.

    Parameters
    ----------
    settings
        The operator's settings.
    """
    return {
        "target_pc": settings.target_pc,
        "compute": settings.compute,
        "policy": settings.policy,
        "combined_hard_body_radius": settings.combined_hard_body_radius,
        "thresholds": settings.thresholds,
    }


def _table(data: Mapping[str, object], name: str) -> Mapping[str, object]:
    """The table ``name`` of a settings file, empty if it is not there.

    Raises
    ------
    SettingsError
        If it is not a table or has an unknown key.
    """
    found = data.get(name, {})
    if not isinstance(found, dict):
        raise SettingsError(f"{name} must be a table ([{name}]), got {found!r}")
    table = cast("dict[str, object]", found)  # a TOML table's keys are strings
    unknown = sorted(set(table) - SETTINGS_KEYS[name])
    if unknown:
        raise SettingsError(
            f"unknown key(s) {', '.join(f'{name}.{key}' for key in unknown)}; "
            f"expected {', '.join(sorted(SETTINGS_KEYS[name]))}"
        )
    return table


def _number(table: Mapping[str, object], name: str, key: str) -> float:
    """``table[key]``, a number.

    Raises
    ------
    SettingsError
        If it is not one (a bool is not a number).
    """
    value = table[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SettingsError(f"{name}.{key} must be a number, got {value!r}")
    return float(value)


def _choice(
    table: Mapping[str, object], name: str, key: str, choices: tuple[_C, ...]
) -> _C:
    """``table[key]``, one of ``choices``.

    Raises
    ------
    SettingsError
        If it is none of them.
    """
    try:
        return require_choice(table[key], choices, key)
    except ValueError as exc:
        raise SettingsError(f"{name}.{key}: {exc}") from exc


def _checked(key: str, make: Callable[[], _C]) -> _C:
    """``make()``, whose :class:`ValueError` becomes a :class:`SettingsError`
    naming the settings key.

    Raises
    ------
    SettingsError
        If ``make`` raises :class:`ValueError`.
    """
    try:
        return make()
    except ValueError as exc:
        raise SettingsError(f"{key}: {exc}") from exc


def settings_to_data(settings: Settings) -> SettingsData:
    """The tables of a settings file holding ``settings``, the inverse of
    :func:`settings_from_data`.

    Parameters
    ----------
    settings
        The settings; ``compute`` without options or fallback, ``thresholds`` on the
        default basis.

    Returns
    -------
    SettingsData
        The tables of :data:`SETTINGS_KEYS` with every key, but ``pc.hbr_m``
        only when the radius is set.

    Raises
    ------
    ValueError
        If the settings file cannot hold ``settings``: ``compute`` has
        options or a fallback, or ``thresholds.basis`` is not the default's;
        the message names the field.
    """
    unexpressible: dict[str, object] = {
        "compute.options": dict(settings.compute.options) or None,
        "compute.fallback": settings.compute.fallback,
        "thresholds.basis": (
            None
            if settings.thresholds.basis == DEFAULT_THRESHOLDS.basis
            else settings.thresholds.basis
        ),
    }
    for field, value in unexpressible.items():
        if value is not None:
            raise ValueError(f"a settings file cannot hold {field}, got {value!r}")
    pc: PcTable = {"method": settings.compute.method, "policy": settings.policy}
    if settings.combined_hard_body_radius is not None:
        pc["hbr_m"] = settings.combined_hard_body_radius
    return {
        "pc": pc,
        "risk": {
            "green_below": settings.thresholds.green_below,
            "red_from": settings.thresholds.red_from,
        },
        "maneuver": {"target_pc": settings.target_pc},
    }


def settings_from_data(data: Mapping[str, object]) -> Settings:
    """The settings of a settings file's tables (its parsed TOML); a missing
    key keeps the default of :class:`Settings`, which checks the ranges.

    Parameters
    ----------
    data
        The tables of :data:`SETTINGS_KEYS`, as :func:`tomllib.loads` or
        :func:`settings_to_data` gives them.

    Returns
    -------
    Settings
        The settings.

    Raises
    ------
    SettingsError
        If a key is unknown or a value is wrong; the message names the key.
    """
    unknown = sorted(set(data) - set(SETTINGS_KEYS))
    if unknown:
        raise SettingsError(
            f"unknown key(s) {', '.join(unknown)}; expected the tables "
            f"{', '.join(f'[{name}]' for name in SETTINGS_KEYS)}"
        )
    pc, risk, maneuver = (_table(data, name) for name in SETTINGS_KEYS)
    settings = Settings()
    if "method" in pc:
        compute = PcComputation(_choice(pc, "pc", "method", METHOD_CHOICES))
        settings = replace(settings, compute=compute)
    if "policy" in pc:
        settings = replace(settings, policy=_choice(pc, "pc", "policy", POLICIES))
    if "hbr_m" in pc:
        radius = _number(pc, "pc", "hbr_m")
        settings = _checked(
            "pc.hbr_m", lambda: replace(settings, combined_hard_body_radius=radius)
        )
    if risk:
        green = _number(risk, "risk", "green_below") if "green_below" in risk else None
        red = _number(risk, "risk", "red_from") if "red_from" in risk else None
        thresholds = _checked(
            "risk",
            lambda: replace(
                settings.thresholds,
                green_below=settings.thresholds.green_below if green is None else green,
                red_from=settings.thresholds.red_from if red is None else red,
            ),
        )
        settings = replace(settings, thresholds=thresholds)
    if "target_pc" in maneuver:
        target_pc = _number(maneuver, "maneuver", "target_pc")
        settings = _checked(
            "maneuver.target_pc", lambda: replace(settings, target_pc=target_pc)
        )
    return settings


def load_settings(path: Path) -> Settings:
    """The settings of a settings file (UTF-8 TOML; see the module's
    description).

    Parameters
    ----------
    path
        The settings file.

    Returns
    -------
    Settings
        The file's settings, with the library's default for every key the
        file leaves out.

    Raises
    ------
    SettingsError
        If the file is not UTF-8 TOML, a key is unknown, or a value has the
        wrong type or is out of range; the message starts with the file's
        path and names the key.
    OSError
        If the file cannot be read.

    Examples
    --------
    ```python
    from keepout.settings import load_settings

    settings = load_settings("settings.toml")  # the file of the module's example
    print(settings.compute.method, settings.combined_hard_body_radius)
    # foster 20.0
    print(settings.thresholds.red_from, settings.target_pc)
    # 0.0001 1e-05
    ```
    """
    raw = Path(path).read_bytes()
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise SettingsError(f"{path}: not UTF-8 text: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise SettingsError(f"{path}: not TOML: {exc}") from exc
    try:
        return settings_from_data(data)
    except SettingsError as exc:
        raise SettingsError(f"{path}: {exc}") from exc
