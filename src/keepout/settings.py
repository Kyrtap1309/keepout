"""An operator's settings: how Pc is computed, risk thresholds and the
manoeuvre target, read from a TOML file (the tables and keys of
:data:`SETTINGS_KEYS`; an example is in the command line's documentation,
under Settings).

Every key is optional; a missing one takes the library's default. No other
key is allowed, so a typo is an error, not a silent default.
:func:`load_settings` raises :class:`~keepout.errors.SettingsError` naming
the file and the key at fault.
"""

from __future__ import annotations

import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, TypeVar

from keepout._defaults import TARGET_PC
from keepout._methods import METHOD_CHOICES
from keepout._validation import (
    positive_float,
    require_choice,
    require_open_probability,
)
from keepout.covariance import POLICIES, Policy
from keepout.errors import SettingsError
from keepout.pc import AUTO, PcComputation
from keepout.risk import DEFAULT_THRESHOLDS, Thresholds

__all__ = ["SETTINGS_KEYS", "Settings", "load_settings"]

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


def _table(data: Mapping[str, object], name: str) -> Mapping[str, object]:
    """The table ``name`` of a settings file, empty if it is not there.

    Raises
    ------
    SettingsError
        If it is not a table or has an unknown key.
    """
    table = data.get(name, {})
    if not isinstance(table, dict):
        raise SettingsError(f"{name} must be a table ([{name}]), got {table!r}")
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


def _settings(data: Mapping[str, object]) -> Settings:
    """The settings of a settings file's TOML; a missing key keeps the
    default of :class:`Settings`, which checks the ranges.

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

    Raises
    ------
    SettingsError
        If the file is not UTF-8 TOML, a key is unknown, or a value has the
        wrong type or is out of range; the message starts with the file's
        path and names the key.
    OSError
        If the file cannot be read.
    """
    raw = Path(path).read_bytes()
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise SettingsError(f"{path}: not UTF-8 text: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise SettingsError(f"{path}: not TOML: {exc}") from exc
    try:
        return _settings(data)
    except SettingsError as exc:
        raise SettingsError(f"{path}: {exc}") from exc
