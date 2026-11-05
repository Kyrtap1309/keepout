"""An operator's settings: how Pc is computed, risk thresholds and the
manoeuvre target, read from a TOML file.

Every key is optional; a missing one takes the library's default. For
example::

    [pc]
    method = "foster"     # a Pc method, or "auto"
    policy = "repair"     # what to do with an invalid covariance
    hbr_m = 20.0          # combined hard-body radius [m]; else the CDMs'

    [risk]
    green_below = 1e-7
    red_from = 1e-4

    [maneuver]
    target_pc = 1e-5

No other key is allowed, so a typo is an error, not a silent default.
:func:`load_settings` raises :class:`~keepout.errors.StoreError` naming the
file and the key at fault.
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeVar

from keepout._defaults import TARGET_PC
from keepout._methods import METHOD_CHOICES
from keepout._validation import positive_float, require_choice
from keepout.covariance import POLICIES, Policy
from keepout.errors import StoreError
from keepout.pc import AUTO, PcComputation
from keepout.risk import DEFAULT_THRESHOLDS, Thresholds
from keepout.store.codec import result_key

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
    """An operator's settings for assessing the events of a store.

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
        if not 0.0 < self.target_pc < 1.0:
            raise ValueError(f"target_pc must be in (0, 1), got {self.target_pc}")

    @property
    def result_key(self) -> str:
        """The key of the observations these settings compute (see
        :func:`keepout.store.codec.result_key`); thresholds and target do not
        change it."""
        return result_key(
            compute=self.compute,
            policy=self.policy,
            combined_hard_body_radius=self.combined_hard_body_radius,
        )


def _table(data: Mapping[str, object], name: str) -> Mapping[str, object]:
    """The table ``name`` of a settings file, empty if it is not there.

    Raises
    ------
    StoreError
        If it is not a table or has an unknown key.
    """
    table = data.get(name, {})
    if not isinstance(table, dict):
        raise StoreError(f"{name} must be a table ([{name}]), got {table!r}")
    unknown = sorted(set(table) - SETTINGS_KEYS[name])
    if unknown:
        raise StoreError(
            f"unknown key(s) {', '.join(f'{name}.{key}' for key in unknown)}; "
            f"expected {', '.join(sorted(SETTINGS_KEYS[name]))}"
        )
    return table


def _number(table: Mapping[str, object], name: str, key: str) -> float:
    """``table[key]``, a number.

    Raises
    ------
    StoreError
        If it is not one (a bool is not a number).
    """
    value = table[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise StoreError(f"{name}.{key} must be a number, got {value!r}")
    return float(value)


def _choice(
    table: Mapping[str, object], name: str, key: str, choices: tuple[_C, ...]
) -> _C:
    """``table[key]``, one of ``choices``.

    Raises
    ------
    StoreError
        If it is none of them.
    """
    try:
        return require_choice(table[key], choices, key)
    except ValueError as exc:
        raise StoreError(f"{name}.{key}: {exc}") from exc


def _settings(data: Mapping[str, object]) -> Settings:
    """The settings of a settings file's TOML.

    Raises
    ------
    StoreError
        If a key is unknown or a value is wrong; the message names the key.
    """
    unknown = sorted(set(data) - set(SETTINGS_KEYS))
    if unknown:
        raise StoreError(
            f"unknown key(s) {', '.join(unknown)}; expected the tables "
            f"{', '.join(f'[{name}]' for name in SETTINGS_KEYS)}"
        )
    pc, risk, maneuver = (_table(data, name) for name in SETTINGS_KEYS)
    default = Settings()
    compute = default.compute
    policy: Policy = default.policy
    radius = default.combined_hard_body_radius
    if "method" in pc:
        compute = PcComputation(_choice(pc, "pc", "method", METHOD_CHOICES))
    if "policy" in pc:
        policy = _choice(pc, "pc", "policy", POLICIES)
    if "hbr_m" in pc:
        radius = _number(pc, "pc", "hbr_m")
        if not 0.0 < radius < math.inf:
            raise StoreError(f"pc.hbr_m must be positive, got {radius}")
    try:
        thresholds = Thresholds(
            green_below=_number(risk, "risk", "green_below")
            if "green_below" in risk
            else default.thresholds.green_below,
            red_from=_number(risk, "risk", "red_from")
            if "red_from" in risk
            else default.thresholds.red_from,
            basis=default.thresholds.basis,
        )
    except ValueError as exc:
        raise StoreError(f"risk: {exc}") from exc
    target_pc = default.target_pc
    if "target_pc" in maneuver:
        target_pc = _number(maneuver, "maneuver", "target_pc")
        if not 0.0 < target_pc < 1.0:
            raise StoreError(f"maneuver.target_pc must be in (0, 1), got {target_pc}")
    return Settings(
        compute=compute,
        policy=policy,
        combined_hard_body_radius=radius,
        thresholds=thresholds,
        target_pc=target_pc,
    )


def load_settings(path: Path) -> Settings:
    """The settings of a settings file (UTF-8 TOML; see the module's
    description).

    Parameters
    ----------
    path
        The settings file.

    Raises
    ------
    StoreError
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
        raise StoreError(f"{path}: not UTF-8 text: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise StoreError(f"{path}: not TOML: {exc}") from exc
    try:
        return _settings(data)
    except StoreError as exc:
        raise StoreError(f"{path}: {exc}") from exc
